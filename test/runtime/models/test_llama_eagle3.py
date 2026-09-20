"""CPU-only tests for the EAGLE3 llama draft model variants.

Covers the torchspec checkpoint layout (e.g. Inferact/MiniMax-M3-EAGLE3):
per-aux-state ``fc_norm`` RMSNorms and the ``norm_output`` aux convention.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

import tokenspeed.runtime.models.llama_eagle3 as llama_eagle3
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.models.llama_eagle3 import (
    LlamaAttention,
    LlamaForCausalLMEagle3,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

_HIDDEN = 32
_INTERMEDIATE = 64
_VOCAB = 64


def _draft_config(**overrides) -> LlamaConfig:
    values = dict(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        draft_vocab_size=_VOCAB,
    )
    values.update(overrides)
    return LlamaConfig(**values)


def _tp1_mapping() -> Mapping:
    return Mapping(
        rank=0,
        world_size=1,
        attn_tp_size=1,
        attn_cp_size=1,
        attn_dp_size=1,
        dense_tp_size=1,
        dense_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        moe_dp_size=1,
        nprocs_per_node=1,
        nnodes=1,
    )


def _build_model(
    monkeypatch: pytest.MonkeyPatch, config: LlamaConfig
) -> LlamaForCausalLMEagle3:
    mapping = _tp1_mapping()
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    monkeypatch.setitem(global_server_args_dict, "max_model_len", 256)
    monkeypatch.setitem(global_server_args_dict, "mapping", mapping)
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 256)

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            return LlamaForCausalLMEagle3(config, mapping)
    finally:
        torch.set_default_dtype(old_dtype)


def _torchspec_checkpoint_weights() -> list[tuple[str, torch.Tensor]]:
    """Tensor names/shapes of a torchspec EAGLE3 checkpoint (tiny config)."""

    def meta(*shape: int) -> torch.Tensor:
        return torch.empty(*shape, dtype=torch.bfloat16, device="meta")

    return [
        ("embed_tokens.weight", meta(_VOCAB, _HIDDEN)),
        ("fc.weight", meta(_HIDDEN, 3 * _HIDDEN)),
        ("fc_norm.0.weight", meta(_HIDDEN)),
        ("fc_norm.1.weight", meta(_HIDDEN)),
        ("fc_norm.2.weight", meta(_HIDDEN)),
        ("layers.0.hidden_norm.weight", meta(_HIDDEN)),
        ("layers.0.input_layernorm.weight", meta(_HIDDEN)),
        ("layers.0.mlp.down_proj.weight", meta(_HIDDEN, _INTERMEDIATE)),
        ("layers.0.mlp.gate_proj.weight", meta(_INTERMEDIATE, _HIDDEN)),
        ("layers.0.mlp.up_proj.weight", meta(_INTERMEDIATE, _HIDDEN)),
        ("layers.0.post_attention_layernorm.weight", meta(_HIDDEN)),
        ("layers.0.self_attn.q_proj.weight", meta(_HIDDEN, 2 * _HIDDEN)),
        ("layers.0.self_attn.k_proj.weight", meta(_HIDDEN, 2 * _HIDDEN)),
        ("layers.0.self_attn.v_proj.weight", meta(_HIDDEN, 2 * _HIDDEN)),
        ("layers.0.self_attn.o_proj.weight", meta(_HIDDEN, _HIDDEN)),
        ("lm_head.weight", meta(_VOCAB, _HIDDEN)),
        ("norm.weight", meta(_HIDDEN)),
    ]


def test_eagle3_default_config_has_no_fc_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(monkeypatch, _draft_config())

    assert model.model.fc_norm is None
    assert model.model.input_norm is None
    assert model.model.norm_output is False


def test_eagle3_attention_declares_full_visibility_and_no_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The draft sees the whole history; which of the target's cache groups
    it rides is the plan's call, bound at startup (bind_cache_groups)."""
    model = _build_model(monkeypatch, _draft_config())
    paged_layers = [
        module for module in model.modules() if isinstance(module, PagedAttention)
    ]

    assert paged_layers
    assert {layer.sliding_window_size for layer in paged_layers} == {-1}
    for layer in paged_layers:
        with pytest.raises(RuntimeError, match="no cache group bound"):
            layer.group_id


def test_eagle3_extend_first_step_uses_decode_prewrite_capability() -> None:
    """MHA can prewrite an extend span, then attend only the last Q as decode."""

    class Backend:
        def __init__(self) -> None:
            self.queried_modes = []
            self.forward_call = None

        def support_kv_cache_prewrite(self, forward_mode):
            self.queried_modes.append(forward_mode)
            return forward_mode.is_decode()

        def forward(self, q, k, v, layer, pool, forward_mode, bs, **kwargs):
            self.forward_call = (q, k, v, layer, pool, forward_mode, bs, kwargs)
            return q

    class Narrowing:
        def __init__(self) -> None:
            self.publish_count = 0

        def publish_accepted_prefix(self) -> None:
            self.publish_count += 1

    backend = Backend()
    narrowing = Narrowing()
    gather_ids = torch.tensor([2, 5], dtype=torch.int64)
    q = torch.arange(24, dtype=torch.float32).view(6, 4)
    k = q + 100
    v = q + 200
    positions = torch.arange(6)
    fused_arg = object()
    seen = {}

    def build_fused_kv_arg(value, ctx):
        seen["build"] = (value, ctx)
        return fused_arg

    def fused_rope_kv_write(pos, query, key, arg):
        seen["prewrite"] = (pos, query, key, arg)
        return query + 1

    def fallback_rotary(*args, **kwargs):
        raise AssertionError("full EXTEND attention fallback should not run")

    attention = SimpleNamespace(
        attn=object(),
        _build_fused_kv_arg=build_fused_kv_arg,
        _fused_rope_kv_write=fused_rope_kv_write,
        rotary_emb=fallback_rotary,
    )
    ctx = SimpleNamespace(
        attn_backend=backend,
        token_to_kv_pool=object(),
        bs=2,
        forward_mode=ForwardMode.EXTEND,
        gather_ids=gather_ids,
        draft_narrowing=narrowing,
    )

    output = LlamaAttention._attn(attention, positions, q, k, v, ctx)

    assert backend.queried_modes == [ForwardMode.DECODE]
    built_v, built_ctx = seen["build"]
    assert built_v is v and built_ctx is ctx
    prewrite_positions, prewrite_q, prewrite_k, prewrite_arg = seen["prewrite"]
    assert prewrite_positions is positions
    assert prewrite_q is q and prewrite_k is k
    assert prewrite_arg is fused_arg
    assert narrowing.publish_count == 1
    torch.testing.assert_close(output, (q + 1).index_select(0, gather_ids))

    forwarded = backend.forward_call
    assert forwarded is not None
    forwarded_q, forwarded_k, forwarded_v, layer, pool, mode, bs, kwargs = forwarded
    torch.testing.assert_close(forwarded_q, output)
    assert forwarded_k is None and forwarded_v is None
    assert layer is attention.attn
    assert pool is ctx.token_to_kv_pool
    assert mode == ForwardMode.DECODE
    assert bs == ctx.bs
    assert kwargs == {"save_kv_cache": False, "record_kv_cache": True}


@pytest.mark.parametrize(
    ("forward_mode", "capture_active", "expected_queried_modes"),
    [
        (ForwardMode.MIXED, False, [ForwardMode.MIXED]),
        (ForwardMode.EXTEND, True, []),
    ],
)
def test_eagle3_does_not_force_decode_prewrite_capability(
    monkeypatch: pytest.MonkeyPatch,
    forward_mode: ForwardMode,
    capture_active: bool,
    expected_queried_modes: list[ForwardMode],
) -> None:
    """MIXED and captured EXTEND must not force the DECODE prewrite path.

    MIXED asks the backend with the real MIXED mode (which advertises no
    prewrite, so the full-attn fallback runs); captured EXTEND short-circuits
    on the capture check before ever querying the backend.
    """

    class Backend:
        def __init__(self) -> None:
            self.queried_modes = []

        def support_kv_cache_prewrite(self, mode):
            self.queried_modes.append(mode)
            return mode.is_decode()

    class Narrowing:
        def __init__(self) -> None:
            self.publish_count = 0

        def publish_accepted_prefix(self) -> None:
            self.publish_count += 1

    class Attention:
        def __init__(self) -> None:
            self.call = None

        def __call__(self, q, k, v, *, ctx):
            self.call = (q, k, v, ctx)
            return q + 1

    monkeypatch.setattr(
        llama_eagle3,
        "is_breakable_capture_active",
        lambda: capture_active,
    )
    backend = Backend()
    narrowing = Narrowing()
    paged_attention = Attention()
    gather_ids = torch.tensor([1, 4], dtype=torch.int64)
    q = torch.arange(20, dtype=torch.float32).view(5, 4)
    k = q + 100
    v = q + 200
    positions = torch.arange(5)

    def rotary_emb(pos, query, key):
        assert pos is positions
        assert query is q and key is k
        return query + 10, key + 10

    attention = SimpleNamespace(attn=paged_attention, rotary_emb=rotary_emb)
    ctx = SimpleNamespace(
        attn_backend=backend,
        token_to_kv_pool=object(),
        bs=2,
        forward_mode=forward_mode,
        gather_ids=gather_ids,
        draft_narrowing=narrowing,
    )

    output = LlamaAttention._attn(attention, positions, q, k, v, ctx)

    # Neither fallback path forces the DECODE capability: MIXED queries with
    # the real mode and falls back, and captured EXTEND short-circuits on the
    # capture check before asking.
    assert backend.queried_modes == expected_queried_modes
    assert narrowing.publish_count == 0
    assert paged_attention.call is not None
    forwarded_q, forwarded_k, forwarded_v, forwarded_ctx = paged_attention.call
    torch.testing.assert_close(forwarded_q, q + 10)
    torch.testing.assert_close(forwarded_k, k + 10)
    assert forwarded_v is v
    assert forwarded_ctx is ctx
    torch.testing.assert_close(output, (q + 11).index_select(0, gather_ids))


def test_eagle3_fc_norm_and_norm_output_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(monkeypatch, _draft_config(fc_norm=True, norm_output=True))

    assert model.model.norm_output is True
    assert model.model.input_norm is None
    fc_norm = model.model.fc_norm
    assert fc_norm is not None and len(fc_norm) == 3
    for norm in fc_norm:
        assert isinstance(norm, RMSNorm)
        assert norm.weight.shape == (_HIDDEN,)
    assert model.model.fc.weight.shape == (_HIDDEN, 3 * _HIDDEN)


def test_eagle3_torchspec_checkpoint_names_all_reach_a_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(monkeypatch, _draft_config(fc_norm=True, norm_output=True))

    loaded: list[str] = []

    def _track(name: str):
        def wrapper(param, weight, *args, **kwargs):
            loaded.append(name)

        return wrapper

    for name, param in model.named_parameters():
        param.weight_loader = _track(name)

    weights = _torchspec_checkpoint_weights()
    model.load_weights(weights)

    assert len(loaded) == len(weights), (
        f"only {len(loaded)}/{len(weights)} checkpoint tensors reached a "
        f"weight loader; loaded params: {sorted(loaded)}"
    )
