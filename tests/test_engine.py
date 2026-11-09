import os
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from ncm_engine.decoder import TextDecoder
from ncm_engine.device import select_device
from ncm_engine.inference import Engine
from ncm_engine.sampling import Sampling
from optimization.tokenizer_optimizer import interpolate
from training.finetune import hidden_levels, objective, prepare


@pytest.mark.parametrize("cfg", [0.0, 1.0])
def test_parallel_matches_reference(tiny, cfg):
    baseline = Engine(*tiny, "cpu", "quality")
    optimized = Engine(*tiny, "cpu", "balanced")
    settings = Sampling(temperature=0, cfg_scale=cfg)
    text = "User: hello there\nAssistant:"
    expected = baseline.generate(text, sampling=settings, detailed=True)
    result = optimized.generate(text, sampling=settings, detailed=True)
    assert len(result.concept_codes) == 13
    assert result.concept_codes == expected.concept_codes
    assert result.token_ids == expected.token_ids
    assert result.milliseconds["transformer_level_2"] > 0
    assert all(
        block.attn.cached_k is None for block in optimized.net.transformer.blocks
    )
    reconstructed = optimized.reconstruct("hello again", detailed=True)
    original = baseline.reconstruct("hello again", detailed=True)
    assert reconstructed.concept_codes == original.concept_codes
    assert reconstructed.token_ids == original.token_ids
    assert reconstructed.milliseconds["rvq_search"] > 0


def test_stochastic_generation(tiny):
    engine = Engine(*tiny, "cpu", "balanced")
    sampling = Sampling(temperature=0.8, top_k=4, top_p=0.9)
    torch.manual_seed(42)
    first = engine.generate("hello", sampling=sampling)
    torch.manual_seed(42)
    second = engine.generate("hello", sampling=sampling)
    assert first.concept_codes == second.concept_codes
    assert all(0 <= code < 12 for code in first.concept_codes)
    with pytest.raises(ValueError, match="sampling"):
        engine.generate("hello", sampling=Sampling(top_k=13))


def test_sparse_decode_matches_dense():
    torch.manual_seed(41)
    logits = torch.randn(1, 48, 1024)
    allowance = torch.softmax(torch.randn(1024, dtype=torch.double), dim=0)
    for cfg in (
        {
            "frequency_penalty": 2,
            "frequency_deficit": 1,
            "adjacency_penalty": "5,4,0.5",
        },
        {"adjacency_penalty": "5,4,0.5"},
        {"frequency_penalty": 2},
    ):
        decoder = TextDecoder(cfg, allowance, None)
        assert torch.equal(decoder.select(logits), decoder.select(logits, sparse=True))


def test_cpu_int8_inference(tiny):
    if not {"qnnpack", "x86", "fbgemm"}.intersection(
        torch.backends.quantized.supported_engines
    ):
        pytest.skip("This build has no INT8 CPU kernels")
    engine = Engine(*tiny, "cpu", "ultra")
    result = engine.generate("hello there")
    assert len(result.concept_codes) == 13
    assert len(result.token_ids) == 8
    assert all(0 <= code < 12 for code in result.concept_codes)


def test_teacher_forced_objective_has_gradients(tiny):
    engine = Engine(*tiny, "cpu")
    engine.net.lm_heads.requires_grad_(True)
    prefix, teacher, codes = next(
        prepare(engine, [("User: hello\nAssistant:", "hello again")])
    )
    loss = objective(engine.net, hidden_levels(engine.net, prefix, teacher), codes)
    assert torch.isfinite(loss)
    loss.backward()
    assert engine.net.lm_heads[0].proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in engine.vqvae.parameters())


def test_device_selection():
    assert select_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        select_device("mlx")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_pool_and_generation(tiny):
    x = torch.randn(2, 3, 8)
    actual = interpolate(x.to("mps"), size=(3,), mode="area").cpu()
    torch.testing.assert_close(actual, F.interpolate(x, size=(3,), mode="area"))
    engine = Engine(*tiny, "mps", "balanced")
    result = engine.generate("hello there")
    assert len(result.concept_codes) == 13
    assert len(engine.reconstruct("hello").token_ids) == 8


@pytest.mark.model
def test_official_model():
    if not (root := os.getenv("NCM_TEST_ARTIFACTS")):
        pytest.skip("Set NCM_TEST_ARTIFACTS to test the pinned XSum release")
    directory = Path(root)
    prompt = "The city library has opened a new reading room for children after a six month renovation. The room opens on Monday and offers free weekend workshops."
    baseline = Engine(directory / "xsum", directory / "gpt2", "cpu", "quality")
    original = baseline.generate(prompt)
    reconstructed = baseline.reconstruct(prompt)
    del baseline
    engine = Engine(directory / "xsum", directory / "gpt2", "cpu", "balanced")
    result = engine.generate(prompt)
    assert result.text
    assert result.concept_codes == original.concept_codes
    assert result.token_ids == original.token_ids
    assert engine.reconstruct(prompt).token_ids == reconstructed.token_ids
