import pytest
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from ncm_engine._upstream.models.next_concept import NextConceptTransformer
from ncm_engine._upstream.models.vqvae import VQVAE


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    torch.manual_seed(11)
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("artifacts")
    model = root / "tiny-dialogue"
    model.mkdir()
    vocabulary = root / "vocab"
    items = [
        "<unk>",
        "<eos>",
        "<pad>",
        "User",
        "Assistant",
        ":",
        "hello",
        "again",
        "there",
    ]
    raw = Tokenizer(
        WordLevel({word: i for i, word in enumerate(items)}, unk_token="<unk>")
    )
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>"
    )
    tokenizer.save_pretrained(vocabulary)
    cfg = OmegaConf.create(
        {
            "task": "dialogue",
            "checkpoints": {"tokenizer": "tokenizer.pt", "generator": "generator.pt"},
            "model": {
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 16,
                "bias": True,
                "dropout": 0.0,
                "vocab_size": [12] * 3,
                "prefix_len": 20,
                "prefix_pad_token_id": 2,
                "raw_token_prefix": True,
                "query_level0": True,
                "predict_level0": False,
                "use_rope": True,
                "use_qk_norm": True,
                "use_level_encoding": True,
                "attn_pattern": "block_diagonal",
                "tie_heads_to_codebook": True,
            },
            "tokenizer": {
                "pre_tokenizer": {
                    "special_tokens": {"eos_token": "<eos>", "pad_token": "<pad>"}
                },
                "vqvae": {
                    "n_layers": 2,
                    "in_vocab_size": len(items),
                    "embed_dim": 16,
                    "context_length": 8,
                    "compression_factor": 1,
                    "dropout": 0,
                    "vqvae_vocab_size": [12] * 3,
                    "vector_quantizer_config": {
                        "clss": "multiscale_residual_vector_quantizer",
                        "levels": {
                            "use_manual_levels": True,
                            "manual_levels": [2, 3, 8],
                        },
                    },
                },
            },
            "generation": {"temperature": 0, "top_k": 1, "top_p": 0},
        }
    )
    tk = VQVAE(**cfg.tokenizer.vqvae)
    net = NextConceptTransformer(tk, cfg.model)
    torch.save({"model": tk.state_dict()}, model / "tokenizer.pt")
    torch.save({"model": net.state_dict()}, model / "generator.pt")
    OmegaConf.save(cfg, model / "config.yaml")
    return model, vocabulary
