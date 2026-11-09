from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from omegaconf import OmegaConf

from ncm_engine._upstream.models.next_concept import NextConceptTransformer
from ncm_engine._upstream.models.vqvae import VQVAE
from ncm_engine.decoder import TextDecoder
from ncm_engine.device import select_device, synchronize
from ncm_engine.sampling import Sampling
from ncm_engine.tokenizer import load_vocabulary, prefix_ids, target_ids
from optimization.memory_optimizer import read_checkpoint, release_training_buffers
from optimization.profiler import Profile, trace_reference
from optimization.quantization import HalfProjection, apply_cpu_int8
from optimization.tokenizer_optimizer import FrozenRVQ
from optimization.transformer_optimizer import generate_parallel


@dataclass
class Result:
    text: str
    concept_codes: list[int]
    token_ids: list[int]
    milliseconds: dict[str, float]
    rss_bytes: int


class Engine:
    def __init__(self, directory, vocabulary, backend="auto", profile="quality"):
        start = perf_counter()
        self.device = select_device(backend)
        if profile == "ultra" and backend == "auto" and self.device.type == "cuda":
            self.device = select_device("cpu")
        self.profile = profile
        directory = Path(directory)
        self.config = cfg = OmegaConf.load(directory / "config.yaml")
        if cfg.task not in {"xsum", "dialogue"}:
            raise ValueError(f"Unsupported checkpoint task: {cfg.task}")
        if cfg.model.get("ctx_encoder", {}).get("enabled", False):
            raise ValueError(
                "Checkpoints with an external context encoder are unsupported"
            )
        self.vocab = load_vocabulary(Path(vocabulary), cfg.tokenizer.pre_tokenizer)
        if self.vocab.pad_token_id != cfg.model.prefix_pad_token_id:
            raise ValueError(
                "The text vocabulary does not match the checkpoint's pad token"
            )
        tk_path, gen_path = (
            directory / cfg.checkpoints.tokenizer,
            directory / cfg.checkpoints.generator,
        )
        self.checkpoint_bytes = tk_path.stat().st_size + gen_path.stat().st_size
        self.vqvae = VQVAE(**cfg.tokenizer.vqvae)
        self.vqvae.load_state_dict(
            read_checkpoint(tk_path)["model"], strict=True, assign=profile != "quality"
        )
        self.net = NextConceptTransformer(self.vqvae, cfg.model)
        generator = read_checkpoint(gen_path)
        self.net.load_state_dict(
            generator["model"], strict=True, assign=profile != "quality"
        )
        generation = cfg.generation
        allowance = None
        if generation.get("frequency_allowance"):
            spec = str(generation.frequency_allowance)
            if spec != f"{cfg.checkpoints.generator}:inference.token_unigram":
                raise ValueError(f"Unsupported frequency allowance: {spec}")
            allowance = generator["inference"]["token_unigram"].double().clone()
        del generator
        self.decoder = TextDecoder(generation, allowance, self.vocab)
        self.defaults = Sampling(
            **{k: generation[k] for k in vars(Sampling()) if k in generation}
        )
        if profile != "quality":
            release_training_buffers(self.vqvae.vq)
        self.vqvae.to(self.device).eval().requires_grad_(False)
        self.net.to(self.device).eval().requires_grad_(False)
        self._configure_profile()
        synchronize(self.device)
        self.load_ms = (perf_counter() - start) * 1000

    def _configure_profile(self):
        if self.profile not in {"quality", "balanced", "ultra"}:
            raise ValueError(f"Unsupported profile: {self.profile}")
        if not (
            self.net.query_level0
            and not self.net.l0_ar
            and not self.net.predict_level0
            and self.net.config.raw_token_prefix
            and not self.net.config.get("contextual_prefix", False)
            and self.net.prefix_pad_id is not None
            and self.net.config.attn_pattern == "block_diagonal"
        ):
            raise ValueError(
                "This engine requires the XSum parallel conditional architecture"
            )
        self.rvq = FrozenRVQ(self.vqvae.vq) if self.profile != "quality" else None
        if self.profile == "ultra":
            if self.device.type == "cpu":
                apply_cpu_int8(self.net, self.vqvae.decoder)
            elif self.device.type == "mps":
                decoder = self.vqvae.decoder
                decoder.output_layer = HalfProjection(decoder.output_layer)
            else:
                raise ValueError(
                    "The experimental Ultra-light profile supports CPU and MPS"
                )

    @torch.inference_mode()
    def generate(self, text, *, sampling=None, context_limit=None, detailed=False):
        sampling = sampling or self.defaults
        sampling.validate(self.net.vocab_sizes)
        width = self.net.prefix_len
        limit = width if context_limit is None else context_limit
        if not 1 <= limit <= width:
            raise ValueError(f"Context limit must be between 1 and {width}")
        perf = Profile(self.device, detailed)
        with perf.stage("total"):
            with perf.stage("preprocessing"):
                prefix, valid = prefix_ids(
                    self.vocab, text, limit, width, self.config.task == "dialogue"
                )
            with perf.stage("input_transfer"):
                prefix = prefix.to(self.device)
            codes, latent = self._generate_codes(prefix, valid, sampling, perf)
            with perf.stage("neural_decoder"):
                logits = self.vqvae.decoder(latent)
            with perf.stage("decoder_selection"):
                ids = self.decoder.select(logits, sparse=self.profile != "quality")
            with perf.stage("output_transfer"):
                tokens = ids.cpu()[0].tolist()
                concepts = codes.cpu()[0].tolist()
            with perf.stage("text_reconstruction"):
                text = self.decoder.text(tokens)
        times, rss = perf.result()
        return Result(text, concepts, tokens, times, rss)

    def _generate_codes(self, prefix, valid, sampling, perf):
        with perf.stage("concept_generation"):
            if self.profile != "quality":
                return generate_parallel(
                    self.net, prefix, valid, sampling, self.vocab.eos_token_id, perf
                )
            with trace_reference(self.net, self.vqvae, perf):
                codes = self.net.generate(
                    None, prefix, **vars(sampling), cfg_null_eos=self.vocab.eos_token_id
                )
        with perf.stage("rvq_reconstruction"):
            _, latents = self.vqvae.vq.idx_to_zhats(codes)
        return codes, latents[-1]

    @torch.inference_mode()
    def reconstruct(self, text, *, detailed=False):
        perf = Profile(self.device, detailed)
        with perf.stage("total"):
            ids = target_ids(self.vocab, text, self.vqvae.context_length)
            with perf.stage("input_transfer"):
                tokens = torch.tensor([ids], dtype=torch.long).to(self.device)
            with perf.stage("neural_encoder"):
                encoded = self.vqvae.encode_z(tokens)
            with perf.stage("rvq_search"):
                codes = self._encode_latents(encoded)
            with perf.stage("rvq_reconstruction"):
                latent = self.vqvae.vq.idx_to_zhats(codes, return_zlevels=False)[0][-1]
            with perf.stage("neural_decoder"):
                decoded = self.vqvae.decoder(latent).argmax(-1).cpu()[0].tolist()
            with perf.stage("text_reconstruction"):
                text = self.decoder.text(decoded)
        times, rss = perf.result()
        return Result(text, codes.cpu()[0].tolist(), decoded, times, rss)

    def _encode_latents(self, latents):
        return self.rvq.encode(latents) if self.rvq else self.vqvae.vq.z_to_idx(latents)
