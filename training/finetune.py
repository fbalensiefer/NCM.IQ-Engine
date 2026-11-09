import argparse
import json
import shutil
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from backend.conversation import dialogue_prompt
from ncm_engine.inference import Engine
from ncm_engine.tokenizer import prefix_ids, target_ids


def examples(path):
    with path.open() as handle:
        for line, text in enumerate(handle, start=1):
            if not text.strip():
                continue
            turns = json.loads(text)
            if (
                not isinstance(turns, list)
                or not turns
                or any(
                    not isinstance(turn, dict)
                    or any(
                        not isinstance(turn.get(role), str) or not turn[role].strip()
                        for role in ("user", "assistant")
                    )
                    for turn in turns
                )
            ):
                raise ValueError(f"Invalid dialogue on line {line}")
            for i, turn in enumerate(turns):
                yield dialogue_prompt(turns[:i], turn["user"]), turn["assistant"]


@torch.no_grad()
def prepare(engine, pairs):
    net, vqvae = engine.net, engine.vqvae
    for prompt, answer in pairs:
        prefix, length = prefix_ids(
            engine.vocab, prompt, net.prefix_len, net.prefix_len, keep_recent=True
        )
        prefix = prefix[:, :length].to(engine.device)
        target = torch.tensor(
            [target_ids(engine.vocab, answer, vqvae.context_length)],
            device=engine.device,
        )
        codes = vqvae.encode(target)
        teacher = vqvae.vq.idx_to_ncm_input(codes)
        yield prefix, teacher, codes


def hidden_levels(net, prefix, teacher):
    """Teacher-forced block-diagonal prediction over all hierarchy levels."""
    p = prefix.shape[1]
    t = net.transformer
    x = torch.cat((net.build_l0_query_input(prefix.shape[0]), teacher), dim=1)
    if net.use_post_upsample_conv:
        x = net._apply_post_upsample_convs_multi(x)
    x = t.wte(x)
    source = t.wte(net.vqvae_local[0].encoder.wte(prefix))
    if net.config.use_level_encoding:
        x = x + t.wle(net.lvl_1L)
    if not net.use_rope:
        source = source + net.wpe(net.wp_prefix[:, :p])
        x = x + net.wpe(net.wp_1L)
    x = torch.cat((source, x), dim=1)
    idx = torch.cat(
        (
            torch.arange(p, device=x.device),
            torch.arange(net.prefix_len, net.prefix_len + net.L, device=x.device),
        )
    )
    mask = net.attn_mask_for_masking.index_select(2, idx).index_select(3, idx)
    pad = F.pad(prefix == net.prefix_pad_id, (0, net.L), value=False)
    mask = mask.masked_fill(pad[:, None, None, :], float("-inf"))
    rope = (net.rope_cos[idx], net.rope_sin[idx]) if net.use_rope else None
    for block in t.blocks:
        x = block(x, attn_mask=mask, rope_cos_sin=rope)
    return t.ln_f(x[:, p:]).split(list(net.levels), dim=1)


def objective(net, hidden, codes):
    # Each concept contributes one cross-entropy term. Positions at every
    # scale are supervised from the true cumulative reconstruction.
    loss = 0
    for i, (x, targets) in enumerate(zip(hidden, codes.split(list(net.levels), dim=1))):
        logits = net.lm_heads[i](x)
        loss = loss + F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="sum"
        )
    return loss / codes.numel()


def train(engine, pairs, steps, lr, full):
    net = engine.net
    pairs = list(pairs)
    if not pairs:
        raise ValueError("Training data is empty")
    if full:
        net.requires_grad_(True).train()
    else:
        net.lm_heads.requires_grad_(True)

    @lru_cache(maxsize=128)
    def load_example(index):
        prefix, teacher, codes = next(prepare(engine, [pairs[index]]))
        if full:
            return prefix, teacher, codes
        with torch.no_grad():
            return hidden_levels(net, prefix, teacher), codes

    optimizer = torch.optim.AdamW(
        (p for p in net.parameters() if p.requires_grad), lr=lr
    )
    order = torch.Generator().manual_seed(torch.initial_seed())
    permutation = []
    for step in range(steps):
        if not permutation:
            permutation = torch.randperm(len(pairs), generator=order).tolist()
        sample = load_example(permutation.pop())
        hidden, codes = (hidden_levels(net, *sample[:2]), sample[2]) if full else sample
        optimizer.zero_grad(set_to_none=True)
        loss = objective(net, hidden, codes)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == steps:
            print(f"step={step + 1} concept_ce={loss.item():.4f}", flush=True)


def save(engine, source, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent) as tmp:
        staging = Path(tmp) / "checkpoint"
        staging.mkdir()
        cfg = engine.config
        tokenizer_name = cfg.checkpoints.tokenizer
        shutil.copyfile(source / tokenizer_name, staging / tokenizer_name)
        torch.save(
            {"model": engine.net.cpu().state_dict()},
            staging / cfg.checkpoints.generator,
        )
        cfg.task = "dialogue"
        cfg.generation = {"temperature": 1, "top_k": 1, "top_p": 0, "cfg_scale": 0}
        from omegaconf import OmegaConf

        OmegaConf.save(cfg, staging / "config.yaml")
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(
        description="Adapt the XSum NCM generator to dialogue."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/xsum"))
    parser.add_argument("--vocabulary", type=Path, default=Path("artifacts/gpt2"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default="auto"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--full", action="store_true", help="Train all generator weights"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1 or not 0 < args.learning_rate < 1:
        parser.error(
            "Steps and threads must be positive, and learning rate must be between 0 and 1"
        )
    if args.output.exists():
        parser.error("--output already exists")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    engine = Engine(args.checkpoint, args.vocabulary, args.device, "quality")
    train(engine, examples(args.train), args.steps, args.learning_rate, args.full)
    save(engine, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
