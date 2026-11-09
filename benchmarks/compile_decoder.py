"""Compare eager and Inductor for a fixed-shape official neural decoder."""

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

import torch

from ncm_engine.inference import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/xsum"))
    parser.add_argument("--vocabulary", type=Path, default=Path("artifacts/gpt2"))
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--transformer-block", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    engine = Engine(args.checkpoint, args.vocabulary, "cpu", "quality")
    with torch.inference_mode():
        if args.transformer_block:
            net = engine.net
            stack = net.transformer
            prefix = torch.full(
                (1, net.prefix_len), engine.vocab.pad_token_id, dtype=torch.long
            )
            ids = engine.vocab.encode("A city library has opened a reading room.")
            prefix[:, : len(ids)] = torch.tensor(ids)
            source = stack.wte(engine.vqvae.encoder.wte(prefix))
            target = stack.wte(net.build_l0_query_input(1)) + stack.wle(
                net.lvl_1L[:, : net.levels[0]]
            )
            features = torch.cat((source, target), dim=1)
            keep = torch.arange(features.shape[1]) < len(ids)
            mask = keep.expand(features.shape[1], -1).clone()
            mask[net.prefix_len :, net.prefix_len :] = True
            rope = (net.rope_cos_gen_0, net.rope_sin_gen_0)
            component = stack.blocks[0]
            inputs = (features, mask, rope)
        else:
            codes = engine.generate(
                "A city library has opened a reading room."
            ).concept_codes
            component = engine.vqvae.decoder
            inputs = (engine.vqvae.vq.idx_to_zhats(torch.tensor([codes]))[1][-1],)
        report = {}
        for name, model in (
            ("eager", component),
            ("inductor", torch.compile(component, dynamic=False)),
        ):
            values = []
            for i in range(args.runs + 3):
                start = perf_counter()
                logits = model(*inputs)
                elapsed = (perf_counter() - start) * 1000
                if i == 0:
                    report[name] = {"first_ms": elapsed}
                if i >= 3:
                    values.append(elapsed)
            report[name]["median_ms"] = statistics.median(values)
            if name == "eager":
                baseline = logits
            else:
                report[name]["max_abs_error"] = (baseline - logits).abs().max().item()
                report[name]["argmax_equal"] = bool(
                    torch.equal(baseline.argmax(-1), logits.argmax(-1))
                )
            print(json.dumps({name: report[name]}), flush=True)


if __name__ == "__main__":
    main()
