"""Compare individual rectangular target-attention kernels on Apple Silicon."""

import argparse
import json
from importlib.metadata import version
from statistics import median
from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import scaled_dot_product_attention


def main():
    import mlx.core as mx

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=64)
    args = parser.parse_args()
    if args.runs < 1 or not torch.backends.mps.is_available():
        parser.error("Positive runs and an MPS-capable host are required")

    rng = np.random.default_rng(2026)
    print(
        json.dumps(
            {"torch": str(torch.__version__), "mlx": version("mlx"), "runs": args.runs}
        )
    )
    for dtype in (np.float32, np.float16):
        for prefix, query in ((32, 48), (512, 48)):
            length = prefix + query
            arrays = [
                rng.standard_normal((1, 12, n, 64)).astype(dtype)
                for n in (query, length, length)
            ]
            mask = np.ones((1, 1, 1, length), dtype=np.bool_)
            mask[..., 4:prefix:7] = False
            torch_inputs = [torch.from_numpy(x).to("mps") for x in arrays]
            torch_mask = torch.from_numpy(mask).to("mps")
            mlx_inputs = [mx.array(x) for x in arrays]
            mlx_mask = mx.array(mask)
            mx.eval(*mlx_inputs, mlx_mask)

            def mps():
                output = scaled_dot_product_attention(
                    *torch_inputs, attn_mask=torch_mask
                )
                torch.mps.synchronize()
                return output

            def mlx():
                output = mx.fast.scaled_dot_product_attention(
                    *mlx_inputs, scale=64**-0.5, mask=mlx_mask
                )
                mx.eval(output)
                mx.synchronize()
                return output

            report = {"dtype": dtype.__name__, "prefix": prefix, "query": query}
            for name, run in (("mps", mps), ("mlx", mlx)):
                values = []
                for i in range(args.runs + 8):
                    start = perf_counter()
                    result = run()
                    if i >= 8:
                        values.append((perf_counter() - start) * 1000)
                report[f"{name}_ms"] = median(values)
                if name == "mps":
                    baseline = result.cpu().numpy().astype(np.float32)
                else:
                    report["max_abs_error"] = float(
                        np.max(np.abs(baseline - np.array(result).astype(np.float32)))
                    )
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
