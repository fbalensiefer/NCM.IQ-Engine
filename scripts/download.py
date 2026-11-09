import argparse
import hashlib
import shutil
from pathlib import Path

NCM = "d342ce8aaea58830eef3b8ad473664673e41f692"
GPT2 = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
FILES = {
    "config.yaml": "5dee6b7339645a22362f695fbadb27df6e5254db2d3be0bff8516138d95da493",
    "generator.pt": "f152f390253de886e46303654f01b9e7a2586e609ba3d6b5e86d5969903b1fd4",
    "tokenizer.pt": "1f806ddaf8e023d1933e95efeabcaaf99f795576c0a05045f1a0f9389ab34fa7",
}


def main():
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(
        description="Download the pinned official XSum release."
    )
    parser.add_argument("--directory", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    target = args.directory / "xsum"
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        source = Path(
            hf_hub_download(
                "hazyresearch/ncm-checkpoints", f"xsum/{name}", revision=NCM
            )
        )
        with source.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise RuntimeError(f"Checksum mismatch: {name}")
        output = target / name
        temp = target / f".{name}.download"
        shutil.copyfile(source, temp)
        temp.replace(output)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        hf_hub_download(
            "openai-community/gpt2",
            name,
            revision=GPT2,
            local_dir=args.directory / "gpt2",
        )
    print(f"Offline inference files ready under {args.directory}")


if __name__ == "__main__":
    main()
