# Dialogue adaptation

The Rust trainer freezes the neural tokenizer, encodes each assistant reply, and supervises every target concept scale using teacher-forced residual latents. It adapts the generator heads by default. Use `--full` to train all generator weights.

## Data and model files

Training input is UTF-8 JSON Lines. Each line is a conversation represented by a nonempty array of chronological turns. Each turn contains nonempty `user` and `assistant` strings. Earlier turns become context for later replies. Assistant targets are truncated to the decoder's 48-position budget, with a position reserved for EOS.

```json
[{"user":"Remember this badge color: teal.","assistant":"I will remember teal."},{"user":"What badge color did I give you?","assistant":"The badge color is teal."}]
```

Start with the official XSum checkpoint in `artifacts/xsum`. Training writes a Safetensors generator, tokenizer checkpoint, and configuration to a new directory. The server discovers that directory under its model root at startup. Use a corpus whose terms allow the intended training and evaluation.

## Reproduce the smoke test

```sh
target/release/ncm-chat train --backend cpu --train training/smoke.jsonl \
  --output artifacts/my-dialogue --steps 240 --learning-rate 0.002
bash scripts/smoke-http.sh target/release/ncm-chat my-dialogue
```

The heads-only run used four examples from two conversations and seed 2026. Reported concept loss fell from 2.4709 on the first step to 0.4158 on the last. The HTTP test sent the first generated reply as context for the second request, then checked persistence, stale revision rejection, and reset.

| Badge color | In training | First reply | Follow-up reply |
|---|---|---|---|
| Teal | Yes | I will remember teal. | The badge badge is teal. |
| Purple | Yes | I will remember purple. | The badge badge is purple. |
| Orange | No | I will remember purple. | The badge badge is purple. |

The smoke checkpoint recalled both seen colors and failed on the held-out color. A representative held-out corpus is needed to assess general dialogue quality. [Smoke corpus](../training/smoke.jsonl) · [Raw replies](results/rust/dialogue-smoke.jsonl)
