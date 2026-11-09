# Dialogue adaptation

The official release provides conditional summarization, translation and Sudoku. Dialogue is trained separately using conditional concept cross-entropy. The adaptation freezes the neural tokenizer. It encodes the answer plus EOS and padding, obtains each target code, and computes the teacher input for each scale from the previous scales' true residual sum. One block-diagonal Transformer pass processes the scale blocks. The optional heads-only path reuses hidden states with a bounded cache.

## Smoke run on Apple M2 CPU

The four supervised turn examples in [the authored corpus](../training/smoke.jsonl) were trained for 240 steps with seed 2026 and learning rate 0.002. Inference ran through the FastAPI session endpoints, and the second request used the first generated reply in its context.

| First message requests recall of | Generated first reply | Generated follow-up |
|---|---|---|
| teal, seen | I will remember teal. | The badge badge is teal. |
| purple, seen | I will remember purple. | The badge badge is purple. |
| orange, held out | I will remember purple. | The badge badge is purple. |

This verifies a functioning train, load, generate and history loop and exposes a failure to generalize. The tiny corpus supplies neither representative dialogue quality estimates nor evidence of instruction following. Adaptation still uses the XSum tokenizer and fixed 48-token target shape. Examples longer than that budget are truncated with EOS appended. The local smoke checkpoint is excluded from Git.
