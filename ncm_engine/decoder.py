import numpy as np
import torch


class TextDecoder:
    """Upstream evaluation policy for parallel neural decoder logits."""

    def __init__(self, config, allowance, vocab):
        self.vocab = vocab
        self.freq = float(config.get("frequency_penalty", 0))
        self.deficit = float(config.get("frequency_deficit", 0))
        adj = str(config.get("adjacency_penalty", "") or "")
        self.adj = [float(x) for x in adj.split(",")] if adj else []
        self.allowance = allowance
        self.cut_at_eos = bool(config.get("cut_at_eos", True))

    def select(self, logits, *, sparse=False):
        if self.freq <= 0 and not self.adj:
            return logits.argmax(-1)
        if sparse and logits.shape[0] == 1:
            return self._select_sparse(logits)
        batch, steps, width = logits.shape
        # Upstream uses FP64 here. This stays on CPU for MPS compatibility.
        logits = logits.cpu().double()
        allow = self.allowance[:width] * steps if self.allowance is not None else None
        alpha, window, decay = (self.adj + [1, 1])[:3] if self.adj else (0, 0, 1)
        counts = torch.zeros(batch, width, dtype=logits.dtype)
        out = torch.empty(batch, steps, dtype=torch.long)
        one = torch.ones(batch, 1, dtype=logits.dtype)
        biases = [-alpha * decay**j * one for j in range(int(window))]
        history = []
        for t, scores in enumerate(logits.unbind(1)):
            scores = scores.clone()
            if self.freq > 0:
                excess = (counts - allow).clamp_min(0) if allow is not None else counts
                scores.sub_(excess, alpha=self.freq)
                if self.deficit > 0 and allow is not None:
                    scores.add_(
                        (allow - counts).clamp_min(0),
                        alpha=self.deficit * ((steps - t) / steps),
                    )
            for bias, previous in zip(biases, reversed(history)):
                scores.scatter_add_(1, previous, bias)
            token = scores.argmax(-1)
            out[:, t] = token
            history.append(token[:, None])
            counts.scatter_add_(1, token[:, None], one)
        return out

    def _select_sparse(self, logits):
        # Every unseen token has count zero. Apply its allowance in one
        # broadcast, then recompute only the previously selected columns.
        scores = logits[0].cpu().double().numpy().copy()
        steps, width = scores.shape
        allowance = (
            self.allowance[:width].numpy() * steps
            if self.allowance is not None
            else None
        )
        use_frequency = self.freq > 0
        original = (
            scores.copy()
            if allowance is not None and use_frequency and self.deficit > 0
            else scores
        )
        if original is not scores:
            scores += (self.deficit * (steps - np.arange(steps)) / steps)[
                :, None
            ] * allowance
        alpha, window, decay = (self.adj + [1, 1])[:3] if self.adj else (0, 0, 1)
        counts = {}
        history = []
        for t, row in enumerate(scores):
            if use_frequency:
                coefficient = self.deficit * ((steps - t) / steps)
                for token, count in counts.items():
                    quota = allowance[token] if allowance is not None else 0
                    row[token] = original[t, token] - self.freq * max(count - quota, 0)
                    if allowance is not None and self.deficit > 0:
                        row[token] += coefficient * max(quota - count, 0)
            for j, token in enumerate(
                reversed(history[-int(window) :]) if window else ()
            ):
                row[token] -= alpha * decay**j
            token = int(row.argmax())
            history.append(token)
            counts[token] = counts.get(token, 0) + 1
        return torch.tensor([history], dtype=torch.long)

    def text(self, ids):
        eos, pad = self.vocab.eos_token_id, self.vocab.pad_token_id
        if self.cut_at_eos and eos in ids:
            ids = ids[: ids.index(eos)]
        return self.vocab.decode(
            [i for i in ids if i not in (eos, pad)], skip_special_tokens=True
        )
