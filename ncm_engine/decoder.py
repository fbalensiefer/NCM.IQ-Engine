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

    def select(self, logits):
        if self.freq <= 0 and not self.adj:
            return logits.argmax(-1)
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

    def text(self, ids):
        eos, pad = self.vocab.eos_token_id, self.vocab.pad_token_id
        if self.cut_at_eos and eos in ids:
            ids = ids[: ids.index(eos)]
        return self.vocab.decode(
            [i for i in ids if i not in (eos, pad)], skip_special_tokens=True
        )
