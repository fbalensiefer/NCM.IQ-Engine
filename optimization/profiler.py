from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter

import psutil

from ncm_engine.device import synchronize


class Profile:
    def __init__(self, device, detailed=False):
        self.device = device
        self.detailed = detailed
        self.ms = defaultdict(float)

    @contextmanager
    def stage(self, name, *, detail=False):
        if detail and not self.detailed:
            yield
            return
        # The outer request timer supplies the completion barrier in normal mode.
        if self.detailed or name == "total":
            synchronize(self.device)
        start = perf_counter()
        yield
        if self.detailed or name == "total":
            synchronize(self.device)
        self.ms[name] += (perf_counter() - start) * 1000

    def result(self):
        return dict(self.ms), psutil.Process().memory_info().rss


@contextmanager
def trace_reference(net, vqvae, perf):
    """Add temporary timers around upstream operations without replacing their math."""
    if not perf.detailed:
        yield
        return
    from contextlib import ExitStack
    from functools import wraps

    with ExitStack() as cleanup:

        def trace(obj, method, label):
            original = getattr(obj, method)
            local = obj.__dict__.get(method)

            @wraps(original)
            def timed(*args, **kwargs):
                with perf.stage(label):
                    return original(*args, **kwargs)

            setattr(obj, method, timed)
            if local is None:
                cleanup.callback(delattr, obj, method)
            else:
                cleanup.callback(setattr, obj, method, local)

        trace(net, "_sample_level", "concept_sampling")
        trace(vqvae.vq, "get_next_autoregressive_input", "rvq_generation")
        trace(vqvae.encoder.wte, "forward", "input_embedding")

        def measure_stack(module, args):
            nonlocal active, level
            active = perf.stage(f"transformer_level_{level}")
            active.__enter__()
            level += 1

        def finish_stack(module, args, output):
            if active is not None:
                active.__exit__(None, None, None)

        active, level = None, 0
        blocks = net.transformer.blocks
        cleanup.callback(blocks[0].register_forward_pre_hook(measure_stack).remove)
        cleanup.callback(
            net.transformer.ln_f.register_forward_hook(finish_stack).remove
        )
        yield
