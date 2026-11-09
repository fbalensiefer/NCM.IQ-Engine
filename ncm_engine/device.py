import torch


def supported_backends() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    return devices


def select_device(requested: str = "auto") -> torch.device:
    devices = supported_backends()
    if requested == "auto":
        return torch.device(devices[0])
    if requested not in devices:
        raise ValueError(
            f"{requested} is unavailable. Available backends: {', '.join(devices)}"
        )
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
