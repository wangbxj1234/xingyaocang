"""SigLIP2 批量编码。文本向量缓存。图像向量做 L2 归一化。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from siglip_filter.similarity import l2_normalize


def feature_tensor(output):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    for name in ("image_embeds", "text_embeds"):
        value = getattr(output, name, None)
        if value is not None:
            return value
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class Siglip2Encoder:
    def __init__(self, model_name: str, dtype_name: str, device_name: str, batch_size: int):
        import torch
        from transformers import AutoModel, AutoProcessor

        if device_name == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_name)
        dtype = _dtype(torch, dtype_name, device)
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_name)
        try:
            self.model = AutoModel.from_pretrained(model_name, dtype=dtype)
        except TypeError:
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(device).eval()
        self._torch = torch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def encode_images(self, images) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        rows = []
        for start in range(0, len(images), self.batch_size):
            batch = list(images[start : start + self.batch_size])
            rows.append(self._encode_image_batch(batch))
        return l2_normalize(np.concatenate(rows, axis=0))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalize(self._encode_text_batch(list(texts)))

    def peak_memory_mb(self):
        if self.device.type != "cuda":
            return None
        return float(self._torch.cuda.max_memory_allocated(self.device) / (1024 * 1024))

    def _encode_image_batch(self, images) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(images=images, return_tensors="pt")
        kwargs = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                continue
            if key == "pixel_values":
                kwargs[key] = value.to(self.device, dtype=self.dtype)
            else:
                kwargs[key] = value.to(self.device)
        with torch.inference_mode():
            try:
                output = self.model.get_image_features(**kwargs)
            except TypeError:
                output = self.model.get_image_features(pixel_values=kwargs["pixel_values"])
        return feature_tensor(output).detach().float().cpu().numpy()

    def _encode_text_batch(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        kwargs = {key: value.to(self.device) for key, value in inputs.items() if hasattr(value, "to")}
        with torch.inference_mode():
            output = self.model.get_text_features(**kwargs)
        return feature_tensor(output).detach().float().cpu().numpy()


def _dtype(torch, name: str, device) -> object:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name != "auto":
        raise ValueError(f"unsupported dtype {name}")
    if device.type == "cuda" and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def cached_text(encoder, texts: list[str], path: Path) -> np.ndarray:
    if path.exists():
        return np.load(path)
    vectors = encoder.encode_texts(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, vectors)
    tmp.replace(path)
    return vectors
