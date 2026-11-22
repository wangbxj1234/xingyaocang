"""只解 mcap 里一路相机的 jpeg。深度、位姿、内参都不读。"""

from __future__ import annotations

import io
from pathlib import Path

from google.protobuf import descriptor_pool, message_factory
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from mcap.reader import make_reader
from PIL import Image


def _camera_frame_cls(mcap_path: Path):
    pool = descriptor_pool.DescriptorPool()
    seen = set()
    with mcap_path.open("rb") as handle:
        summary = make_reader(handle).get_summary()
    if summary is None:
        raise ValueError(f"mcap has no summary: {mcap_path}")
    for schema in summary.schemas.values():
        fds = FileDescriptorSet()
        fds.ParseFromString(schema.data)
        for fd in fds.file:
            if fd.name in seen:
                continue
            pool.Add(fd)
            seen.add(fd.name)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("qrdf.v0.CameraFrame"))


def iter_rgb(mcap_path: Path, topic: str, *, epoch_ns: int, start_s: float, end_s: float):
    path = Path(mcap_path)
    CameraFrame = _camera_frame_cls(path)
    kwargs = {
        "topics": [topic],
        "start_time": int(epoch_ns + start_s * 1e9),
        "end_time": int(epoch_ns + end_s * 1e9),
    }
    with path.open("rb") as handle:
        for _schema, _channel, message in make_reader(handle).iter_messages(**kwargs):
            cam = CameraFrame()
            cam.ParseFromString(message.data)
            if cam.encoding.lower() not in ("jpeg", "jpg"):
                raise ValueError(f"unsupported RGB encoding: {cam.encoding}")
            t_s = (int(cam.header.timestamp_ns) - epoch_ns) / 1e9
            if t_s < start_s or t_s > end_s:
                continue
            yield t_s, Image.open(io.BytesIO(cam.data)).convert("RGB")
