"""Decode Assetto Corsa textures (DDS, PNG, JPG) to RGBA arrays."""
import io
import struct

import numpy as np
from PIL import Image

_DDPF_ALPHAPIXELS = 0x1
_DDPF_FOURCC = 0x4
_DDPF_LUMINANCE = 0x20000


def decode(data):
    """Return an (height, width, 4) uint8 RGBA array, top row first."""
    if data[:4] == b"DDS " and len(data) >= 128:
        flags, fourcc, bits = struct.unpack_from("<I4sI", data, 80)
        if not flags & _DDPF_FOURCC:
            return _decode_uncompressed(data, flags, bits)
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGBA"))


def _decode_uncompressed(data, flags, bits):
    height, width = struct.unpack_from("<II", data, 12)
    masks = struct.unpack_from("<4I", data, 92)
    bpp = bits // 8
    raw = np.frombuffer(data, np.uint8, width * height * bpp, 128).reshape(-1, bpp).astype(np.uint32)
    pixels = np.zeros(len(raw), np.uint32)
    for i in range(bpp):
        pixels |= raw[:, i] << (8 * i)

    def channel(mask, default):
        if not mask:
            return np.full(len(pixels), default, np.uint8)
        shift = (mask & -mask).bit_length() - 1
        top = mask >> shift
        return ((pixels & mask) >> shift) * 255 // top

    r, g, b, a = masks
    if flags & _DDPF_LUMINANCE:
        lum = channel(r, 0)
        rgba = [lum, lum, lum, channel(a, 255) if flags & _DDPF_ALPHAPIXELS else np.full(len(pixels), 255)]
    else:
        rgba = [channel(r, 0), channel(g, 0), channel(b, 0), channel(a, 255) if a else np.full(len(pixels), 255)]
    return np.stack(rgba, axis=1).astype(np.uint8).reshape(height, width, 4)
