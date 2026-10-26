"""Minimal binary FBX (7.x) reader: parses the node tree into FbxNode objects."""
import struct
import zlib

_ARRAY_TYPES = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "?"}
_SCALAR = {"Y": "h", "C": "?", "I": "i", "F": "f", "D": "d", "L": "q"}


class FbxNode:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name, self.props, self.children = name, props, children

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_all(self, name):
        return [c for c in self.children if c.name == name]

    def __repr__(self):
        return f"FbxNode({self.name!r}, {self.props[:3]!r}, {len(self.children)} children)"


def _read_props(buf, p, n):
    props = []
    for _ in range(n):
        t = chr(buf[p])
        p += 1
        if t in _SCALAR:
            fmt = "<" + _SCALAR[t]
            props.append(struct.unpack_from(fmt, buf, p)[0])
            p += struct.calcsize(fmt)
        elif t in _ARRAY_TYPES:
            count, enc, clen = struct.unpack_from("<III", buf, p)
            p += 12
            data = bytes(buf[p:p + clen])
            p += clen
            if enc == 1:
                data = zlib.decompress(data)
            fmt = _ARRAY_TYPES[t]
            props.append(list(struct.unpack(f"<{count}{fmt}", data)))
        elif t in "SR":
            (ln,) = struct.unpack_from("<I", buf, p)
            p += 4
            data = bytes(buf[p:p + ln])
            p += ln
            props.append(data.decode("utf-8", "replace") if t == "S" else data)
        else:
            raise ValueError(f"unknown FBX property type {t!r} at {p - 1}")
    return props, p


def _read_node(buf, p, wide):
    if wide:
        end, nprops, _plen = struct.unpack_from("<QQQ", buf, p)
        p += 24
    else:
        end, nprops, _plen = struct.unpack_from("<III", buf, p)
        p += 12
    nlen = buf[p]
    p += 1
    if end == 0:
        return None, p
    name = bytes(buf[p:p + nlen]).decode("ascii")
    p += nlen
    props, p = _read_props(buf, p, nprops)
    children = []
    sentinel = 25 if wide else 13
    while p < end:
        if end - p == sentinel:
            p = end
            break
        child, p = _read_node(buf, p, wide)
        if child is None:
            break
        children.append(child)
    return FbxNode(name, props, children), end


def read(path):
    with open(path, "rb") as f:
        buf = memoryview(f.read())
    if bytes(buf[:20]) != b"Kaydara FBX Binary  ":
        raise ValueError("only binary FBX files are supported")
    (version,) = struct.unpack_from("<I", buf, 23)
    wide = version >= 7500
    p = 27
    top = []
    while p < len(buf):
        node, p = _read_node(buf, p, wide)
        if node is None:
            break
        top.append(node)
    return version, FbxNode("", [], top)
