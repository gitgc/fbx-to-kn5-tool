"""Reader/writer for Assetto Corsa .kn5 model files (version 6)."""
import struct
from dataclasses import dataclass, field

MAGIC = b"sc6969"


@dataclass
class Texture:
    name: str
    data: bytes
    active: int = 1


@dataclass
class MatProp:
    name: str
    a: float = 0.0
    b: tuple = (0.0, 0.0)
    c: tuple = (0.0, 0.0, 0.0)
    d: tuple = (0.0, 0.0, 0.0, 0.0)


@dataclass
class Material:
    name: str
    shader: str
    blend: int = 0
    alpha_tested: int = 0
    depth_mode: int = 0
    props: list = field(default_factory=list)
    mappings: list = field(default_factory=list)  # (name, slot, texture)


@dataclass
class Node:
    cls: int  # 1 = dummy, 2 = mesh, 3 = skinned mesh
    name: str
    active: int = 1
    children: list = field(default_factory=list)
    matrix: tuple = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    cast_shadows: int = 1
    visible: int = 1
    transparent: int = 0
    vertices: bytes = b""  # packed vertex records
    indices: list = field(default_factory=list)
    material: int = 0
    layer: int = 0
    lod_in: float = 0.0
    lod_out: float = 0.0
    sphere: tuple = (0.0, 0.0, 0.0, 0.0)
    renderable: int = 1
    bones: list = field(default_factory=list)  # (name, matrix)
    skinned_misc: bytes = b"\0" * 8

    @property
    def vertex_size(self):
        return 76 if self.cls == 3 else 44

    @property
    def vertex_count(self):
        return len(self.vertices) // self.vertex_size


@dataclass
class Kn5:
    version: int = 6
    extra: int = 0
    textures: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    root: Node = None


class _R:
    def __init__(self, buf):
        self.b, self.p = memoryview(buf), 0

    def take(self, n):
        v = bytes(self.b[self.p:self.p + n])
        self.p += n
        return v

    def unpack(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.p)
        self.p += struct.calcsize("<" + fmt)
        return v

    def i32(self):
        return self.unpack("i")[0]

    def u8(self):
        return self.unpack("B")[0]

    def f32(self):
        return self.unpack("f")[0]

    def s(self):
        return self.take(self.i32()).decode("utf-8", "replace")


def _read_node(r):
    cls = r.i32()
    n = Node(cls, r.s())
    nchild = r.i32()
    n.active = r.u8()
    if cls == 1:
        n.matrix = r.unpack("16f")
    elif cls in (2, 3):
        n.cast_shadows, n.visible, n.transparent = r.u8(), r.u8(), r.u8()
        if cls == 3:
            for _ in range(r.i32()):
                n.bones.append((r.s(), r.unpack("16f")))
        vc = r.i32()
        n.vertices = r.take(vc * n.vertex_size)
        ic = r.i32()
        n.indices = list(r.unpack(f"{ic}H"))
        n.material, n.layer = r.i32(), r.i32()
        if cls == 2:
            n.lod_in, n.lod_out = r.f32(), r.f32()
            n.sphere = r.unpack("4f")
            n.renderable = r.u8()
        else:
            n.skinned_misc = r.take(8)
    else:
        raise ValueError(f"unknown node class {cls} at {r.p}")
    n.children = [_read_node(r) for _ in range(nchild)]
    return n


def read(path):
    with open(path, "rb") as f:
        r = _R(f.read())
    if r.take(6) != MAGIC:
        raise ValueError("not a kn5 file")
    k = Kn5(version=r.i32())
    if k.version > 5:
        k.extra = r.i32()
    for _ in range(r.i32()):
        active = r.i32()
        name = r.s()
        k.textures.append(Texture(name, r.take(r.i32()), active))
    for _ in range(r.i32()):
        m = Material(r.s(), r.s(), r.u8(), r.u8(), r.i32())
        for _ in range(r.i32()):
            name = r.s()
            a, b0, b1, c0, c1, c2, d0, d1, d2, d3 = r.unpack("10f")
            m.props.append(MatProp(name, a, (b0, b1), (c0, c1, c2), (d0, d1, d2, d3)))
        for _ in range(r.i32()):
            m.mappings.append((r.s(), r.i32(), r.s()))
        k.materials.append(m)
    k.root = _read_node(r)
    if r.p != len(r.b):
        raise ValueError(f"{len(r.b) - r.p} trailing bytes")
    return k


class _W:
    def __init__(self):
        self.parts = []

    def raw(self, b):
        self.parts.append(b)

    def pack(self, fmt, *v):
        self.parts.append(struct.pack("<" + fmt, *v))

    def i32(self, v):
        self.pack("i", v)

    def u8(self, v):
        self.pack("B", v)

    def s(self, v):
        b = v.encode("utf-8")
        self.i32(len(b))
        self.raw(b)


def _write_node(w, n):
    w.i32(n.cls)
    w.s(n.name)
    w.i32(len(n.children))
    w.u8(n.active)
    if n.cls == 1:
        w.pack("16f", *n.matrix)
    else:
        w.u8(n.cast_shadows), w.u8(n.visible), w.u8(n.transparent)
        if n.cls == 3:
            w.i32(len(n.bones))
            for name, m in n.bones:
                w.s(name)
                w.pack("16f", *m)
        w.i32(n.vertex_count)
        w.raw(n.vertices)
        w.i32(len(n.indices))
        w.pack(f"{len(n.indices)}H", *n.indices)
        w.i32(n.material)
        w.i32(n.layer)
        if n.cls == 2:
            w.pack("2f", n.lod_in, n.lod_out)
            w.pack("4f", *n.sphere)
            w.u8(n.renderable)
        else:
            w.raw(n.skinned_misc)
    for c in n.children:
        _write_node(w, c)


def write(k, path):
    w = _W()
    w.raw(MAGIC)
    w.i32(k.version)
    if k.version > 5:
        w.i32(k.extra)
    w.i32(len(k.textures))
    for t in k.textures:
        w.i32(t.active)
        w.s(t.name)
        w.i32(len(t.data))
        w.raw(t.data)
    w.i32(len(k.materials))
    for m in k.materials:
        w.s(m.name)
        w.s(m.shader)
        w.u8(m.blend)
        w.u8(m.alpha_tested)
        w.i32(m.depth_mode)
        w.i32(len(m.props))
        for p in m.props:
            w.s(p.name)
            w.pack("10f", p.a, *p.b, *p.c, *p.d)
        w.i32(len(m.mappings))
        for name, slot, tex in m.mappings:
            w.s(name)
            w.i32(slot)
            w.s(tex)
    _write_node(w, k.root)
    with open(path, "wb") as f:
        for p in w.parts:
            f.write(p)
