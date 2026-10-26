"""Mesh building as done by ksEditor's ksNet.dll, with its float32 arithmetic.

After the FBX SDK triangulates a mesh, ksNet:

* transforms control points and polygon-corner normals by the node's geometric
  matrix in float32,
* for each triangle corner, picks the batch for the polygon's material (batches in
  order of first use), normalises the normal, and merges the corner into the batch
  by exact float comparison of position, normal and UV (first match wins),
* computes one tangent per triangle and writes it to its three corners, so the
  last triangle using a vertex decides that vertex's tangent,
* computes the bounding sphere from the vertex average and the farthest vertex.

ksNet does this in single-precision SSE code. Every operation below is rounded to
float32 in the same order, which keeps the output bit-identical to ksEditor's.
"""
import math
from array import array

import fbx_triangulate


def r32(values):
    """Round a sequence of floats to float32."""
    return array("f", values).tolist()


def f32(x):
    return array("f", (x,))[0]


def _add(a, b):
    return r32([p + q for p, q in zip(a, b)])


def _sub(a, b):
    return r32([p - q for p, q in zip(a, b)])


def _mul(a, b):
    return r32([p * q for p, q in zip(a, b)])


def _pick(mask, a, b):
    return [p if m else q for m, p, q in zip(mask, a, b)]


def _scale_nonzero(lengths, cols):
    """Multiply each column by 1/length where length != 0 (ksNet's normalise)."""
    inv = r32([1.0 / v if v != 0.0 else 0.0 for v in lengths])
    return [r32([i * c if v != 0.0 else c for i, c, v in zip(inv, col, lengths)]) for col in cols]


def _length(xs, ys, zs):
    """sqrt((x*x + y*y) + z*z) in float32."""
    return r32([math.sqrt(v) for v in _add(_add(_mul(xs, xs), _mul(ys, ys)), _mul(zs, zs))])


def transform(rows, pts, translate):
    """Row-vector transform ((m1*y + x*m0) + m2*z) + m3, per output column."""
    xs = r32([p[0] for p in pts])
    ys = r32([p[1] for p in pts])
    zs = r32([p[2] for p in pts])
    out = []
    for c in range(3):
        m0, m1, m2, m3 = rows[0][c], rows[1][c], rows[2][c], rows[3][c]
        s = _add(r32([m1 * y for y in ys]), r32([x * m0 for x in xs]))
        s = _add(s, r32([m2 * z for z in zs]))
        if translate:
            s = r32([v + m3 for v in s])
        out.append(s)
    return list(zip(*out))


def normalise(vs):
    xs = [v[0] for v in vs]
    ys = [v[1] for v in vs]
    zs = [v[2] for v in vs]
    return list(zip(*_scale_nonzero(_length(xs, ys, zs), (xs, ys, zs))))


class Batch:
    __slots__ = ("material", "pos", "nrm", "uv", "cp", "indices", "_lookup")

    def __init__(self, material):
        self.material = material
        self.pos, self.nrm, self.uv, self.cp, self.indices = [], [], [], [], []
        self._lookup = {}

    def add(self, pos, nrm, uv, cp):
        # Python float keys compare like ksNet's == (0.0 == -0.0); the first
        # matching vertex is kept, as in its linear search.
        key = pos + nrm + uv
        i = self._lookup.get(key)
        if i is None:
            i = self._lookup[key] = len(self.pos)
            self.pos.append(pos)
            self.nrm.append(nrm)
            self.uv.append(uv)
            self.cp.append(cp)
        self.indices.append(i)


def build_batches(cps, pv_cp, polys, normals, uvs, materials, per_corner, rows):
    """Triangulate one FBX mesh and merge its corners into per-material batches.

    cps: control points; pv_cp: control point of each polygon-vertex; polys: lists
    of polygon-vertex indices; normals, uvs: per polygon-vertex (uvs None when the
    mesh has no UV set); materials: material slot per polygon; per_corner: whether
    normals and UVs are mapped by polygon-vertex; rows: geometric matrix rows as
    float32. Returns the batches in creation order.
    """
    corners, attr_n, attr_uv, corner_mat = [], [], [], []
    for pi, poly in enumerate(polys):
        tris = fbx_triangulate.triangulate([cps[pv_cp[i]] for i in poly])
        # The SDK copies per-corner data onto the new triangles: a polygon left with
        # a single triangle gets its first three corners in order; otherwise each
        # corner takes the first polygon corner on the same control point.
        if len(tris) == 1:
            seq = poly[:3]
        else:
            first = {}
            for i in poly:
                first.setdefault(pv_cp[i], i)
        for t in tris:
            for j, k in enumerate(t):
                pv = poly[k]
                src = seq[j] if len(tris) == 1 else first[pv_cp[pv]]
                corners.append(pv)
                attr_n.append(src if per_corner[0] else pv)
                attr_uv.append(src if per_corner[1] else pv)
                corner_mat.append(materials[pi])
    pos = transform(rows, cps, True)
    nrm = normalise(transform(rows, [normals[i] for i in attr_n], False))
    if uvs is None:
        uv = [(0.0, 0.0)] * len(corners)
    else:
        uv = list(zip(r32([uvs[i][0] for i in attr_uv]), r32([-uvs[i][1] for i in attr_uv])))
    batches = {}
    for i, pv in enumerate(corners):
        b = batches.get(corner_mat[i])
        if b is None:
            b = batches[corner_mat[i]] = Batch(corner_mat[i])
        cp = pv_cp[pv]
        b.add(pos[cp], nrm[i], uv[i], cp)
    return list(batches.values())


def tangents(batch):
    """Per-vertex tangents, last triangle wins (ksNet's tangent pass)."""
    X, Y, Z = (list(c) for c in zip(*batch.pos)) if batch.pos else ([], [], [])
    NX, NY, NZ = (list(c) for c in zip(*batch.nrm)) if batch.nrm else ([], [], [])
    U = [t[0] for t in batch.uv]
    V = [t[1] for t in batch.uv]
    idx = batch.indices
    ii = (idx[0::3], idx[1::3], idx[2::3])

    def g(col, k):
        return [col[i] for i in ii[k]]

    du1 = _sub(g(U, 1), g(U, 0))
    dv2 = _sub(g(V, 2), g(V, 0))
    dv1 = _sub(g(V, 1), g(V, 0))
    du2 = _sub(g(U, 2), g(U, 0))
    T, B = [], []
    for col in (X, Y, Z):
        e1 = _sub(g(col, 1), g(col, 0))
        e2 = _sub(g(col, 2), g(col, 0))
        T.append(_sub(_mul(e1, dv2), _mul(e2, dv1)))
        B.append(_sub(_mul(e2, du1), _mul(e1, du2)))
    tx_, ty_, tz_ = T
    bx, by, bz = B
    inv3 = f32(1.0 / f32(math.sqrt(3.0)))
    per_corner = []
    for k in range(3):
        nx, ny, nz = g(NX, k), g(NY, k), g(NZ, k)
        cx = _sub(_mul(ny, tz_), _mul(nz, ty_))
        cy = _sub(_mul(nz, tx_), _mul(nx, tz_))
        cz = _sub(_mul(nx, ty_), _mul(ny, tx_))
        cz, cx, cy = _scale_nonzero(_length(cy, cx, cz), (cz, cx, cy))
        zero = [a == 0.0 and b == 0.0 and c == 0.0 for a, b, c in zip(cx, cy, cz)]
        tx = _pick(zero, _sub(_mul(nz, by), _mul(ny, bz)), _sub(_mul(nz, cy), _mul(ny, cz)))
        tz = _pick(zero, _sub(_mul(ny, bx), _mul(nx, by)), _sub(_mul(ny, cx), _mul(nx, cy)))
        a = _pick(zero, bx, cx)
        c = _pick(zero, bz, cz)
        ty = _sub(_mul(nx, c), _mul(nz, a))
        tx, tz, ty = _scale_nonzero(_length(ty, tx, tz), (tx, tz, ty))
        per_corner.append([(inv3, inv3, inv3) if p == 0.0 and q == 0.0 and r == 0.0 else (p, q, r)
                           for p, q, r in zip(tx, ty, tz)])
    out = [(0.0, 0.0, 0.0)] * len(batch.pos)
    for t in range(len(ii[0])):
        for k in range(3):
            out[ii[k][t]] = per_corner[k][t]
    return out


def bounding_sphere(points):
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    sx = sy = sz = 0.0
    for x, y, z in points:
        sx = f32(sx + x)
        sy = f32(y + sy)
        sz = f32(z + sz)
    inv = f32(1.0 / f32(float(len(points))))
    cx, cy, cz = f32(sx * inv), f32(sy * inv), f32(sz * inv)
    best = 0.0
    for x, y, z in points:
        dy, dx, dz = f32(y - cy), f32(x - cx), f32(z - cz)
        s = f32(f32(f32(dy * dy) + f32(dx * dx)) + f32(dz * dz))
        d = 0.0 if s == 0.0 else f32(math.sqrt(s))
        if best < d:
            best = d
    return (cx, cy, cz, best)
