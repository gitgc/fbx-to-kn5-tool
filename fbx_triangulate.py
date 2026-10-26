"""Polygon triangulation matching FbxGeometryConverter::TriangulateInPlace (FBX SDK 2014.1).

ksEditor triangulates every mesh with the FBX SDK before building kn5 meshes, so
reproducing its output (diagonals, triangle order and corner order) is needed for
byte-identical kn5 files. For each polygon of four or more corners the SDK:

1. computes a Newell normal and projects the polygon onto the dominant axis plane,
2. builds a Delaunay triangulation inside a large "super triangle" by incremental
   insertion with Lawson edge flips on a half-edge mesh,
3. checks the polygon edges exist, removes faces outside the polygon, and
4. emits the remaining faces in face-list order, each from its first half-edge.

If that fails it falls back to a fixed zig-zag pattern. Degenerate polygons (fewer
than three distinct projected points, or all collinear) produce no triangles.
All arithmetic is double precision, in the same order as the SDK.
"""
import math
import struct

EPS = 2.220446049250313e-16
SQRT3 = struct.unpack("<d", bytes.fromhex("3b3d58e87ab6fb3f"))[0]


class _V:
    __slots__ = ("x", "y", "edge", "index")

    def __init__(self, x, y, index):
        self.x, self.y, self.edge, self.index = x, y, None, index


class _E:
    __slots__ = ("origin", "next", "prev", "twin", "face")

    def __init__(self):
        self.origin = self.next = self.prev = self.twin = self.face = None


class _F:
    __slots__ = ("edge",)

    def __init__(self):
        self.edge = None


def _around(v):
    # Outgoing edges of v, starting at v.edge: rotate one way, then the other way
    # from v.edge if a boundary is hit.
    e, forward = v.edge, True
    while e is not None:
        yield e
        if forward:
            e2 = e.prev.twin
            if e2 is None:
                forward = False
                t = v.edge.twin
                e = t.next if t is not None else None
            elif e2 is v.edge:
                e = None
            else:
                e = e2
        else:
            e = e.twin.next if e.twin is not None else None


def _find_edge(a, b):
    for e in _around(a):
        if e.next.origin is b:
            return e
    return None


def _loop(f):
    e = f.edge
    out = [e]
    e = e.next
    while e is not f.edge:
        out.append(e)
        e = e.next
    return out


class _Mesh:
    def __init__(self):
        self.faces = []

    def make_face(self, v0, v1, v2):
        f = _F()
        self.faces.append(f)
        e0, e1, e2 = _E(), _E(), _E()
        f.edge = e0
        for e in (e0, e1, e2):
            e.face = f
        e0.next, e1.next, e2.next = e1, e2, e0
        e0.prev, e1.prev, e2.prev = e2, e0, e1
        e0.origin, e1.origin, e2.origin = v0, v1, v2
        for v, e in ((v0, e0), (v1, e1), (v2, e2)):
            if v.edge is None:
                v.edge = e
        for e, a, b in ((e0, v1, v0), (e1, v2, v1), (e2, v0, v2)):
            t = _find_edge(a, b)
            if t is not None:
                e.twin, t.twin = t, e
        return f

    def split(self, f, v):
        """Insert vertex v inside face f, fanning it into one triangle per edge."""
        e0 = f.edge
        prev_cur = None
        cur = e0
        while True:
            nxt = cur.next
            if cur is e0:
                face = f
            else:
                face = _F()
                self.faces.append(face)
            face.edge = cur
            a, b = _E(), _E()
            if cur is e0:
                v.edge = b
            a.origin = nxt.origin
            b.origin = v
            cur.next, a.next, b.next = a, b, cur
            cur.prev, a.prev, b.prev = b, cur, a
            cur.face = a.face = b.face = face
            if cur is not e0:
                prev_a = prev_cur.next
                prev_a.twin, b.twin = b, prev_a
            if nxt is e0:
                break
            prev_cur = cur
            cur = nxt
        last_a, first_b = cur.next, e0.prev
        last_a.twin, first_b.twin = first_b, last_a

    def flip(self, e):
        ep, en, t = e.prev, e.next, e.twin
        tp, tn = t.prev, t.next
        c, d = ep.origin, tp.origin
        if _find_edge(c, d) is not None or _find_edge(d, c) is not None:
            return False
        en.prev = tp
        en.origin.edge = en
        tp.next = en
        tp.origin.edge = tp
        tn.prev = ep
        tn.origin.edge = tn
        ep.next = tn
        ep.origin.edge = ep
        e.origin = c
        c.edge = e
        e.next, e.prev = tp, en
        en.next = e
        tp.prev = e
        en.face = tp.face = e.face
        e.face.edge = e
        t.origin = d
        d.edge = t
        t.next, t.prev = ep, tn
        tn.next = t
        ep.prev = t
        tn.face = ep.face = t.face
        t.face.edge = t
        return True

    def split_face(self, f, v1, v2):
        """Add the diagonal v1->v2 across polygon face f; f keeps the v1->v2 side."""
        p1 = _edge_from(f, v1)
        p2 = _edge_from(f, v2)
        e1, e2 = _E(), _E()
        h = _F()
        self.faces.append(h)
        f.edge, h.edge = e1, e2
        e1.origin, e2.origin = v1, v2
        e1.next, e2.next = p2, p1
        e1.prev, e2.prev = p1.prev, p2.prev
        p1.prev.next = e1
        p2.prev.next = e2
        p2.prev = e1
        p1.prev = e2
        e1.face, e2.face = f, h
        e = p1
        while e is not e2:
            e.face = h
            e = e.next
        e1.twin, e2.twin = e2, e1
        return e1

    def delete_edge(self, e):
        """Remove e and its twin, merging the twin's face into e's face."""
        t = e.twin
        if t is None:
            return False
        g = t.face
        for x in _loop(g):
            x.face = e.face
        if e.face.edge is e:
            e.face.edge = e.next
        for x in (e, t):
            v = x.origin
            if v.edge is x:
                repl = None
                for o in _around(v):
                    if o is not x:
                        repl = o
                        break
                v.edge = repl
        ep, en, tn, tp = e.prev, e.next, t.next, t.prev
        ep.next = tn
        en.prev = tp
        tn.prev = ep
        tp.next = en
        self.faces.remove(g)
        return True

    def delete_face(self, f):
        loop = _loop(f)
        for e in loop:
            v = e.origin
            if v.edge is e:
                repl = None
                for o in _around(v):
                    if o is not e:
                        repl = o
                        break
                v.edge = repl
        for e in loop:
            if e.twin is not None:
                e.twin.twin = None
        self.faces.remove(f)


def _illegal(e):
    t = e.twin
    if t is None:
        return False
    a, b = e.origin, e.next.origin
    c, d = e.prev.origin, t.prev.origin
    d13 = a.y - c.y
    d10 = a.y - d.y
    d11 = b.y - c.y
    d12 = b.y - d.y
    cos_c = d11 * d13 + (b.x - c.x) * (a.x - c.x)
    cos_d = d12 * d10 + (b.x - d.x) * (a.x - d.x)
    if cos_c < 0.0 or cos_d < 0.0:
        if cos_c < 0.0 and cos_d < 0.0:
            return True
        return (d10 * (b.x - d.x) - d12 * (a.x - d.x)) * cos_c + \
            (d11 * (a.x - c.x) - d13 * (b.x - c.x)) * cos_d < 0.0
    return False


def _side(px, py, ax, ay, bx, by):
    # Orientation of p against segment a-b, evaluated from the lexicographically
    # smaller end point first: 1 or 2 for either side, 0 on the line.
    if ax <= bx and (ax != bx or ay <= by):
        d = (ay - by) * (px - bx) - (py - by) * (ax - bx)
        return 1 if d > 0.0 else (2 if d < 0.0 else 0)
    d = (by - ay) * (px - ax) - (py - ay) * (bx - ax)
    return 2 if d > 0.0 else (1 if d < 0.0 else 0)


def _classify(px, py, t):
    """1 inside/on the triangle, 2/3/4 outside edge 0/1/2, -1 degenerate."""
    x0, y0, x1, y1, x2, y2 = t
    if x1 < x0 or (x0 == x1 and y1 < y0):
        d = (px - x0) * (y1 - y0) - (x1 - x0) * (py - y0)
        s0 = 2 if d > 0.0 else (1 if d < 0.0 else 0)
    else:
        d = (y0 - y1) * (px - x1) - (x0 - x1) * (py - y1)
        s0 = 1 if d > 0.0 else (2 if d < 0.0 else 0)
    if s0 == 2:
        return 2
    if x2 < x1 or (x1 == x2 and y2 < y1):
        d = (y2 - y1) * (px - x1) - (x2 - x1) * (py - y1)
        s1 = 2 if d > 0.0 else (1 if d < 0.0 else 0)
    else:
        d = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
        s1 = 1 if d > 0.0 else (2 if d < 0.0 else 0)
    if s1 == 2:
        return 3
    s2 = _side(px, py, x2, y2, x0, y0)
    if s2 == 2:
        return 4
    if s2 == 0 and s1 == 0 and s0 == 0:
        return -1
    return 1


def _tri(f):
    e0 = f.edge
    e1 = e0.next
    e2 = e1.next
    return (e0.origin.x, e0.origin.y, e1.origin.x, e1.origin.y, e2.origin.x, e2.origin.y), (e0, e1, e2)


def _locate(mesh, px, py, start):
    limit = len(mesh.faces)
    steps = 0
    f = start
    while True:
        coords, edges = _tri(f)
        c = _classify(px, py, coords)
        if c == -1:
            break
        if c == 1:
            return f
        f = edges[c - 2].twin.face
        steps += 1
        if steps > limit:
            break
    for f in mesh.faces:
        if _classify(px, py, _tri(f)[0]) == 1:
            return f
    return None


def _insertion_order(n):
    return list(range(0, n, 5)) + [i for i in range(n) if i % 5]


def _degenerate(pts):
    uniq = sorted(set(pts))
    if len(uniq) < 3:
        return True
    (ox, oy), (qx, qy) = uniq[0], uniq[1]
    dx, dy = qx - ox, qy - oy
    ln2 = dy * dy + dx * dx
    if ln2 > 0.0:
        inv = 1.0 / math.sqrt(ln2)
        dx, dy = dx * inv, dy * inv
    else:
        dx = dy = 0.0
    for px, py in uniq[2:]:
        wx, wy = px - ox, py - oy
        dot = wy * dy + wx * dx
        rx, ry = wx - dx * dot, wy - dy * dot
        if math.sqrt(rx * rx + ry * ry) > 0.0:
            return False
    return True


def _delaunay(pts):
    """Constrained Delaunay triangulation of one projected polygon.

    Returns (ok, triangles); ok is False when the SDK would fall back.
    """
    n = len(pts)
    if _degenerate(pts):
        return True, []
    mesh = _Mesh()
    minx, miny = pts[0]
    maxx, maxy = pts[0]
    for x, y in pts[1:]:
        if minx <= maxx:
            if x < minx:
                minx = x
            if y < miny:
                miny = y
            if maxx < x:
                maxx = x
            if y > maxy:
                maxy = y
    cx, cy = (maxx + minx) * 0.5, (maxy + miny) * 0.5
    r = math.sqrt((cy - maxy) * (cy - maxy) + (cx - maxx) * (cx - maxx))
    d = r * 2.0
    top = _V(cx + d * 0.0, cy + d * 2.0, n)
    left = _V(cx + d * -SQRT3, cy + d * -1.0, n + 1)
    right = _V(cx + d * SQRT3, cy + d * -1.0, n + 2)
    mesh.make_face(top, left, right)

    verts = [None] * n
    order = _insertion_order(n)
    first = order[0]
    v = _V(pts[first][0], pts[first][1], first)
    mesh.split(mesh.faces[0], v)
    verts[first] = v
    start = v.edge.face
    for i in order[1:]:
        px, py = pts[i]
        f = _locate(mesh, px, py, start)
        if f is None:
            return False, []
        v = _V(px, py, i)
        mesh.split(f, v)
        verts[i] = v
        start = v.edge.face
        ok = True
        for e in list(_around(v)):
            ok = _legalize(mesh, e.next, v)
        if not ok:
            return False, []

    loop = list(range(n)) + [0]
    for a, b in zip(loop, loop[1:]):
        if a != b and _find_edge(verts[a], verts[b]) is None:
            if not _insert_edge(mesh, verts[a], verts[b]) and n != 1:
                return False, []

    marked = []
    edges = [_find_edge(verts[loop[k]], verts[loop[k + 1]]) for k in range(n)]
    for k in range(n):
        e_in, e_out = edges[k - 1], edges[k]
        if e_in is None or e_out is None:
            continue
        e = e_in.twin
        while e is not None:
            if e.face not in marked:
                marked.append(e.face)
            e = e.prev.twin
            if e is e_out:
                break
    for f in marked:
        mesh.delete_face(f)

    tris = []
    for f in mesh.faces:
        lp = _loop(f)
        if len(lp) == 3:
            ids = [e.origin.index for e in lp]
            if all(0 <= i < n for i in ids):
                tris.append(tuple(ids))
    return len(tris) > 0, tris


def _side_of(px, py, e):
    a, b = e.origin, e.next.origin
    if a.x > b.x or (a.x == b.x and a.y > b.y):
        d = (b.y - a.y) * (px - a.x) - (py - a.y) * (b.x - a.x)
        return 2 if d > 0.0 else (1 if d < 0.0 else 0)
    d = (a.y - b.y) * (px - b.x) - (py - b.y) * (a.x - b.x)
    return 1 if d > 0.0 else (2 if d < 0.0 else 0)


def _crosses(e, va, vb):
    """Does segment va-vb meet edge e? (The SDK's bounding-box reject, then its
    segment intersection test.)"""
    a, b = e.origin, e.next.origin
    minx, maxx = (a.x, b.x) if a.x <= b.x else (b.x, a.x)
    miny, maxy = (a.y, b.y) if a.y <= b.y else (b.y, a.y)

    def code(x, y):
        c = 1 if x < minx else (2 if x > maxx else 0)
        return c + (4 if y < miny else (8 if y > maxy else 0))

    if code(va.x, va.y) & code(vb.x, vb.y):
        return False

    hit, touching = _segment_hit((a.x, a.y, 0.0), (b.x, b.y, 0.0), (va.x, va.y, 0.0), (vb.x, vb.y, 0.0))
    return hit and touching


# 3D vector helpers in the SDK's operand order (points here have z = 0).

def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], b[0] * a[2] - a[0] * b[2], a[0] * b[1] - b[0] * a[1])


def _unit3(v):
    s = (v[1] * v[1] + v[0] * v[0]) + v[2] * v[2]
    if s > 0.0:
        inv = 1.0 / math.sqrt(s)
        return (v[0] * inv, v[1] * inv, v[2] * inv), True
    return (0.0, 0.0, 0.0), False


def _param(p, o, d):
    """Parameter of p's projection on the line o + t*d."""
    v = _sub3(p, o)
    ln = math.sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2])
    return ((d[1] * v[1] + d[0] * v[0]) + d[2] * v[2]) / (ln * ln)


def _at(o, d, t):
    return (o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t)


def _segment_hit(a, b, s0, s1, eps=1e-6):
    """The SDK's segment-segment intersection of a-b with s0-s1.

    Returns (result, touching); the constraint step counts a crossing only when
    both are true.
    """
    d1, d2 = _sub3(b, a), _sub3(s1, s0)
    u1, ok1 = _unit3(d1)
    u2, ok2 = _unit3(d2)
    if not ok1 or not ok2:
        return False, False
    t0 = _param(s0, a, d1)
    p0 = _at(a, d1, t0)
    t1 = _param(s1, a, d1)
    p1 = _at(a, d1, t1)
    dy, dx, dz = s0[1] - p0[1], s0[0] - p0[0], s0[2] - p0[2]
    off0 = abs(((dy * dy + dx * dx) + dz * dz) - 0.0)
    dy, dx, dz = s1[1] - p1[1], s1[0] - p1[0], s1[2] - p1[2]
    off1 = abs(((dy * dy + dx * dx) + dz * dz) - 0.0)
    e2 = eps * eps
    if e2 > off0 and e2 > off1:
        # Collinear: overlap of the projected range with [0, 1].
        lo = hi = t0
        if t1 < lo:
            lo = t1
        if hi < t1:
            hi = t1
        if hi <= 0.0 - 0.0 or 1.0 + 0.0 <= lo:
            return False, False
        lo = lo if lo > 0.0 else 0.0
        hi = 1.0 if hi > 1.0 else hi
        touching = False
        if 0.001 > hi - lo:
            q0, q1 = _at(a, d1, lo), _at(a, d1, hi)
            touching = all(abs(q0[i] - q1[i]) < eps for i in range(3))
        return True, touching
    # Intersect the s0-s1 line with the plane through a-b perpendicular to both.
    n, _ = _unit3(_cross3(_cross3(u1, u2), u1))
    k = -((n[1] * a[1] + n[0] * a[0]) + n[2] * a[2])
    denom = (u2[1] * n[1] + n[0] * u2[0]) + u2[2] * n[2]
    num = ((n[1] * s0[1] + s0[0] * n[0]) + n[2] * s0[2]) + k
    try:
        t = -(num / denom)
    except ZeroDivisionError:
        t = math.copysign(math.inf, num) * -math.copysign(1.0, denom) if num != 0.0 else math.nan
    x = (s0[0] + u2[0] * t, s0[1] + u2[1] * t, s0[2] + u2[2] * t)
    ta, tb = _param(x, a, d1), _param(x, s0, d2)

    def near(q):
        return abs(x[0] - q[0]) < eps and abs(x[1] - q[1]) < eps

    end_a = (0.0 > ta and near(a)) or (ta > 1.0 and near(b))
    end_b = (0.0 > tb and near(s0)) or (tb > 1.0 and near(s1))
    if not end_a and not (0.0 <= ta <= 1.0):
        return False, True
    if not end_b and not (0.0 <= tb <= 1.0):
        return False, True
    return True, True


def _fan(v):
    """Outgoing edges around an interior vertex; empty if v is on a boundary."""
    out = []
    e = v.edge
    while e is not None:
        out.append(e)
        if e.face is None or len(_loop(e.face)) != 3 or e.prev is None:
            return []
        e = e.prev.twin
        if e is None:
            return []
        if e is v.edge:
            return out
    return out


def _insert_edge(mesh, va, vb):
    """Force the edge va-vb into the triangulation (the SDK's constraint step)."""
    fan = _fan(va)
    face = cross = None
    if fan:
        prev_edge, prev_side = fan[0], _side_of(vb.x, vb.y, fan[0])
        for k in range(1, len(fan) + 1):
            e = fan[k % len(fan)]
            side = _side_of(vb.x, vb.y, e)
            if prev_side == 1 and side == 2:
                cross = prev_edge.next
                face = cross.face
                break
            prev_edge, prev_side = e, side
    if face is None or cross is None:
        return True
    if _edge_or_none(face, va) is None or _edge_or_none(face, vb) is not None:
        return False
    crossed = []
    limit = sum(len(_loop(f)) for f in mesh.faces)
    while True:
        crossed.append(cross)
        if len(crossed) > limit:
            return False
        t = cross.twin
        if t is None or t.face is None:
            return False
        if _edge_or_none(t.face, vb) is not None:
            break
        if _crosses(t.next, va, vb):
            cross = t.next
        elif _crosses(t.prev, va, vb):
            cross = t.prev
        else:
            return False
    if len(crossed) == 1:
        mesh.flip(crossed[0])
    else:
        merged = crossed[0].face
        for e in crossed:
            mesh.delete_edge(e)
        e1 = mesh.split_face(merged, va, vb)
        ok1 = _retriangulate(mesh, e1)
        ok2 = _retriangulate(mesh, e1.twin) if e1.twin is not None else False
        if not ok1 or not ok2:
            return False
    return _find_edge(va, vb) is not None


def _retriangulate(mesh, e):
    """Triangulate the polygon face of base edge e by the in-circle rule."""
    f = e.face
    if len(_loop(f)) == 3:
        return True
    a, b = e.origin, e.next.origin
    cur = e.next.next
    best = cur.origin
    while cur is not e.prev:
        cur = cur.next
        c = cur.origin
        ax, ay = a.x - c.x, a.y - c.y
        bx, by = b.x - c.x, b.y - c.y
        px, py = best.x - c.x, best.y - c.y
        det = ((by * by + bx * bx) * (px * ay - py * ax) + (py * bx - px * by) * (ay * ay + ax * ax)) + \
            (py * py + px * px) * (by * ax - bx * ay)
        if not (0.0 > abs(det) or 0.0 > det):
            best = c
    sub = []
    if _find_edge(b, best) is None:
        d = mesh.split_face(f, b, best)
        f = d.face
        sub.append(d.twin)
    else:
        sub.append(None)
    if _find_edge(best, a) is None:
        d = mesh.split_face(f, best, a)
        f = d.face
        sub.append(d.twin)
    else:
        sub.append(None)
    if len(_loop(f)) != 3:
        return False
    ok = True
    if sub[0] is not None:
        ok = _retriangulate(mesh, sub[0])
    if sub[1] is not None and ok:
        ok = _retriangulate(mesh, sub[1])
    return ok


def _edge_or_none(f, v):
    for e in _loop(f):
        if e.origin is v:
            return e
    return None


def _legalize(mesh, e, p):
    if not _illegal(e):
        return True
    if not mesh.flip(e):
        return False
    _legalize(mesh, _edge_from(e.face, p).next, p)
    return _legalize(mesh, _edge_from(e.twin.face, p).next, p)


def _edge_from(f, v):
    for e in _loop(f):
        if e.origin is v:
            return e
    raise RuntimeError("triangulation lost track of the inserted vertex")


def _zigzag(n):
    out = [0, 1, 2]
    hi, lo, a, nxt = n - 1, 2, 0, 3
    if 2 < hi:
        while True:
            cur = nxt
            out += [lo, cur, a]
            if len(out) == (n - 2) * 3:
                break
            out += [a, cur, hi]
            nhi = hi - 1
            nxt = cur + 1
            a, lo, hi = hi, cur, nhi
            if not cur + 1 <= nhi:
                break
    return [tuple(out[i:i + 3]) for i in range(0, len(out), 3)]


def _project(pts3):
    nx = ny = nz = 0.0
    n = len(pts3)
    for i in range(n):
        p, q = pts3[i], pts3[(i + 1) % n]
        nx = (q[2] + p[2]) * (p[1] - q[1]) + nx
        ny = (q[0] + p[0]) * (p[2] - q[2]) + ny
        nz = (p[0] - q[0]) * (q[1] + p[1]) + nz
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if ln == 0.0:
        nx = ny = nz = 0.0
    else:
        nx, ny, nz = nx / ln, ny / ln, nz / ln
    az, ay, ax = abs(nz), abs(ny), abs(nx)
    axis = 1
    if az - ay <= EPS or az - ax <= EPS:
        if ay - ax <= EPS or ay - az <= EPS:
            if ax - ay <= EPS or ax - az <= EPS:
                if az - ay <= EPS:
                    if EPS < ax - az:
                        axis = 0
                else:
                    axis = 2
            else:
                axis = 0
    else:
        axis = 2
    if nz > EPS and axis == 2:
        case = 0
    elif -nz > EPS and axis == 2:
        case = 1
    elif ny > EPS and axis == 1:
        case = 2
    elif -ny > EPS and axis == 1:
        case = 3
    elif nx > EPS and axis == 0:
        case = 4
    elif -nx > EPS and axis == 0:
        case = 5
    else:
        case = 2
    if case == 0:
        return [(p[0], p[1]) for p in pts3]
    if case == 1:
        return [(p[0], -p[1]) for p in pts3]
    if case == 2:
        return [(-p[0], p[2]) for p in pts3]
    if case == 3:
        return [(p[0], p[2]) for p in pts3]
    if case == 4:
        return [(p[1], p[2]) for p in pts3]
    return [(-p[1], p[2]) for p in pts3]


def triangulate(pts3):
    """Triangulate one polygon given its corner positions (doubles).

    Returns a list of corner index triples in FBX SDK order.
    """
    n = len(pts3)
    if n < 4:
        return [tuple(range(n))] if n == 3 else []
    ok, tris = _delaunay(_project(pts3))
    return tris if ok else _zigzag(n)
