"""Car scene for skin previews, set up the way Content Manager's preview renderer does.

Matrices are numpy 4x4 in the row-vector convention used by kn5 and Direct3D
(p' = p @ M, translation in the last row).
"""
import configparser
import math
import os

import numpy as np

import kn5

# Preview framing, after Content Manager's built-in "Kunos" preset.
PREVIEW_WIDTH, PREVIEW_HEIGHT = 1022, 575
SSAA = 4
CAMERA_POSITION = (-3.8676, 1.4236, 4.7038)
CAMERA_LOOK_AT = (0.0, 0.7, 0.5)
CAMERA_FOV_DEG = 30.0


def translation(x, y, z):
    m = np.eye(4)
    m[3, :3] = (x, y, z)
    return m


def scaling(x, y, z):
    return np.diag([x, y, z, 1.0])


def rotation_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]], float)


def rotation_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0], [0, 0, 0, 1]], float)


def look_at_lh(eye, target, up=(0.0, 1.0, 0.0)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = x, y, z
    m[3, :3] = (-x @ eye, -y @ eye, -z @ eye)
    return m


def look_at_rh(eye, target, up=(0.0, 1.0, 0.0)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = x, y, z
    m[3, :3] = (-x @ eye, -y @ eye, -z @ eye)
    return m


def perspective_rh(fov_y, aspect, zn, zf):
    h = 1.0 / math.tan(fov_y / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = h / aspect, h
    m[2, 2], m[2, 3] = zf / (zn - zf), -1.0
    m[3, 2] = zn * zf / (zn - zf)
    return m


def perspective_lh(fov_y, aspect, zn, zf):
    h = 1.0 / math.tan(fov_y / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = h / aspect, h
    m[2, 2], m[2, 3] = zf / (zf - zn), 1.0
    m[3, 2] = -zn * zf / (zf - zn)
    return m


def ortho_lh(w, h, zn, zf):
    m = np.eye(4)
    m[0, 0], m[1, 1] = 2 / w, 2 / h
    m[2, 2] = 1 / (zf - zn)
    m[3, 2] = zn / (zn - zf)
    return m


def transform_points(points, m):
    p = np.hstack([points, np.ones((len(points), 1))]) @ m
    return p[:, :3] / p[:, 3:4]


class Mesh:
    """A drawable kn5 mesh with world-space data."""

    def __init__(self, node, world, material, path):
        self.node, self.world, self.material, self.path = node, world, material, path
        stride = node.vertex_size // 4
        v = np.frombuffer(node.vertices, np.float32).reshape(-1, stride)
        self.local = v
        self.indices = np.asarray(node.indices, np.uint32)
        self.visible = True

    def world_positions(self):
        return transform_points(self.local[:, 0:3].astype(np.float64), self.world)


class Car:
    def __init__(self, car_dir, skin=None):
        self.dir = car_dir
        self.data_dir = os.path.join(car_dir, "data")
        self.kn5_name = self._main_kn5()
        self.kn5 = kn5.read(os.path.join(car_dir, self.kn5_name))
        self.textures = {t.name.lower(): t.data for t in self.kn5.textures}
        self.skin_textures = {}
        self.csp_hidden = set()
        self.meshes = []
        self.dummies = {}
        self._blurred = self._blurred_objects()
        self._walk(self.kn5.root, np.eye(4), [], True)
        self._apply_visibility()
        self._skin_skinned_meshes()
        self._ground()
        for m in self.meshes:
            m.base_visible = m.visible
        if skin:
            self.set_skin(skin)

    # car data ----------------------------------------------------------------
    def _ini(self, name):
        cp = configparser.ConfigParser(interpolation=None, strict=False, inline_comment_prefixes=(";",))
        cp.optionxform = str
        path = os.path.join(self.data_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig", errors="replace") as f:
                cp.read_string(f.read())
        return cp

    def _main_kn5(self):
        lods = self._ini("lods.ini")
        if lods.has_section("LOD_0"):
            return lods["LOD_0"]["FILE"]
        names = sorted(f for f in os.listdir(self.dir) if f.endswith(".kn5") and f != "collider.kn5")
        return names[0]

    def set_skin(self, skin):
        folder = os.path.join(self.dir, "skins", skin)
        self.skin_textures = {}
        for f in os.listdir(folder):
            if f.lower().endswith((".dds", ".png", ".jpg", ".jpeg")):
                with open(os.path.join(folder, f), "rb") as fh:
                    self.skin_textures[f.lower()] = fh.read()
        hidden = csp_hidden_meshes(os.path.join(self.dir, "extension", "ext_config.ini"), self.kn5_name, skin)
        hidden |= csp_hidden_meshes(os.path.join(folder, "ext_config.ini"), self.kn5_name, skin)
        for m in getattr(self, "meshes", []):
            m.visible = m.base_visible and not any(p in hidden for p in m.path)
        self.csp_hidden = hidden

    def texture_bytes(self, name):
        key = name.lower()
        return self.skin_textures.get(key) or self.textures.get(key)

    # scene graph ---------------------------------------------------------------
    def _blurred_objects(self):
        """Blurred-rim switching at standstill: {node name: shown}; overrides the kn5 active flag."""
        result = {}
        blurred = self._ini("blurred_objects.ini")
        for sec in blurred.sections():
            s = blurred[sec]
            if s.get("NAME"):
                lo, hi = float(s.get("MIN_SPEED", 0)), float(s.get("MAX_SPEED", 0))
                result[s["NAME"]] = lo <= 0.0 < hi
        return result

    def _walk(self, node, parent_world, path, active):
        active = active and (self._blurred[node.name] if node.name in self._blurred else bool(node.active))
        if node.cls == 1:
            world = np.array(node.matrix, float).reshape(4, 4) @ parent_world
            self.dummies.setdefault(node.name, (world, len(self.meshes)))
            for c in node.children:
                self._walk(c, world, path + [node.name], active)
            self.dummies[node.name] = (self.dummies[node.name][0], self.dummies[node.name][1], len(self.meshes))
        else:
            material = self.kn5.materials[node.material]
            mesh = Mesh(node, parent_world, material, path + [node.name])
            mesh.visible = active and bool(node.renderable)
            self.meshes.append(mesh)

    def _dummy_meshes(self, name):
        entry = self.dummies.get(name)
        if not entry:
            return []
        _, start, end = entry
        return self.meshes[start:end]

    def _apply_visibility(self):
        # Preview defaults: low-res cockpit off, seatbelt "off" state, no driver.
        hidden = {"COCKPIT_LR", "STEER_LR", "CINTURE_ON"}
        for m in self.meshes:
            if any(p in hidden for p in m.path):
                m.visible = False
            if m.material.shader == "ksBrokenGlass":
                m.visible = False

    def _skin_skinned_meshes(self):
        # Skinned meshes are drawn in their bind pose: bake bones into world space.
        for m in self.meshes:
            if m.node.cls != 3:
                continue
            v = m.local
            pos = v[:, 0:3].astype(np.float64)
            out = np.zeros_like(pos)
            nrm = np.zeros_like(pos)
            tan = np.zeros_like(pos)
            for bi, (bone, inv_bind) in enumerate(m.node.bones):
                if bone not in self.dummies:
                    continue
                mat = np.array(inv_bind, float).reshape(4, 4) @ self.dummies[bone][0]
                w = np.zeros(len(v))
                for k in range(4):
                    w += np.where(v[:, 15 + k] == bi, v[:, 11 + k], 0.0)
                out += w[:, None] * transform_points(pos, mat)
                nrm += w[:, None] * (v[:, 3:6] @ mat[:3, :3])
                tan += w[:, None] * (v[:, 8:11] @ mat[:3, :3])
            baked = v.copy()
            baked[:, 0:3], baked[:, 3:6], baked[:, 8:11] = out, nrm, tan
            m.local = np.ascontiguousarray(baked[:, :11], np.float32)
            m.world = np.eye(4)

    def _bbox(self, meshes):
        pts = [m.world_positions() for m in meshes if getattr(m, "base_visible", m.visible) and len(m.local)]
        if not pts:
            return None
        pts = np.vstack(pts)
        return pts.min(axis=0), pts.max(axis=0)

    def _ground(self):
        """Level the car on its wheels and put the tyres on y = 0 (CM's AdjustPosition)."""
        boxes = {w: self._bbox(self._dummy_meshes(w)) for w in ("WHEEL_LF", "WHEEL_RF", "WHEEL_LR", "WHEEL_RR")}
        adjust = None
        if all(boxes.values()):
            y1 = min(boxes["WHEEL_LF"][0][1], boxes["WHEEL_RF"][0][1]) - 0.001
            y2 = min(boxes["WHEEL_LR"][0][1], boxes["WHEEL_RR"][0][1]) - 0.001
            x1 = (boxes["WHEEL_LF"][0][2] + boxes["WHEEL_LF"][1][2]) / 2
            x2 = -(boxes["WHEEL_LR"][0][2] + boxes["WHEEL_LR"][1][2]) / 2
            if x1 + x2 >= 0.01:
                y = y2 + x2 * (y1 - y2) / (x1 + x2)
                adjust = rotation_x(math.atan2(y1 - y2, x1 + x2)) @ translation(0, -y, 0)
        if adjust is None:
            box = self._bbox(self.meshes)
            adjust = translation(0, -box[0][1], 0)
        for m in self.meshes:
            m.world = m.world @ adjust
        self.dummies = {k: (w @ adjust, s, e) for k, (w, s, e) in self.dummies.items()}

    # ambient shadows ---------------------------------------------------------
    def ambient_shadows(self):
        """(texture name, world matrix) for the body and wheel shadow quads."""
        out = []
        s = self._ini("ambient_shadows.ini")
        width = float(s.get("SETTINGS", "WIDTH", fallback="1"))
        length = float(s.get("SETTINGS", "LENGTH", fallback="1"))
        out.append(("body_shadow.png", scaling(width, 1, length) @ rotation_y(math.pi)))
        for i, w in enumerate(("WHEEL_LF", "WHEEL_RF", "WHEEL_LR", "WHEEL_RR")):
            if w in self.dummies:
                t = self.dummies[w][0][3, :3]
                out.append((f"tyre_{i}_shadow.png",
                            scaling(0.3, 1, 0.3) @ rotation_y(math.pi) @ translation(t[0], 0.0, t[2])))
        return [(n, m) for n, m in out if os.path.exists(os.path.join(self.dir, n))]

    def bounding_box(self, exclude_seatbelt=False):
        meshes = [m for m in self.meshes if not (exclude_seatbelt and m.node.name.startswith("CINTURE_OFF"))]
        return self._bbox(meshes)


class Camera:
    def __init__(self, position, target, fov_y, aspect):
        self.position, self.target = np.asarray(position, float), np.asarray(target, float)
        self.fov_y, self.aspect = fov_y, aspect
        self.near = 0.04 / fov_y ** 2
        self.far = 1e3 / fov_y ** 0.5

    # Content Manager's cameras are right-handed (CameraBase.RhMode).
    @property
    def view(self):
        return look_at_rh(self.position, self.target)

    @property
    def proj(self):
        return perspective_rh(self.fov_y, self.aspect, self.near, self.far)

    @property
    def view_proj(self):
        return self.view @ self.proj

    def move(self, offset):
        self.position = self.position + offset
        self.target = self.target + offset


def preview_camera(car, width, height):
    """The preview camera, moved so it looks at the car's centre (CM's AlignCar)."""
    cam = Camera(CAMERA_POSITION, CAMERA_LOOK_AT, math.radians(CAMERA_FOV_DEG), width / height)
    lo, hi = car.bounding_box(exclude_seatbelt=True)
    center = (lo + hi) / 2
    center[1] = 0.0
    cam.move(center)
    return cam


def _read_ext_config(path):
    """Sections of a CSP ext_config.ini as (name, {key: value}) pairs, keeping repeated names."""
    sections = []
    if not os.path.exists(path):
        return sections
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.split(";", 1)[0].strip()
            if line.startswith("[") and line.endswith("]"):
                sections.append((line[1:-1].strip(), {}))
            elif "=" in line and sections:
                k, v = line.split("=", 1)
                sections[-1][1][k.strip().upper()] = v.strip()
    return sections


def _names(value):
    return {n.strip() for n in value.split(",") if n.strip()}


def csp_hidden_meshes(path, kn5_name, skin):
    """Meshes a CSP ext_config hides for this kn5 and skin (MODEL_REPLACEMENT / MESH_ADJUSTMENT)."""
    hidden = set()
    for name, keys in _read_ext_config(path):
        if keys.get("ACTIVE", "1") == "0":
            continue
        skins = _names(keys.get("SKINS", ""))
        if skins and skin not in skins and "?" not in skins and "*" not in skins:
            continue
        if name.startswith("MODEL_REPLACEMENT_"):
            if keys.get("FILE", "").lower() == kn5_name.lower():
                hidden |= _names(keys.get("HIDE", ""))
        elif name.startswith("MESH_ADJUSTMENT_") and keys.get("IS_RENDERABLE", "1") == "0":
            hidden |= _names(keys.get("MESHES", ""))
    return hidden
