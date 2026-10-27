"""OpenGL port of Content Manager's dark preview renderer (AcTools.Render DarkKn5ObjectRenderer).

Matrices are uploaded in their row-vector form; GLSL then sees the transpose, so
shaders multiply as `M * v`. Direct3D clip-space depth (0..1) is converted to
OpenGL's (-1..1) in the vertex shaders.
"""
import math
import os

import moderngl
import numpy as np

import preview_scene as ps
import textures

# The look approximates Content Manager's "Kunos" preview preset (warm key light, cool ambient,
# filmic tone mapping) without its showroom: a procedural studio stands in, with a spot-lit floor
# fading to black and soft studio lights for reflections. Values were fitted against previews
# rendered by Content Manager.
def _rgb(hex_color):
    return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


LOOK = {
    "light_az": 147.671, "light_el": 39.458, "light": 0.255,       # key light: azimuth/elevation (deg), brightness
    "amb_up": 2.884, "amb_down": 3.621,                       # ambient gradient brightness (sky / ground)
    "refl": 0.94,                                           # materials reflectiveness
    "env_top": 2.604, "env_top_edge": 0.799,                   # overhead softbox in reflections
    "env_horizon": 0.157, "env_horizon_width": 24.913,          # band of light at the horizon
    "env_wall": 0.033, "env_floor": 0.104,                     # studio walls / floor as seen in reflections
    "glass_refl": 1.573, "glass_wall": 0.029,                # transparent materials: reflection strength, wall light
    "windscreen_light": 20.0,                                # key light multiplier on ksWindscreen (inner glass)
    "floor_r": 0.09, "floor_g": 0.105, "floor_b": 0.117,      # floor colour under the car
    "floor_rx": 4.154, "floor_rz": 4.649, "floor_shadow": 0.041,  # floor spot size (m) and key light shadow strength
    "exposure": 1.633, "gamma": 0.79,                       # filmic tone mapping
    "shadow_op": 1.052, "bloom": 0.31,                         # ambient shadow opacity, bloom radius
}
LIGHT_HUE, AMBIENT_UP_HUE, AMBIENT_DOWN_HUE = _rgb("FFEACC"), _rgb("C6E6FF"), _rgb("515E6D")
BACKGROUND = np.zeros(3)
TONE_WHITE_POINT = 2.0
SHADOW_MAP_SIZE = 2048
PCSS_SCENE_SCALE, PCSS_LIGHT_SCALE = 0.06, 2.0


def light_direction():
    az, el = math.radians(LOOK["light_az"]), math.radians(LOOK["light_el"])
    return np.array([math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)])


# Material techniques (DarkMaterial.Kunos.fx).
STANDARD, ALPHA, REFLECTIVE, NM, NM_UV_MULT, AT_NM, MAPS, SKINNED_MAPS, TYRES, WINDSCREEN, GL, INVISIBLE = range(12)

# Material flags (DarkMaterial.Base.fx).
HAS_NORMAL_MAP, NM_OBJECT_SPACE, HAS_DETAILS_MAP = 1, 2, 4
IS_ADDITIVE, IS_CARPAINT, USE_NORMAL_ALPHA_AS_ALPHA, ALPHA_TEST = 16, 32, 64, 128

COMMON = """
#version 410
vec4 fixDepth(vec4 h) { h.z = 2.0 * h.z - h.w; return h; }
"""

CAR_VS = COMMON + """
uniform mat4 uWorld, uWorldIT, uWVP;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in vec3 in_tan;
out vec3 vPosW; out vec3 vNrmW; out vec3 vTanW; out vec2 vUv;
void main() {
    vec4 p = vec4(in_pos, 1.0);
    vPosW = (uWorld * p).xyz;
    vNrmW = (uWorldIT * vec4(in_nrm, 0.0)).xyz;
    vTanW = (uWorldIT * vec4(in_tan, 0.0)).xyz;
    vUv = in_uv;
    gl_Position = fixDepth(uWVP * p);
}
"""

SHADOWS = """
uniform mat4 uShadowVP;
uniform sampler2DShadow uShadowCmp;
uniform sampler2D uShadowRaw;
uniform sampler2D uNoise;
uniform float uPcssScene, uPcssLight, uShadowDx;
uniform int uUsePcss;

const vec2 blockerDisk[25] = vec2[](
    vec2(-0.9192313, 0.2457843), vec2(-0.5790563, 0.3955263), vec2(-0.6346537, -0.04642371),
    vec2(-0.9375812, -0.2063955), vec2(-0.3735094, -0.5372211), vec2(-0.1980452, -0.01467185),
    vec2(-0.6887965, -0.4922413), vec2(-0.1709323, 0.5356379), vec2(-0.3866155, 0.8176227),
    vec2(-0.7152032, 0.6910197), vec2(0.1532712, 0.7972272), vec2(0.2496267, 0.3971553),
    vec2(0.4972632, 0.709724), vec2(0.2775968, -0.1397319), vec2(0.04903403, -0.464276),
    vec2(-0.3010814, -0.8965761), vec2(0.5884293, -0.3083335), vec2(0.3185155, -0.9042132),
    vec2(0.6583639, 0.09571647), vec2(0.8057352, 0.5108864), vec2(0.9027404, -0.08043767),
    vec2(0.8957611, -0.4370965), vec2(-0.01861715, -0.7930358), vec2(0.5294089, -0.6455017),
    vec2(-0.01366961, 0.2491239));
const vec2 pcfDisk[48] = vec2[](
    vec2(0.855832, 0.1898994), vec2(0.6336077, 0.4661179), vec2(0.9748058, 0.0005619526),
    vec2(0.6545785, 0.0303187), vec2(0.8826399, 0.4462606), vec2(0.5820866, 0.2491302),
    vec2(0.7813198, -0.1931429), vec2(0.4344696, -0.463068), vec2(0.3466403, -0.2568325),
    vec2(0.7616104, -0.465308), vec2(0.3484367, -0.6709957), vec2(0.1399384, -0.1253062),
    vec2(0.06888496, -0.5598317), vec2(0.3422709, -0.007572813), vec2(0.1778579, -0.8323114),
    vec2(0.6398911, -0.7186768), vec2(-0.002360195, -0.3007), vec2(0.3574681, 0.2342279),
    vec2(0.3560048, 0.5762256), vec2(0.4195986, 0.8352308), vec2(0.05449425, 0.8639945),
    vec2(0.6891651, 0.6874872), vec2(0.1294608, 0.6387498), vec2(-0.2249427, -0.9602716),
    vec2(-0.1009111, -0.7166787), vec2(-0.1966588, -0.4243899), vec2(-0.3251926, -0.7385348),
    vec2(0.1280651, 0.1115537), vec2(-0.4393473, -0.4411973), vec2(-0.3216322, -0.1805354),
    vec2(-0.6028456, -0.2775587), vec2(-0.2305238, 0.8397814), vec2(-0.06095055, 0.5215759),
    vec2(-0.7246163, -0.6027799), vec2(0.130333, 0.4074706), vec2(-0.09729999, -0.02041483),
    vec2(-0.8133149, -0.04492685), vec2(-0.4851737, 0.03152225), vec2(-0.9180767, -0.2857039),
    vec2(-0.4836228, 0.6344832), vec2(-0.1267498, 0.2698318), vec2(0.006003978, -0.9894701),
    vec2(-0.3231294, 0.4236689), vec2(-0.7657595, 0.1987602), vec2(-0.5273226, 0.2657178),
    vec2(-0.5848147, -0.8110186), vec2(-0.8137172, 0.5681945), vec2(0.4109656, -0.9079512));

float pcss(vec2 uv, float z) {
    float searchWidth = uPcssScene * (z - 1.0) / z;
    float blockerSum = 0.0, blockers = 0.0;
    for (int i = 0; i < 25; i++) {
        float d = texture(uShadowRaw, uv + blockerDisk[i] * searchWidth).r;
        if (d < z) { blockerSum += d; blockers += 1.0; }
    }
    if (blockers < 1.0) return 1.0;
    float avg = blockerSum / blockers;
    float radius = min((z - avg) / avg, 0.05) * uPcssLight + uShadowDx;
    vec2 rnd = normalize(texture(uNoise, uv * 4000.0).xy);
    float sum = 0.0;
    for (int i = 0; i < 48; i++) {
        sum += texture(uShadowCmp, vec3(uv + reflect(pcfDisk[i], rnd) * radius, z));
    }
    return sum / 48.0;
}

float getShadow(vec3 pos) {
    vec4 uv = uShadowVP * vec4(pos, 1.0);
    uv.xyz /= uv.w;
    if (uv.x < 0.03 || uv.x > 0.97 || uv.y < 0.03 || uv.y > 0.97) return 1.0;
    // Shadow uv is Direct3D-style (v down); the map was rendered by OpenGL (v up).
    vec2 st = vec2(uv.x, 1.0 - uv.y);
    if (uUsePcss == 1) return pcss(st, uv.z);
    float shadow = 0.0;
    for (float y = -1.5; y <= 1.501; y += 1.0)
        for (float x = -1.5; x <= 1.501; x += 1.0)
            shadow += texture(uShadowCmp, vec3(st + vec2(x, y) * uShadowDx, uv.z));
    return clamp((shadow / 16.0 - 0.5) * 4.0 + 0.5, 0.0, 1.0);
}
"""

FRAME = """
uniform vec3 uEyePos, uLightDir, uLightColor, uAmbientDown, uAmbientRange, uBackground;
uniform float uReflectionPower;
float luminance(vec3 c) { return clamp(dot(c, vec3(0.299, 0.587, 0.114)), 0.0, 1.0); }
"""

CAR_FS = COMMON + SHADOWS + FRAME + """
uniform int uMode, uFlags, uAlphaOutput;
uniform mat4 uWorldITf;
uniform float uKa, uKd, uKs, uKsExp; uniform vec3 uEmissive;
uniform float uFresnelC, uFresnelExp, uFresnelMax;
uniform float uDetailsUvMult, uDetailsNormalBlend, uSunSpecular, uSunSpecularExp;
uniform float uAlpha, uDiffuseMult, uNormalMult, uBlurLevel, uWindscreenLight;
uniform sampler2D txDiffuse, txNormal, txMaps, txDetails, txDetailsNormal, txDiffuseBlur, txNormalBlur;
in vec3 vPosW; in vec3 vNrmW; in vec3 vTanW; in vec2 vUv;
out vec4 fragColor;

bool hasFlag(int f) { return (uFlags & f) == f; }
void alphaTest(float a) { if (hasFlag(128) && a < 0.5) discard; }

vec3 normalSample(vec3 s, vec3 nW, vec3 tW) {
    vec3 n = vec3(2.0 * s.x - 1.0, 1.0 - 2.0 * s.y, 2.0 * s.z - 1.0);
    if (hasFlag(2)) {
        n = vec3(n.x, n.z, n.y);
        return (uWorldITf * vec4(n, 0.0)).xyz;
    }
    vec3 N = normalize(nW);
    vec3 t = normalize(tW);
    vec3 T = normalize(t - dot(t, N) * N);
    vec3 B = normalize(cross(N, T));
    return n.x * T + n.y * B + n.z * N;
}

vec3 getAmbient(vec3 n) { return uAmbientDown + clamp(n.y * 0.5 + 0.5, 0.0, 1.0) * uAmbientRange; }

uniform float uEnvTop, uEnvTopEdge, uEnvHorizon, uEnvHorizonWidth, uEnvWall, uEnvFloor, uGlassRefl, uGlassWall;
vec3 getReflection(vec3 r, float specExp) {
    // Rougher materials see a blurrier studio: widen the light edges.
    float soft = mix(0.45, 0.06, clamp(specExp / 255.0, 0.0, 1.0));
    float top = smoothstep(uEnvTopEdge - soft, uEnvTopEdge + soft, r.y);
    float wall = smoothstep(-0.05 - soft, 0.05 + soft, r.y) * (1.0 - top);
    float horizon = exp(-abs(r.y) * uEnvHorizonWidth / (1.0 + soft * 8.0));
    float floorLight = r.y < 0.0 ? smoothstep(-0.6, 0.0, r.y) : 0.0;
    bool glass = uAlphaOutput == 1;
    return (uBackground + vec3(top * uEnvTop + wall * (glass ? uGlassWall : uEnvWall) + horizon * uEnvHorizon
        + floorLight * uEnvFloor)) * uReflectionPower * (glass ? uGlassRefl : 1.0);
}
float reflectionStrength(vec3 n, vec3 toEye) {
    float rid = 1.0 - clamp(dot(toEye, n), 0.0, 1.0);
    return min(pow(rid, uFresnelExp) + uFresnelC, uFresnelMax);
}
vec4 calcReflection(vec3 pos, vec3 n, float specExp) {
    vec3 toEye = normalize(uEyePos - pos);
    return vec4(getReflection(reflect(-toEye, n), specExp), reflectionStrength(n, toEye));
}

float specularLight(float nDotH, float e, float level) { return pow(nDotH, max(e, 0.1)) * level; }
float nDotH(vec3 n, vec3 pos) { return clamp(dot(normalize(normalize(uEyePos - pos) + uLightDir), n), 0.0, 1.0); }

void lightCustom(vec3 n, vec3 pos, float e, float level, out vec3 diffuse, out vec3 specular) {
    float shadow = getShadow(pos) * clamp(dot(n, uLightDir), 0.0, 1.0);
    diffuse = uLightColor * shadow;
    specular = specularLight(nDotH(n, pos), e, level) * uLightColor * shadow;
}
void lightSun(vec3 n, vec3 pos, float mult, out vec3 diffuse, out vec3 specular) {
    float shadow = getShadow(pos) * clamp(dot(n, uLightDir), 0.0, 1.0);
    diffuse = uLightColor * shadow;
    float h = nDotH(n, pos);
    specular = (specularLight(h, uKsExp * mult, uKs) + specularLight(h, uSunSpecularExp * mult, uSunSpecular))
        * uLightColor * shadow;
}
vec3 calcLight(vec3 c, vec3 n, vec3 pos) {
    vec3 d, s; lightCustom(n, pos, uKsExp, uKs, d, s);
    return c * (uKa * getAmbient(n) + uKd * d + uEmissive) + s;
}
vec3 calcLightMaps(vec3 c, vec3 n, vec3 pos, float specMult, float expMult) {
    vec3 d, s; lightCustom(n, pos, uKsExp * expMult, uKs, d, s);
    return c * (uKa * getAmbient(n) + uKd * d + uEmissive) + s * specMult;
}
vec3 calcLightMapsSun(vec3 c, vec3 n, vec3 pos, float specMult, float expMult) {
    vec3 d, s; lightSun(n, pos, expMult, d, s);
    return c * (uKa * getAmbient(n) + uKd * d + uEmissive) + s * specMult;
}

vec4 withReflection(vec3 lighted, float alpha, vec4 refl, float reflMult) {
    if (!hasFlag(16)) {
        lighted *= 1.0 - refl.a;
        alpha = alpha + refl.a * (1.0 - alpha);
    }
    return vec4(lighted + refl.rgb * refl.a * reflMult, alpha);
}

void main() {
    if (vPosW.y < 0.0) discard;  // flat mirror side test: nothing below the ground
    vec3 color, normal; float alpha; vec4 result;
    if (uMode == 0 || uMode == 1 || uMode == 2) {
        vec4 dv = texture(txDiffuse, vUv);
        color = dv.rgb; alpha = uMode == 1 ? dv.a * uAlpha : dv.a;
        normal = normalize(vNrmW);
        alphaTest(alpha);
        vec3 lighted = calcLight(color, normal, vPosW);
        result = uMode == 2 ? withReflection(lighted, alpha, calcReflection(vPosW, normal, uKsExp), 1.0)
                            : vec4(lighted, alpha);
    } else if (uMode == 3 || uMode == 4) {
        vec2 uvD = uMode == 4 ? vUv * (1.0 + uDiffuseMult) : vUv;
        vec2 uvN = uMode == 4 ? vUv * (1.0 + uNormalMult) : vUv;
        vec4 dv = texture(txDiffuse, uvD);
        vec4 nv = texture(txNormal, uvN);
        color = dv.rgb; alpha = uMode == 4 ? dv.a : nv.a;
        normal = normalize(normalSample(nv.xyz, vNrmW, vTanW));
        alphaTest(alpha);
        result = withReflection(calcLight(color, normal, vPosW), alpha, calcReflection(vPosW, normal, uKsExp), 1.0);
    } else if (uMode == 5) {
        vec4 dv = texture(txDiffuse, vUv);
        vec4 nv = texture(txNormal, vUv);
        color = dv.rgb; alpha = dv.a;
        normal = normalize(normalSample(nv.xyz, vNrmW, vTanW));
        alphaTest(alpha);
        result = vec4(calcLight(color, normal, vPosW), alpha);
    } else if (uMode == 6 || uMode == 7) {
        vec3 maps = texture(txMaps, vUv).rgb;
        vec4 dv = texture(txDiffuse, vUv);
        float mask = dv.a;
        if (hasFlag(4)) {
            vec4 det = texture(txDetails, vUv * uDetailsUvMult);
            color = dv.rgb * mix(det.rgb, vec3(1.0), mask);
            float x = mask, y = det.a, y1 = y - 1.0;
            maps.x *= x * x * y1 * y1 - x * y * y1 + y;
        } else {
            color = dv.rgb;
        }
        if (uMode == 7 || hasFlag(1)) {
            vec4 nv = texture(txNormal, vUv);
            alpha = uMode == 7 ? nv.a * mask : (hasFlag(64) ? nv.a : 1.0);
            if (uDetailsNormalBlend > 0.0) {
                vec4 dn = texture(txDetailsNormal, vUv * uDetailsUvMult);
                nv += (dn - 0.5) * uDetailsNormalBlend * (1.0 - mask);
            }
            normal = normalize(normalSample(nv.xyz, vNrmW, vTanW));
        } else {
            normal = normalize(vNrmW);
            alpha = 1.0;
        }
        alphaTest(alpha);
        float exp = (uKsExp + 400.0 * float(hasFlag(32))) * maps.y;
        vec4 refl = calcReflection(vPosW, normal, exp);
        result = withReflection(calcLightMapsSun(color, normal, vPosW, maps.x, maps.y), alpha, refl, maps.z);
    } else if (uMode == 8) {
        vec4 dv = mix(texture(txDiffuse, vUv), texture(txDiffuseBlur, vUv), uBlurLevel);
        vec4 nv = mix(texture(txNormal, vUv), texture(txNormalBlur, vUv), uBlurLevel);
        color = dv.rgb; alpha = dv.a;
        normal = normalize(normalSample(nv.xyz, vNrmW, vTanW));
        alphaTest(dv.a);
        vec4 refl = calcReflection(vPosW, normal, uKsExp);
        vec3 lighted = calcLightMaps(color, normal, vPosW, alpha, 1.0);
        if (!hasFlag(16)) lighted *= 1.0 - refl.a;
        result = vec4(lighted + refl.rgb * refl.a, 1.0);
    } else if (uMode == 9) {
        vec4 tx = texture(txDiffuse, vUv);
        alpha = tx.a;
        alphaTest(alpha);
        normal = normalize(vNrmW);
        vec3 fromEye = normalize(vPosW - uEyePos);
        vec3 diffuse = uLightColor * uWindscreenLight * getShadow(vPosW) * clamp(dot(fromEye, uLightDir), 0.0, 1.0);
        vec3 lighted = tx.rgb * (uKa * getAmbient(normal) + uKd * diffuse + uEmissive);
        result = vec4(lighted, alpha > 0.5 ? alpha : alpha * clamp(luminance(diffuse) * 0.76, 0.0, 1.0));
    } else {
        result = vec4(normalize(vNrmW), 1.0);
    }
    fragColor = vec4(result.rgb, uAlphaOutput == 1 ? result.a : 1.0);
}
"""

PT_VS = COMMON + """
uniform mat4 uWorld, uWVP;
in vec3 in_pos; in vec2 in_uv;
out vec3 vPosW; out vec2 vUv;
void main() {
    vPosW = (uWorld * vec4(in_pos, 1.0)).xyz;
    vUv = in_uv;
    gl_Position = fixDepth(uWVP * vec4(in_pos, 1.0));
}
"""

GROUND_FS = COMMON + SHADOWS + FRAME + """
uniform vec3 uCarCenter, uFloorColor;
uniform vec2 uFloorRadius;
uniform float uFloorShadow;
in vec3 vPosW; in vec2 vUv;
out vec4 fragColor;
void main() {
    // A soft spot of light on the floor around the car, fading to the black background.
    vec2 d = (vPosW.xz - uCarCenter.xz) / uFloorRadius;
    float spot = exp(-dot(d, d) * 1.6);
    float shadow = 1.0 - uFloorShadow * (1.0 - getShadow(vPosW));
    fragColor = vec4(uBackground + uFloorColor * spot * shadow, 1.0);
}
"""

AMBIENT_SHADOW_FS = COMMON + SHADOWS + FRAME + """
uniform sampler2D txDiffuse;
uniform float uOpacity;
in vec3 vPosW; in vec2 vUv;
out vec4 fragColor;
void main() {
    float value = texture(txDiffuse, vUv).r * uOpacity;
    float lightBrightness = clamp(luminance(uLightColor) * 1.5, 0.0, 1.0);
    fragColor = vec4(0.0, 0.0, 0.0, value * (1.0 - getShadow(vPosW) * lightBrightness * 0.95));
}
"""

DEPTH_VS = COMMON + """
uniform mat4 uWVP;
in vec3 in_pos;
void main() { gl_Position = fixDepth(uWVP * vec4(in_pos, 1.0)); }
"""

DEPTH_FS = COMMON + "void main() {}\n"


def _f32(m):
    return np.ascontiguousarray(m, np.float32).tobytes()


def material_setup(material):
    """Technique, flags and blending for a kn5 material (MaterialsProviderDark)."""
    shader = material.shader
    props = {p.name: p for p in material.props}
    mappings = {name: tex for name, _slot, tex in material.mappings}

    def a(name):
        return props[name].a if name in props else 0.0

    if shader == "ksBrokenGlass":
        mode = INVISIBLE
    elif shader == "GL":
        mode = GL
    elif shader in ("ksTyres", "ksBrakeDisc"):
        mode = TYRES
    elif shader == "ksWindscreen":
        mode = WINDSCREEN
    elif shader in ("ksPerPixel", "ksPerPixelAT", "ksPerPixelAT_NS", "ksTree"):
        mode = STANDARD
    elif shader == "ksPerPixelAT_NM":
        mode = AT_NM
    elif shader in ("ksPerPixelReflection", "ksPerPixelSimpleRefl"):
        mode = REFLECTIVE
    elif shader in ("ksPerPixelNM", "ksPerPixelNM_UV2"):
        mode = NM
    elif shader == "ksPerPixelNM_UVMult":
        mode = NM_UV_MULT
    elif shader.startswith("ksPerPixelMultiMap"):
        mode = MAPS
    elif shader == "ksPerPixelAlpha":
        mode = ALPHA
    elif shader in ("ksSkinnedMesh", "ksSkinnedMesh_NMDetaill"):
        mode = SKINNED_MAPS
    elif shader in ("ksSky", "ksSkyBox"):
        mode = INVISIBLE
    elif "skinned" in shader.lower():
        mode = GL
    elif "txMaps" in mappings:
        mode = MAPS
    elif "txNormal" in mappings:
        mode = NM
    else:
        mode = STANDARD

    flags = ALPHA_TEST if material.alpha_tested else 0
    if mode in (REFLECTIVE, NM, NM_UV_MULT, TYRES) and a("isAdditive") != 0.0:
        flags |= IS_ADDITIVE
    if mode in (NM, NM_UV_MULT, AT_NM) and a("nmObjectSpace") != 0:
        flags |= NM_OBJECT_SPACE
    if mode in (MAPS, SKINNED_MAPS):
        additive = a("isAdditive")
        if additive not in (0.0, 2.0):
            flags |= IS_ADDITIVE
        if "damage" not in shader and "txNormal" in mappings:
            flags |= HAS_NORMAL_MAP
            if a("nmObjectSpace") != 0:
                flags |= NM_OBJECT_SPACE
        if additive == 2.0:
            flags |= IS_CARPAINT
        if a("useDetail") > 0:
            flags |= HAS_DETAILS_MAP
        if "_AT" in shader or shader == "ksSkinnedMesh":
            flags |= USE_NORMAL_ALPHA_AS_ALPHA

    blending = material.blend == 1
    uniforms = {
        "uKa": a("ksAmbient"), "uKd": a("ksDiffuse"), "uKs": a("ksSpecular"), "uKsExp": a("ksSpecularEXP"),
        "uEmissive": tuple(props["ksEmissive"].c) if "ksEmissive" in props else (0.0, 0.0, 0.0),
        "uFresnelC": a("fresnelC"), "uFresnelExp": max(a("fresnelEXP"), 0.0001), "uFresnelMax": a("fresnelMaxLevel"),
        "uDetailsUvMult": a("detailUVMultiplier"),
        "uDetailsNormalBlend": a("detailNormalBlend") if "txNormalDetail" in mappings else 0.0,
        "uSunSpecular": a("sunSpecular"), "uSunSpecularExp": a("sunSpecularEXP"),
        "uAlpha": a("alpha"), "uDiffuseMult": a("diffuseMult"), "uNormalMult": a("normalMult"), "uBlurLevel": 0.0,
        "uMode": mode, "uFlags": flags,
        "uAlphaOutput": 1 if (blending or mode == ALPHA) and mode not in (TYRES, GL) else 0,
    }
    textures_needed = {"txDiffuse": "txDiffuse", "txNormal": "txNormal", "txMaps": "txMaps",
                       "txDetails": "txDetail", "txDetailsNormal": "txNormalDetail",
                       "txDiffuseBlur": "txBlur", "txNormalBlur": "txNormalBlur"}
    tex = {unit: mappings.get(kn5_name) for unit, kn5_name in textures_needed.items()}
    # Missing normal maps fall back to a flat one where Content Manager does so.
    flat_nm = mode in (NM, NM_UV_MULT) or (mode in (MAPS, SKINNED_MAPS) and flags & HAS_NORMAL_MAP)
    return mode, blending, uniforms, tex, flat_nm


class Renderer:
    def __init__(self, width=ps.PREVIEW_WIDTH * ps.SSAA, height=ps.PREVIEW_HEIGHT * ps.SSAA):
        self.width, self.height = width, height
        try:
            self.ctx = moderngl.create_standalone_context(require=410)
        except Exception:
            # Headless Linux (CI): EGL, loading the runtime sonames so no -dev packages are needed.
            self.ctx = moderngl.create_standalone_context(require=410, backend="egl",
                                                          libgl="libGL.so.1", libegl="libEGL.so.1")
        ctx = self.ctx
        self.car_prog = ctx.program(vertex_shader=CAR_VS, fragment_shader=CAR_FS)
        self.ground_prog = ctx.program(vertex_shader=PT_VS, fragment_shader=GROUND_FS)
        self.shadow_prog = ctx.program(vertex_shader=PT_VS, fragment_shader=AMBIENT_SHADOW_FS)
        self.depth_prog = ctx.program(vertex_shader=DEPTH_VS, fragment_shader=DEPTH_FS)

        self.shadow_depth = ctx.depth_texture((SHADOW_MAP_SIZE, SHADOW_MAP_SIZE))
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_depth)
        self.shadow_cmp = ctx.sampler(texture=self.shadow_depth, compare_func="<", filter=(moderngl.LINEAR, moderngl.LINEAR),
                                      repeat_x=False, repeat_y=False)
        self.shadow_raw = ctx.sampler(texture=self.shadow_depth, compare_func="", filter=(moderngl.NEAREST, moderngl.NEAREST),
                                      repeat_x=False, repeat_y=False)
        rng = np.random.default_rng(0)
        self.noise = ctx.texture((16, 16), 4, rng.integers(0, 256, (16, 16, 4), dtype=np.uint8).tobytes())
        self.noise.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.color = ctx.texture((width, height), 4, dtype="f2")
        self.depth = ctx.depth_renderbuffer((width, height))
        self.fbo = ctx.framebuffer(color_attachments=[self.color], depth_attachment=self.depth)

        self.null_tex = self._make_texture(np.zeros((1, 1, 4), np.uint8))
        self.flat_nm = self._make_texture(np.array([[[128, 128, 255, 255]]], np.uint8))
        self._tex_cache = {}
        self._vaos, self._buffers = {}, {}
        self.ssaa = max(1, round(width / ps.PREVIEW_WIDTH))
        self.quad = ctx.buffer(np.array([
            [1, 0, -1, 0, 0], [1, 0, 1, 0, 1], [-1, 0, -1, 1, 0],
            [-1, 0, 1, 1, 1], [-1, 0, -1, 1, 0], [1, 0, 1, 0, 1]], np.float32).tobytes())

    # resources -----------------------------------------------------------------
    def _make_texture(self, rgba):
        h, w = rgba.shape[:2]
        tex = self.ctx.texture((w, h), 4, np.ascontiguousarray(rgba).tobytes())
        tex.build_mipmaps()
        tex.anisotropy = 16.0
        tex.repeat_x = tex.repeat_y = True
        return tex

    def texture(self, car, name):
        if not name:
            return None
        data = car.texture_bytes(name)
        if not data:
            return None
        key = (name.lower(), hash(data))
        if key not in self._tex_cache:
            try:
                self._tex_cache[key] = self._make_texture(textures.decode(data))
            except Exception:
                self._tex_cache[key] = None
        return self._tex_cache[key]

    def _ambient_shadow_texture(self, path):
        # Kept for the renderer's lifetime: releasing textures between renders makes macOS's driver
        # log "texture unloadable" on the next frame.
        key = ("ambient", os.path.abspath(path))
        if key not in self._tex_cache:
            with open(path, "rb") as f:
                tex = self._make_texture(textures.decode(f.read()))
            tex.repeat_x = tex.repeat_y = False
            self._tex_cache[key] = tex
        return self._tex_cache[key]

    def _mesh_vao(self, mesh, prog, cols):
        key = (id(mesh), id(prog))
        vao = self._vaos.get(key)
        if vao is None:
            buffers = self._buffers.get(id(mesh))
            if buffers is None:
                verts = np.ascontiguousarray(mesh.local[:, :11], np.float32)
                buffers = self._buffers[id(mesh)] = (self.ctx.buffer(verts.tobytes()),
                                                     self.ctx.buffer(mesh.indices.tobytes()), mesh)
            fmt, names = cols
            vao = self._vaos[key] = self.ctx.vertex_array(prog, [(buffers[0], fmt, *names)],
                                                          index_buffer=buffers[1], index_element_size=4)
        return vao

    # frame ---------------------------------------------------------------------
    def _set_frame(self, prog, camera, shadow_vp, size):
        values = {
            "uEyePos": tuple(camera.position), "uLightDir": tuple(light_direction()),
            "uLightColor": tuple(LIGHT_HUE * LOOK["light"]),
            "uAmbientDown": tuple(AMBIENT_DOWN_HUE * LOOK["amb_down"]),
            "uAmbientRange": tuple(AMBIENT_UP_HUE * LOOK["amb_up"] - AMBIENT_DOWN_HUE * LOOK["amb_down"]),
            "uBackground": tuple(BACKGROUND), "uReflectionPower": LOOK["refl"],
            "uPcssScene": PCSS_SCENE_SCALE / size, "uPcssLight": PCSS_LIGHT_SCALE / size,
            "uShadowDx": 1.0 / SHADOW_MAP_SIZE, "uUsePcss": 0,
            "uEnvTop": LOOK["env_top"], "uEnvTopEdge": LOOK["env_top_edge"], "uEnvHorizon": LOOK["env_horizon"],
            "uEnvHorizonWidth": LOOK["env_horizon_width"], "uEnvWall": LOOK["env_wall"], "uEnvFloor": LOOK["env_floor"],
            "uFloorColor": (LOOK["floor_r"], LOOK["floor_g"], LOOK["floor_b"]),
            "uFloorRadius": (LOOK["floor_rx"], LOOK["floor_rz"]), "uFloorShadow": LOOK["floor_shadow"],
            "uGlassRefl": LOOK["glass_refl"], "uGlassWall": LOOK["glass_wall"],
            "uWindscreenLight": LOOK["windscreen_light"],
        }
        for k, v in values.items():
            if k in prog:
                prog[k].value = v
        if "uShadowVP" in prog:
            prog["uShadowVP"].write(_f32(shadow_vp))
        for name, unit in (("uShadowCmp", 13), ("uShadowRaw", 14), ("uNoise", 15)):
            if name in prog:
                prog[name].value = unit

    def render(self, car, camera):
        """Render one preview; returns a (height, width, 3) uint8 RGB image at preview size."""
        ctx = self.ctx
        light = light_direction()

        # Shadow map: one orthographic cascade around the car (ShadowsDirectional).
        lo, hi = car.bounding_box()
        center = (lo + hi) / 2
        size = min(float(np.linalg.norm(hi - lo)), 10.0)
        shadow_cam = ps.look_at_lh(center + 100.0 * light, center)
        shadow_proj = ps.ortho_lh(size, size, 1.0, 200.0)
        tex_matrix = np.array([[0.5, 0, 0, 0], [0, -0.5, 0, 0], [0, 0, 1, 0], [0.5, 0.5, 0, 1]])
        shadow_vp = shadow_cam @ shadow_proj @ tex_matrix

        meshes = [m for m in car.meshes if m.visible and len(m.indices)]
        setups = {id(m.material): material_setup(m.material) for m in meshes}

        self.shadow_fbo.use()
        self.shadow_fbo.clear(depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.cull_face = "front"
        ctx.polygon_offset = (1.0, 100.0)
        for m in meshes:
            if not m.node.cast_shadows or setups[id(m.material)][0] == INVISIBLE:
                continue
            self.depth_prog["uWVP"].write(_f32(m.world @ shadow_cam @ shadow_proj))
            self._mesh_vao(m, self.depth_prog, ("3f 32x", ["in_pos"])).render()
        ctx.polygon_offset = (0.0, 0.0)
        ctx.cull_face = "back"

        # Main pass.
        self.fbo.use()
        self.fbo.clear(*BACKGROUND, 0.0, depth=1.0)
        self.shadow_cmp.use(13)
        self.shadow_raw.use(14)
        self.noise.use(15)
        vp = camera.view_proj

        # Ground: the "flat background ground" (no mirror, black background).
        ctx.disable(moderngl.CULL_FACE)
        ground = ps.scaling(200, 200, 200)
        self._set_frame(self.ground_prog, camera, shadow_vp, size)
        self.ground_prog["uCarCenter"].value = tuple(center)
        self.ground_prog["uWorld"].write(_f32(ground))
        self.ground_prog["uWVP"].write(_f32(ground @ vp))
        ctx.vertex_array(self.ground_prog, [(self.quad, "3f 2f", "in_pos", "in_uv")]).render()

        # Ambient shadows (body and wheels), blended over the ground.
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.depth_mask = False
        ctx.polygon_offset = (-1.0, -1000.0)
        self._set_frame(self.shadow_prog, camera, shadow_vp, size)
        self.shadow_prog["uOpacity"].value = LOOK["shadow_op"]
        self.shadow_prog["txDiffuse"].value = 0
        for name, matrix in car.ambient_shadows():
            self._ambient_shadow_texture(os.path.join(car.dir, name)).use(0)
            self.shadow_prog["uWorld"].write(_f32(matrix))
            self.shadow_prog["uWVP"].write(_f32(matrix @ vp))
            ctx.vertex_array(self.shadow_prog, [(self.quad, "3f 2f", "in_pos", "in_uv")]).render()
        ctx.polygon_offset = (0.0, 0.0)
        ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)

        # Car: opaque, then transparent (read-only depth), in node order.
        ctx.enable(moderngl.CULL_FACE)
        ctx.depth_func = "<="
        prog = self.car_prog
        self._set_frame(prog, camera, shadow_vp, size)
        for unit, name in enumerate(("txDiffuse", "txNormal", "txMaps", "txDetails", "txDetailsNormal",
                                     "txDiffuseBlur", "txNormalBlur")):
            if name in prog:
                prog[name].value = unit
        for transparent in (False, True):
            if transparent:
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            for m in meshes:
                mode, blending, uniforms, tex, flat_nm = setups[id(m.material)]
                if mode == INVISIBLE or blending != transparent:
                    continue
                depth_mode = m.material.depth_mode
                ctx.depth_mask = not transparent and depth_mode == 0
                if depth_mode == 2:
                    ctx.disable(moderngl.DEPTH_TEST)
                for k, v in uniforms.items():
                    if k in prog:
                        prog[k].value = v
                world_it = np.linalg.inv(m.world.T)
                prog["uWorld"].write(_f32(m.world))
                prog["uWorldIT"].write(_f32(world_it))
                if "uWorldITf" in prog:
                    prog["uWorldITf"].write(_f32(world_it))
                prog["uWVP"].write(_f32(m.world @ vp))
                for unit, (name, kn5_name) in enumerate(tex.items()):
                    t = self.texture(car, kn5_name)
                    if t is None:
                        t = self.flat_nm if (name == "txNormal" and flat_nm) else self.null_tex
                    t.use(unit)
                self._mesh_vao(m, prog, ("3f 3f 2f 3f", ["in_pos", "in_nrm", "in_uv", "in_tan"])).render()
                ctx.enable(moderngl.DEPTH_TEST)
            ctx.depth_mask = True
            ctx.disable(moderngl.BLEND)
        ctx.depth_func = "<"

        hdr = np.frombuffer(self.color.read(), np.float16).reshape(self.height, self.width, 4)[::-1, :, :3]
        return finish(hdr.astype(np.float32), self.ssaa)


def _gaussian_blur(img, sigma):
    radius = int(math.ceil(sigma * 3))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-x * x / (2 * sigma * sigma))
    k /= k.sum()
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (radius, radius)
        p = np.pad(img, pad, mode="edge")
        out = np.zeros_like(img)
        for i, w in enumerate(k):
            sl = [slice(None)] * img.ndim
            sl[axis] = slice(i, i + img.shape[axis])
            out += w * p[tuple(sl)]
        img = out
    return img


def _filmic_curve(x):
    return ((x * (0.22 * x + 0.1 * 0.3) + 0.2 * 0.01) / (x * (0.22 * x + 0.3) + 0.2 * 0.3)) - 0.01 / 0.3


def tone_map(x):
    """Content Manager's "Filmic" tone mapping (PpHdr.fx)."""
    v = _filmic_curve(np.maximum(1.17504 * LOOK["exposure"] * x, 0.0)) / _filmic_curve(TONE_WHITE_POINT)
    return np.power(np.maximum(v, 0.0), 1.0 / LOOK["gamma"])


def finish(hdr, ssaa=ps.SSAA):
    """Bloom, tone mapping, downsample by the SSAA factor, 8-bit."""
    h, w = hdr.shape[:2]
    bright = np.clip(hdr - 1.5, 0.0, 1.0)
    half = bright[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    bloom = _gaussian_blur(half, 0.5 * ssaa * LOOK["bloom"] * 2.0)
    bloom = np.repeat(np.repeat(bloom, 2, axis=0), 2, axis=1)
    img = hdr.copy()
    img[: bloom.shape[0], : bloom.shape[1]] += bloom
    img = tone_map(img)
    s = ssaa
    small = np.clip(img, 0.0, 1.0)[: h // s * s, : w // s * s].reshape(h // s, s, w // s, s, 3).mean(axis=(1, 3))
    return np.clip(np.rint(small * 255.0), 0, 255).astype(np.uint8)
