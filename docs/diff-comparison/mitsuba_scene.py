"""The scene of `diff_point_light_gradients_match_finite_differences` and
`diff_area_light_gradients_match_finite_differences` of `tests/diff` in
Mitsuba 3, with the same meshes, camera, light, loss and gradient check.

Gradients come from Mitsuba's projective integrators, against central
differences of the same unbiased loss, sum((A - T) * (B - T)) / 3N, with
A and B rendered from independent seeds.

    python mitsuba_scene.py VARIANT INTEGRATOR ITERATIONS SPP FD_SPP LIGHT

for example `llvm_ad_rgb direct_projective 32 4 8192 area`, where LIGHT is
`point` or `area`. See `README.md` for the results.
"""
import json
import math
import os
import sys
import tempfile
import time

import drjit as dr
import mitsuba as mi

mi.set_variant(sys.argv[1] if len(sys.argv) > 1 else "llvm_ad_rgb")
INTEGRATOR = sys.argv[2] if len(sys.argv) > 2 else "direct_projective"
WIDTH, HEIGHT = 80, 64
FOV_Y = 0.8
CAMERA_Z = 0.5 * 6.0 / math.tan(0.5 * FOV_Y)


def uv_sphere(center, radius, segments=32, rings=16):
    """The sphere of `tests/pbr_scene.rs`, with the same vertices and faces."""
    positions, normals, faces = [], [], []
    for ring in range(rings + 1):
        theta = math.pi * ring / rings
        for segment in range(segments + 1):
            phi = 2.0 * math.pi * segment / segments
            n = [math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi)]
            positions.append([c + radius * v for c, v in zip(center, n)])
            normals.append(n)
    stride = segments + 1
    for ring in range(rings):
        for segment in range(segments):
            base = ring * stride + segment
            faces.append([base, base + 1, base + stride])
            faces.append([base + 1, base + stride + 1, base + stride])
    mesh = mi.Mesh("ball", len(positions), len(faces), has_vertex_normals=True)
    params = mi.traverse(mesh)
    params["vertex_positions"] = dr.ravel(mi.Point3f(*[list(c) for c in zip(*positions)]))
    params["vertex_normals"] = dr.ravel(mi.Normal3f(*[list(c) for c in zip(*normals)]))
    params["faces"] = dr.ravel(mi.Vector3u(*[list(c) for c in zip(*faces)]))
    params.update()
    return mesh


def principled(color, roughness):
    return {
        "type": "principled",
        "base_color": {"type": "rgb", "value": color},
        "roughness": roughness,
        "metallic": 0.0,
        "specular": 0.5,
    }


MESHES = tempfile.mkdtemp()
BALL_PATH = os.path.join(MESHES, "ball.ply")
uv_sphere([0.0, 4.2, -0.8], 0.75).write_ply(BALL_PATH)
# A point light, or a small sphere that emits about as much, which
# Mitsuba differentiates by moving its vertices.
LIGHT = sys.argv[6] if len(sys.argv) > 6 else "point"
LIGHT_RADIUS = 0.1
LIGHT_PATH = os.path.join(MESHES, "light.ply")
TARGET_LIGHT = [0.0, 8.0, -0.2]
uv_sphere(TARGET_LIGHT, LIGHT_RADIUS).write_ply(LIGHT_PATH)
# The intensity of a sphere of radiance L, towards anywhere.
RADIANCE_PER_INTENSITY = 1.0 / (math.pi * LIGHT_RADIUS**2)
scene = mi.load_dict(
    {
        "type": "scene",
        "integrator": {"type": INTEGRATOR, **({"max_depth": 2} if "prb" in INTEGRATOR else {})},
        "sensor": {
            "type": "perspective",
            "fov": math.degrees(FOV_Y),
            "fov_axis": "y",
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, 0, CAMERA_Z], target=[0, 0, 0], up=[0, 1, 0]
            ),
            "film": {
                "type": "hdrfilm",
                "width": WIDTH,
                "height": HEIGHT,
                "rfilter": {"type": "box"},
                "pixel_format": "rgb",
            },
            "sampler": {"type": "independent"},
        },
        "wall": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f()
            .translate([0.0, 0.0, -1.6])
            .scale([4.0, 3.0, 0.2]),
            "bsdf": principled([0.7, 0.7, 0.7], 0.5),
        },
        "ball": {
            "type": "ply",
            "filename": BALL_PATH,
            "bsdf": principled([0.95, 0.72, 0.35], 0.65),
        },
        "light": {
            "type": "point",
            "position": TARGET_LIGHT,
            "intensity": {"type": "rgb", "value": [800.0] * 3},
        }
        if LIGHT == "point"
        else {
            "type": "ply",
            "filename": LIGHT_PATH,
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": [800.0 * RADIANCE_PER_INTENSITY] * 3},
            },
        },
    }
)
params = mi.traverse(scene)
BASE = dr.unravel(mi.Point3f, mi.Float(params["ball.vertex_positions"]))
if LIGHT != "point":
    LIGHT_BASE = dr.unravel(mi.Point3f, mi.Float(params["light.vertex_positions"]))


def apply(offset, position, intensity):
    params["ball.vertex_positions"] = dr.ravel(BASE + offset)
    if LIGHT == "point":
        params["light.position"] = position
        params["light.intensity.value"] = intensity
    else:
        params["light.vertex_positions"] = dr.ravel(
            LIGHT_BASE + (position - mi.Point3f(*TARGET_LIGHT))
        )
        params["light.emitter.radiance.value"] = intensity * RADIANCE_PER_INTENSITY
    params.update()


def set_scene(offset, position, intensity):
    apply(mi.Vector3f(*offset), mi.Point3f(*position), mi.Color3f(*intensity))


def render(spp, seed):
    return mi.render(scene, params, spp=spp, seed=seed)


set_scene([0.0] * 3, [0.0, 8.0, -0.2], [800.0] * 3)
target = mi.TensorXf(render(64, 1_000_000))
count = WIDTH * HEIGHT * 3

START_BALL = [0.3, 0.25, 0.0]
START_LIGHT = [0.3, 7.7, 0.1]
START_INTENSITY = [700.0, 800.0, 900.0]


def unbiased_loss(offset, position, intensity, spp, seed):
    set_scene(offset, position, intensity)
    a = render(spp, seed)
    b = render(spp, seed + 1)
    return dr.sum(dr.ravel((a - target) * (b - target)))[0] / count


def gradient(iterations, spp, batch):
    """Per batch of `batch` iterations, the mean gradients of the loss."""
    results = []
    for start in range(0, iterations, batch):
        sums = {"ball": [0.0] * 3, "position": [0.0] * 3, "intensity": 0.0}
        for i in range(start, start + batch):
            offset = mi.Vector3f(*START_BALL)
            position = mi.Point3f(*START_LIGHT)
            intensity = mi.Color3f(*START_INTENSITY)
            dr.enable_grad(offset, position, intensity)
            apply(offset, position, intensity)
            # The primal and the adjoint of independent seeds, like blade.
            image = mi.render(scene, params, spp=spp, spp_grad=spp, seed=2 * i, seed_grad=2 * i + 1)
            loss = dr.sum(dr.ravel(dr.square(image - target))) / count
            dr.backward(loss)
            dr.eval()
            print(f"iteration {i}: {time.time() - START:.1f} s", flush=True)
            for axis, name in enumerate("xyz"):
                sums["ball"][axis] += dr.grad(getattr(offset, name))[0]
                sums["position"][axis] += dr.grad(getattr(position, name))[0]
            sums["intensity"] += dr.grad(intensity.x)[0]
        results.append(
            {
                "ball": [g / batch for g in sums["ball"]],
                "position": [g / batch for g in sums["position"]],
                "intensity": sums["intensity"] / batch,
            }
        )
    return results


def mean_and_error(values):
    n = len(values)
    mean = sum(values) / n
    spread = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(spread / n)


ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
SPP = int(sys.argv[4]) if len(sys.argv) > 4 else 16
FD_SPP = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
clock = time.time()
START = clock
batches = gradient(ITERATIONS, SPP, ITERATIONS // 8)
gradient_seconds = time.time() - clock

checks = [
    ("Ball(0)", 0.05, lambda b: b["ball"][0]),
    ("Ball(1)", 0.05, lambda b: b["ball"][1]),
    ("Position(0)", 0.1, lambda b: b["position"][0]),
    ("Position(1)", 0.1, lambda b: b["position"][1]),
    ("Position(2)", 0.1, lambda b: b["position"][2]),
    ("Intensity", 20.0, lambda b: b["intensity"]),
]
rows = []
clock = time.time()
for name, h, pick in checks:
    analytic, error = mean_and_error([pick(b) for b in batches])
    losses = []
    for sign in (1.0, -1.0):
        offset = list(START_BALL)
        position = list(START_LIGHT)
        intensity = list(START_INTENSITY)
        kind, axis = name.rstrip(")").split("(") if "(" in name else (name, None)
        if kind == "Ball":
            offset[int(axis)] += sign * h
        elif kind == "Position":
            position[int(axis)] += sign * h
        else:
            intensity[0] += sign * h
        losses.append(unbiased_loss(offset, position, intensity, FD_SPP // 2, 2_000_000))
    numeric = (losses[0] - losses[1]) / (2 * h)
    print(f"d loss / d {name}: {analytic:e} ± {error:e}, central difference {numeric:e}", flush=True)
    rows.append({"param": name, "analytic": analytic, "error": error, "numeric": numeric})
fd_seconds = time.time() - clock
print(json.dumps({
    "integrator": INTEGRATOR,
    "light": LIGHT,
    "iterations": ITERATIONS,
    "spp": SPP,
    "gradient_seconds": gradient_seconds,
    "fd_seconds": fd_seconds,
    "rows": rows,
}))
