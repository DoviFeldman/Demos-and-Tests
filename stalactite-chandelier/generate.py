#!/usr/bin/env python3
"""
Stalactite chandelier generator.

Sculpts a cluster of stalactites as a signed-distance field, hollows it into a
thin translucent shell (wall thickness varies with the surface detail, so the
flutes and drip rings show up as light and dark bands when lit from inside),
and writes:

  stalactite_chandelier.stl   print-ready, tips UP, flat rim on the bed (mm)
  preview.ply                 lighter mesh for the web viewer, hanging
                              orientation, vertex colours = simulated glow

Usage:  python3 generate.py [--voxel 0.5] [--seed 7] [--scale 1.0]

Needs: numpy scipy scikit-image scikit-fmm fast-simplification trimesh
"""
import argparse
import os
import struct

import numpy as np
import skfmm
import trimesh
import fast_simplification
from scipy import ndimage
from skimage import measure

ap = argparse.ArgumentParser()
ap.add_argument("--voxel", type=float, default=0.5, help="grid size in mm")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--scale", type=float, default=1.0, help="uniform scale of the export")
ap.add_argument("--stl-faces", type=int, default=450_000)
ap.add_argument("--ply-faces", type=int, default=220_000)
ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
args = ap.parse_args()

# ---------------------------------------------------------------- parameters
VOX = args.voxel
WALL = 1.8        # nominal shell thickness, measured from the smooth form (mm)
MIN_WALL = 1.2    # the shell never gets thinner than this, even in creases
BLEND = 5.0       # how softly neighbouring stalactites melt into each other
FLANGE_R = 95.0   # radius of the "cave ceiling" rim
FLANGE_T = 3.0    # thickness of that rim
HALF = 102.0      # half-width of the modelling volume
H_MAX = 238.0     # depth of the modelling volume
BULB_H = 70.0     # where the lamp sits (only used for the glow preview)

rng = np.random.default_rng(args.seed)
F32 = np.float32

# Modelling frame: h = depth below the ceiling. h is also the print Z axis,
# i.e. the lamp is printed upside down with the tips pointing up.
xs = (np.arange(-HALF, HALF + VOX / 2, VOX)).astype(F32)
hs = (np.arange(-2.5 * VOX, H_MAX, VOX)).astype(F32)   # h=0 falls between samples
NX, NH = len(xs), len(hs)
print(f"grid {NX} x {NX} x {NH}  ({NX * NX * NH / 1e6:.0f} M voxels)")

detail = np.full((NX, NX, NH), 20.0, F32)   # full surface with all the texture
base = np.full((NX, NX, NH), 20.0, F32)     # same forms without texture


def smin(a, b, k):
    """Polynomial smooth minimum: a soft union of two distance fields."""
    t = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - t) + a * t - k * t * (1 - t)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def add_stalactite(cx, cy, R0, L, power=(1.5, 0.78), flare=0.45, flutes=None,
                   ring_gap=None, bend=None):
    """Blend one stalactite into the fields.

    R0     radius just below the ceiling flare
    L      length
    power  (a, b) of the carrot profile (1 - t^a)^b
    flare  how much it widens where it meets the ceiling (fraction of R0)
    """
    a, b = power
    tip_r = min(1.8, 0.45 * R0)                 # rounded tip
    n_fl = flutes if flutes is not None else int(np.clip(round(R0 / 4.0), 3, 9))
    amp_fl = min(0.13 * R0, 3.2)                # depth of the drapery flutes
    amp_rg = min(0.10 * R0, 1.3)                # height of the drip rings
    lam = ring_gap if ring_gap is not None else float(np.clip(R0 * 0.5, 5.0, 14.0))
    bend = bend if bend is not None else min(0.10 * R0, 3.0)
    ph = rng.uniform(0, 2 * np.pi, 10)
    twist = rng.uniform(-1.6, 1.6)
    hf = 0.10 * L + 4.0                         # flare decay length

    rmax = R0 * (1 + flare) + amp_fl + amp_rg + bend + BLEND + 3.0
    i0, i1 = np.searchsorted(xs, [cx - rmax, cx + rmax])
    j0, j1 = np.searchsorted(xs, [cy - rmax, cy + rmax])
    k1 = np.searchsorted(hs, L + BLEND + 3.0)
    X = xs[i0:i1, None, None]
    Y = xs[None, j0:j1, None]
    H = hs[None, None, :k1]

    t = np.clip(H / L, 0.0, 1.0)
    # the axis wanders a little so nothing is perfectly straight
    ax = cx + bend * np.sin(np.pi * 1.3 * t + ph[0]) * t
    ay = cy + bend * np.sin(np.pi * 1.1 * t + ph[1]) * t
    dx, dy = X - ax, Y - ay
    rho = np.sqrt(dx * dx + dy * dy)
    th = np.arctan2(dy, dx)

    prof = (1 - t ** a) ** b
    r_base = (R0 - tip_r) * prof + flare * R0 * np.exp(-np.maximum(H, 0) / hf)
    past_tip = np.sqrt(rho * rho + (H - L) ** 2)

    # --- drapery flutes: rounded lobes separated by sharp creases
    lobes = np.abs(np.sin(0.5 * (n_fl * th + twist * t * 3.0
                                 + 0.9 * np.sin(2.6 * np.pi * t + ph[2]))))
    r_fl = amp_fl * (lobes - 0.637) * (1 - 0.55 * t) * np.clip(r_base / (4 * amp_fl), 0, 1)

    # --- drip rings: each one swells downward then tucks back in sharply
    u = (H / lam + 0.6 * np.sin(H * 0.045 + ph[3]) + 0.35 * np.sin(H * 0.11 + ph[4])
         + 0.10 * np.sin(th + ph[5]) + 0.05 * np.sin(2 * th + ph[6]))
    s = u - np.floor(u)
    ring = s ** 1.5 * (1 - smoothstep(0.80, 1.0, s)) - 0.33
    # rings come and go: strong in some zones, nearly smooth in others
    patch = (0.30 + 0.70 * smoothstep(-0.5, 0.7, np.sin(H * 0.05 + ph[7]) + 0.6 * np.sin(H * 0.021 + ph[8]))) \
        * (0.65 + 0.35 * np.sin(2 * th + H * 0.03 + ph[9]))
    r_rg = amp_rg * ring * patch * np.clip(r_base / (4 * amp_rg), 0, 1) * smoothstep(2.0, 10.0, H)

    d_base = np.where(H <= L, rho - r_base, past_tip) - tip_r
    d_det = np.where(H <= L, rho - (r_base + r_fl + r_rg), past_tip) - tip_r

    sl = (slice(i0, i1), slice(j0, j1), slice(0, k1))
    base[sl] = smin(base[sl], d_base.astype(F32), BLEND)
    detail[sl] = smin(detail[sl], d_det.astype(F32), BLEND)


# ---------------------------------------------------------------- composition
def ring(n, radius, r_jit, R0, L, ang0=0.0, ang_jit=0.25, **kw):
    for i in range(n):
        ang = ang0 + (i + rng.uniform(-ang_jit, ang_jit)) * 2 * np.pi / n
        rad = radius + rng.uniform(-r_jit, r_jit)
        add_stalactite(rad * np.cos(ang), rad * np.sin(ang),
                       rng.uniform(*R0), rng.uniform(*L), **kw)


# the great central stalactite - the lamp hangs inside its upper half
add_stalactite(0.0, 0.0, 46.0, 228.0, power=(1.6, 0.80), flare=0.35, flutes=9, bend=3.0)
# inner ring of long companions
ring(6, 41.0, 4.0, (23.0, 29.0), (125.0, 190.0), ang0=0.3)
# middle ring
ring(10, 64.0, 4.0, (14.0, 19.0), (65.0, 135.0), ang0=0.1)
# outer fringe of short ones
ring(15, 79.0, 3.0, (7.5, 10.5), (28.0, 75.0), ang0=0.5)
# soda straws and side growths scattered through the cluster
for _ in range(22):
    ang, rad = rng.uniform(0, 2 * np.pi), 88.0 * np.sqrt(rng.uniform(0.04, 1.0))
    add_stalactite(rad * np.cos(ang), rad * np.sin(ang), rng.uniform(2.8, 5.0),
                   rng.uniform(30.0, 60.0) + (88.0 - rad) * rng.uniform(0.6, 1.5),
                   power=(2.2, 0.6), flare=1.2)
print("stalactites placed")

# fine calcite texture over everything (sum of a few skewed sine waves)
dirs = rng.normal(size=(6, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
freqs = rng.uniform(0.9, 1.9, 6)
phases = rng.uniform(0, 2 * np.pi, 6)
for i in range(0, NX, 32):
    X = xs[i:i + 32, None, None]
    Y = xs[None, :, None]
    H = hs[None, None, :]
    n = sum(np.sin(f * (d[0] * X + d[1] * Y + d[2] * H) + p)
            for d, f, p in zip(dirs, freqs, phases))
    detail[i:i + 32] += (0.11 * n).astype(F32)

# the ceiling rim: a thin disc with a gently wavy edge, filleted into the cluster
X = xs[:, None, None]
Y = xs[None, :, None]
H = hs[None, None, :]
rho2d = np.sqrt(X * X + Y * Y)
th2d = np.arctan2(Y, X)
rim_r = FLANGE_R - 3.0 + 1.5 * np.sin(5 * th2d + 1.0) + 1.0 * np.sin(9 * th2d + 4.0) \
    + 0.6 * np.sin(14 * th2d + 2.0)
for k in range(0, np.searchsorted(hs, FLANGE_T + 12.0)):
    d_fl = np.maximum(rho2d - rim_r, hs[k] - FLANGE_T)[:, :, 0]
    detail[:, :, k] = smin(detail[:, :, k], d_fl.astype(F32), 4.0)
del rho2d, th2d, rim_r


# ---------------------------------------------------------------- hollowing
def redistance(phi, band):
    """Turn an approximate field into a true distance field near its surface."""
    d = skfmm.distance(phi.astype(np.float64), dx=VOX, narrow=band)
    out = np.where(phi > 0, band, -band).astype(F32)
    m = ~np.ma.getmaskarray(d)
    out[m] = np.ma.getdata(d)[m].astype(F32)
    return out


print("measuring true distances ...")
detail = redistance(detail, 8.0)
base = redistance(base, 8.0)

# cavity: WALL inside the smooth form, but never closer than MIN_WALL to the
# textured surface. Negative = inside the cavity.
cavity = np.maximum(base + WALL, detail + MIN_WALL)
del base
cavity = redistance(cavity, 8.0)

# solid = inside the textured surface, outside the cavity, below the ceiling plane
Hfull = np.broadcast_to(hs[None, None, :], detail.shape)
solid = np.maximum(np.maximum(detail, -cavity), -Hfull)

# bulb clearance report: largest free cylinder around the axis at each depth
R2 = (xs[:, None] ** 2 + xs[None, :] ** 2)
print("free radius around the axis inside the shell:")
clear = {}
for hq in range(0, 151, 10):
    k = np.searchsorted(hs, hq + 0.5)
    blocked = cavity[:, :, k] >= 0
    clear[hq] = float(np.sqrt(R2[blocked].min())) if blocked.any() else HALF
    print(f"   depth {hq:3d} mm : r = {clear[hq]:5.1f} mm")

# ---------------------------------------------------------------- meshing
print("meshing ...")
verts, faces, _, _ = measure.marching_cubes(solid, level=0.0, spacing=(VOX, VOX, VOX))
verts += np.array([xs[0], xs[0], hs[0]])
del solid
mesh = trimesh.Trimesh(verts, faces, process=False)
parts = mesh.split(only_watertight=False)
mesh = max(parts, key=lambda m: len(m.faces))     # drop stray specks / sealed voids
print(f"   raw: {len(mesh.faces):,} faces, watertight={mesh.is_watertight}")


def decimate(m, target):
    if len(m.faces) <= target:
        return m.copy()
    v, f = fast_simplification.simplify(np.asarray(m.vertices, np.float32),
                                        np.asarray(m.faces), target_count=target, agg=4)
    out = trimesh.Trimesh(v, f, process=True)
    out.update_faces(out.nondegenerate_faces())
    out.update_faces(out.unique_faces())
    out.remove_unreferenced_vertices()
    out.fix_normals()
    return out


def write_stl(path, m):
    tri = np.asarray(m.triangles, np.float32)
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = np.asarray(m.face_normals, np.float32)
    rec["v"] = tri
    with open(path, "wb") as fh:
        fh.write(b"stalactite chandelier".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tri)))
        fh.write(rec.tobytes())


stl = decimate(mesh, args.stl_faces)
stl.vertices[:, 2] -= stl.vertices[:, 2].min()        # sit exactly on the bed
stl.apply_scale(args.scale)
write_stl(os.path.join(args.out, "stalactite_chandelier.stl"), stl)
ext = stl.extents
print(f"STL: {len(stl.faces):,} faces, watertight={stl.is_watertight}, "
      f"size {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm")
_, edge_use = np.unique(np.sort(stl.edges, axis=1), axis=0, return_counts=True)
print(f"     edges not shared by exactly two faces: {(edge_use != 2).sum()}")
print(f"     volume {stl.volume / 1000:.0f} cm3  (~{stl.volume / 1000 * 1.24:.0f} g of PLA at 100% wall fill)")

# printability: share of surface that overhangs by more than 45 / 60 degrees
# when printed tips-up (faces on the bed excluded)
nz = stl.face_normals[:, 2]
off_bed = stl.triangles_center[:, 2] > 0.3 * args.scale
area = stl.area_faces
for deg in (45, 60, 75):
    frac = area[(nz < -np.sin(np.radians(deg))) & off_bed].sum() / area.sum()
    print(f"     overhang steeper than {deg} deg: {100 * frac:.2f}% of surface")

# ---------------------------------------------------------------- glow preview
ply = decimate(mesh, args.ply_faces)
v = np.asarray(ply.vertices)
idx = ((v - np.array([xs[0], xs[0], hs[0]])) / VOX).T
thick = np.clip(ndimage.map_coordinates(cavity, idx, order=1, mode="nearest"), 0, 8)
dist = np.linalg.norm(v - np.array([0, 0, BULB_H]), axis=1)
glow = np.exp(-thick / 1.9) * (0.30 + 0.70 / (1 + (dist / 85.0) ** 2))
glow = np.clip(glow / np.percentile(glow, 99.5), 0, 1) ** 0.8
# warm white ramp: deep amber in the thick parts to near white in the thin
c0, c1, c2 = np.array([70, 30, 8]), np.array([255, 170, 80]), np.array([255, 248, 228])
g = glow[:, None]
col = np.where(g < 0.5, c0 + (c1 - c0) * (g / 0.5), c1 + (c2 - c1) * ((g - 0.5) / 0.5))
col = np.clip(col, 0, 255).astype(np.uint8)

# hanging orientation, Y up, for the viewer
pv = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1).astype("<f4")
vrec = np.zeros(len(pv), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
vrec["p"], vrec["c"] = pv, col
frec = np.zeros(len(ply.faces), dtype=[("n", "u1"), ("i", "<i4", 3)])
frec["n"], frec["i"] = 3, ply.faces
with open(os.path.join(args.out, "preview.ply"), "wb") as fh:
    fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pv)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(frec)}\nproperty list uchar int vertex_indices\n"
              "end_header\n").encode())
    fh.write(vrec.tobytes())
    fh.write(frec.tobytes())
print(f"PLY: {len(ply.faces):,} faces")
