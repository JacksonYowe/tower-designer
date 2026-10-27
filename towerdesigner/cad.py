"""3D models: the tower (STEP for Onshape) and the 3D-printable assembly jig.

The jig is made of:
  * base plate  - pockets that locate every leg foot and the base ring, plus a socket for the mast
  * mast        - 12 mm square rod up the middle (thinner than the 25 mm test pole on purpose)
  * spacers     - square tubes that stack on the mast and set each collar's height; a disc on top holds the segments
  * collars     - split into segments; notches hold every leg at the right radius and angle.
                  After the glue dries the mast is pulled out the top and the segments drop out
                  through the base ring.
  * cutting guides - mitre channels for any sticks you cut by hand
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cadquery as cq
import numpy as np

from .sections import Section
from .structure import Part, Structure

MAST = 12.0          # mm, square mast
MAST_CLEAR = 0.4     # mm, hole clearance around the mast
SPACER_OD = 18.0     # mm, square spacer tubes
COLLAR_T = 5.0       # mm, collar thickness
FIT = 0.25           # mm, clearance around wood in the jig
PLATE_T = 5.0        # mm, base plate thickness (top sits 2 mm above the tower's base plane)
POCKET = 2.0         # mm, how deep the legs sit in the base plate
MAX_COLLARS = 6      # collar levels in the jig
KEY = 3.0            # mm, one corner of the mast is cut off this much so everything on it fits one way only
SLOT = 3.3           # mm, slot width in the preparation jigs (from the fit-test coupon, Spec.jig_slot)
HUB_T = 2.0          # mm, support disc on top of each spacer that the collar segments rest on

COLORS = {
    "leg": (0.80, 0.55, 0.30), "corner": (0.80, 0.55, 0.30), "brace": (0.95, 0.80, 0.55),
    "ring": (0.90, 0.70, 0.45), "cap": (0.70, 0.45, 0.25), "base": (0.70, 0.45, 0.25),
    "panel": (0.92, 0.76, 0.50), "diag": (0.85, 0.62, 0.38), "gusset": (0.98, 0.88, 0.70),
}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _frame(p0, p1, ref):
    x = _unit(p1 - p0)
    r = np.asarray(ref if ref is not None else (0, 0, 1.0), float)
    y = r - (r @ x) * x
    if np.linalg.norm(y) < 1e-6:
        r = np.array([1.0, 0, 0]) if abs(x[0]) < 0.9 else np.array([0, 1.0, 0])
        y = r - (r @ x) * x
    return x, _unit(y)


def member_solid(p0, p1, sec, ref, grow: float = 0.0, extend: float = 0.0) -> cq.Shape:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    x, y = _frame(p0, p1, ref)
    L = float(np.linalg.norm(p1 - p0)) + 2 * extend
    o = p0 - x * extend
    wp = cq.Workplane(cq.Plane(origin=tuple(o), xDir=tuple(y), normal=tuple(x)))
    w, t = sec.w + 2 * grow, sec.t + 2 * grow
    if sec.shape == "rect":
        wp = wp.rect(w, t)
    elif sec.shape == "round":
        wp = wp.circle(w / 2)
    elif grow > 0 or sec.shape == "box" and sec.w - 2 * sec.t < 0.5:
        wp = wp.rect(w, w)
    elif sec.shape == "box":
        wp = wp.rect(sec.w, sec.w).rect(sec.w - 2 * sec.t, sec.w - 2 * sec.t)
    else:  # angle, corner on the -y/-z side
        a, tt = sec.w / 2, sec.t
        wp = wp.polyline([(-a, -a), (a, -a), (a, -a + tt), (-a + tt, -a + tt), (-a + tt, a), (-a, a)]).close()
    return wp.extrude(L).val()


def _poly_face(rings: list[np.ndarray]) -> cq.Face:
    wires = [cq.Wire.makePolygon([cq.Vector(float(x), float(y), 0) for x, y in r], close=True) for r in rings]
    return cq.Face.makeFromWires(wires[0], wires[1:])


def frame_solid(part: Part, grow: float = 0.0, solid_fill: bool = False) -> cq.Shape:
    """One-piece laser-cut ring/cap, put back into tower coordinates."""
    g, z, t = part.info["g_ang"], part.info["z"], part.info["t"]
    rings = part.outline[0]
    if solid_fill:
        face = _poly_face([rings[0]])
        if grow:
            face = cq.Face.makeFromWires(face.outerWire().offset2D(grow)[0])
    else:
        face = _poly_face(rings)
    z0 = z - t if part.kind == "cap" else z
    solid = cq.Solid.extrudeLinear(face, cq.Vector(0, 0, t + 2 * grow))
    solid = solid.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), math.degrees(g))
    return solid.translate(cq.Vector(0, 0, z0 - grow))


def _radial_offset(p0, p1, off):
    mid = (p0 + p1) / 2
    r = np.array([mid[0], mid[1], 0.0])
    r = r / (np.linalg.norm(r) or 1)
    return p0 + r * off, p1 + r * off


def _radial(p):
    r = np.array([p[0], p[1], 0.0])
    return r / (np.linalg.norm(r) or 1.0)


def _line_gap(a0, a1, b0, b1) -> float:
    """Shortest distance between two infinite lines."""
    u, v = a1 - a0, b1 - b0
    n = np.cross(u, v)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        w = b0 - a0
        return float(np.linalg.norm(w - (w @ u) / (u @ u) * u))
    return float(abs((b0 - a0) @ n) / nn)


def _straight_chain(st: Structure, part: Part):
    """(first node, last node, a member) if the part's members form one straight run, else None."""
    if not part.members:
        return None
    ms = [st.members[k] for k in part.members]
    if len(ms) == 1:
        m = ms[0]
        return st.nodes[m.i], st.nodes[m.j], m
    # order the chain: each member's start is the previous member's end
    ends = [ms[0].i, ms[0].j]
    for m in ms[1:]:
        if m.i == ends[-1]:
            ends.append(m.j)
        elif m.j == ends[-1]:
            ends.append(m.i)
        else:
            return None
    pts = np.array([st.nodes[n] for n in ends])
    a, b = pts[0], pts[-1]
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-6:
        return None
    u = d / L
    off = pts - a
    dev = np.linalg.norm(off - np.outer(off @ u, u), axis=1).max()
    return (a, b, ms[0]) if dev < 0.3 else None


_NORMALS: dict[int, tuple] = {}


def surface_normal(st: Structure, p) -> np.ndarray:
    """Outward normal of the tower surface (a surface of revolution through the A strips) at point p."""
    key = id(st)
    if key not in _NORMALS:
        a_line = None
        for part in st.parts:
            if part.kind in ("diag", "leg") and not part.offset and len(part.members) > 1:
                ch = _straight_chain(st, part)
                if ch is not None:
                    a_line = (ch[0], _unit(ch[1] - ch[0]))
                    break
        _NORMALS.clear()
        _NORMALS[key] = (st, a_line)
    _, a_line = _NORMALS[key]
    r = _radial(p)
    if a_line is None or abs(a_line[1][2]) < 1e-6:
        return r
    a0, u = a_line
    R = lambda z: float(np.hypot(*(a0 + (z - a0[2]) / u[2] * u)[:2]))
    dRdz = (R(p[2] + 0.5) - R(p[2] - 0.5)) / 1.0
    return _unit(r - np.array([0, 0, 1.0]) * dRdz)


def strip_line(st: Structure, pi: int):
    """Centre line (start, end, first member) of a straight strip as built: the outer (B) strips are pushed out
    until they just touch every strip they cross."""
    part = st.parts[pi]
    p0, p1, m = _straight_chain(st, part)
    if part.offset:
        crossed = []
        for jt in st.joints:
            if jt.kind == "lap" and pi in jt.parts[:2]:
                other = st.parts[jt.parts[0] if jt.parts[1] == pi else jt.parts[1]]
                ch = _straight_chain(st, other)
                if ch is not None and not other.offset and other.kind in ("diag", "leg"):
                    crossed.append((ch[0], ch[1]))
        k = 1.0
        for _ in range(80):
            q0 = p0 + _radial(p0) * part.offset * k
            q1 = p1 + _radial(p1) * part.offset * k
            if not crossed:
                break
            if min(_line_gap(q0, q1, a0, a1) for a0, a1 in crossed) >= part.offset * 1.005:
                break
            k += 0.0025
        p0, p1 = q0, q1
    return p0, p1, m


def strip_twist(st: Structure, pi: int) -> float:
    """How far (degrees) a strip turns about its own length from foot to top to stay flat to the surface."""
    q0, q1, _ = strip_line(st, pi)
    u = _unit(q1 - q0)
    f = [(_unit(n - (n @ u) * u)) for n in (surface_normal(st, q0), surface_normal(st, q1))]
    return float(np.degrees(np.arccos(np.clip(f[0] @ f[1], -1, 1))))


def _twisted_stick(st: Structure, p0, p1, sec, grow: float, extend: float = 6.0, stations: int = 9) -> cq.Shape:
    """A straight stick twisted slightly along its length so its faces stay flat to the tower surface (how a
    strip really sits once it's seated in the collars and glued at every crossing)."""
    u = _unit(p1 - p0)
    L = float(np.linalg.norm(p1 - p0))
    wires = []
    for t in np.linspace(-extend / L, 1 + extend / L, stations):
        c = p0 + u * L * t
        n = surface_normal(st, c)
        nf = _unit(n - (n @ u) * u)
        side = np.cross(u, nf)
        pl = cq.Plane(origin=tuple(map(float, c)), xDir=tuple(map(float, side)), normal=tuple(map(float, u)))
        wires.append(cq.Workplane(pl).rect(sec.w + 2 * grow, sec.t + 2 * grow).wires().val())
    return cq.Solid.makeLoft(wires, False)


def _lap_ring_solid(st: Structure, part: Part, p0, p1, m, grow: float) -> cq.Shape:
    """Half-lap ring piece, lying on the outer face of the B strip at each end and tilted to the tower surface.
    Each end is cut along the middle of its strip (a plane through the strip's axis and its face normal), so
    neighbouring pieces meet flush on the strip."""
    dz = part.info.get("dz", 0.0)
    ends = []
    for node in (m.i, m.j):
        host = outer_strip_at(st, node, None)
        z = st.nodes[node][2] + dz
        if host is not None and st.parts[host].kind in ("diag", "leg"):
            h0, h1, _ = strip_line(st, host)
            hu = _unit(h1 - h0)
            hc = h0 + (z - h0[2]) / hu[2] * hu          # strip centre at this height
            n = surface_normal(st, hc)
            ends.append((hc + n * (st.parts[host].sec.size_z / 2 + part.sec.size_z / 2), hu, n))
        else:
            q = st.nodes[node] + _radial(st.nodes[node]) * part.offset + np.array([0, 0, dz])
            ends.append((q, None, _radial(q)))
    c0, c1 = ends[0][0], ends[1][0]
    nmid = surface_normal(st, (c0 + c1) / 2)
    u = _unit(c1 - c0)
    ref = np.cross(nmid, u)
    body = cq.Workplane().add(member_solid(c0, c1, m.sec, ref, grow, extend=4.0))
    for (c, axis, nface), other in ((ends[0], c1), (ends[1], c0)):
        if axis is None:
            continue
        n = np.cross(axis, nface)
        if np.linalg.norm(n) < 1e-6:
            continue
        n = _unit(n)
        if n @ (c - other) < 0:
            n = -n
        cut = cq.Workplane(cq.Plane(origin=tuple(map(float, c + n * grow)), xDir=tuple(map(float, axis)),
                                    normal=tuple(map(float, n)))).rect(200, 200).extrude(50)
        body = body.cut(cut)
    return body.val()


def part_solids(st: Structure, pi: int, grow: float = 0.0) -> list[cq.Shape]:
    part = st.parts[pi]
    if part.kind in ("cap", "base") and part.info.get("g_ang") is not None:
        return [frame_solid(part, grow, solid_fill=grow > 0)]
    out = []
    if part.kind == "corner":
        k = st.leg_parts.index(pi)
        n = len(st.leg_parts)
        nb = [st.parts[st.leg_parts[(k + d) % n]] for d in (-1, 1)]
        for m_i in part.members:
            m = st.members[m_i]
            p0, p1 = st.nodes[m.i], st.nodes[m.j]
            for other in nb:
                om = st.members[other.members[st.parts[pi].members.index(m_i)]]
                q = st.nodes[om.i] - p0
                x = _unit(p1 - p0)
                d = _unit(q - (q @ x) * x)
                # strip lies in the face, from the corner towards the neighbouring corner
                sec = Section("rect", m.sec.t, m.sec.w, m.sec.stock, True)
                c0 = p0 + d * (sec.w / 2)
                c1 = p1 + d * (sec.w / 2)
                out.append(member_solid(c0, c1, sec, d, grow))
        return out
    dz = part.info.get("dz", 0.0)
    chain = _straight_chain(st, part)
    if chain is not None:
        # one physical stick = one solid, drawn straight from its first node to its last
        p0, p1, m = chain
        if part.info.get("offsets"):
            # shingled ring piece: on the strip at its first end, on top of the next piece at its second,
            # running on past both crossings so each one is fully covered
            o0, o1 = part.info["offsets"]
            q0, q1 = p0 + _radial(p0) * o0, p1 + _radial(p1) * o1
            if dz:
                q0, q1 = q0 + np.array([0, 0, dz]), q1 + np.array([0, 0, dz])
            ext = max(0.0, (part.length - float(np.linalg.norm(q1 - q0))) / 2)
            return [member_solid(q0, q1, m.sec, m.ref, grow, extend=ext)]
        if part.kind in ("ring", "base", "cap") and "lap" in part.info.get("end_style", []) and len(part.members) == 1:
            return [_lap_ring_solid(st, part, p0, p1, m, grow)]
        if part.kind in ("diag", "leg") and len(part.members) > 1:
            q0, q1, _ = strip_line(st, pi)
            solid = _twisted_stick(st, q0, q1, m.sec, grow)
            if not grow:
                slab = cq.Solid.makeBox(1000, 1000, st.height, cq.Vector(-500, -500, 0))
                solid = solid.intersect(slab)
            return [solid]
        if part.offset:
            # push the straight outer stick out until it just touches every node it crosses
            # the sticks it crosses (centre lines), from its lap joints
            pi_here = st.parts.index(part)
            crossed = []
            for jt in st.joints:
                if jt.kind == "lap" and pi_here in jt.parts[:2]:
                    other = st.parts[jt.parts[0] if jt.parts[1] == pi_here else jt.parts[1]]
                    ch = _straight_chain(st, other)
                    if ch is not None and not other.offset:
                        crossed.append((ch[0], ch[1]))
            k = 1.0
            for _ in range(80):
                q0 = p0 + _radial(p0) * part.offset * k
                q1 = p1 + _radial(p1) * part.offset * k
                if not crossed:                   # rings: nothing crossed to clear, the offset is exact
                    break
                gap = min(_line_gap(q0, q1, a0, a1) for a0, a1 in crossed)
                if gap >= part.offset * 1.02:     # faces just touch (a hair of clearance for the corners)
                    break
                k += 0.01
            p0, p1 = q0, q1
        if dz:
            p0, p1 = p0 + np.array([0, 0, dz]), p1 + np.array([0, 0, dz])
        ref = m.ref
        if part.kind in ("ring", "base", "cap") and "lap" in part.info.get("end_style", []) and len(part.members) == 1:
            return [_lap_ring_solid(st, part, p0, p1, m, grow)]
        if len(part.members) > 1:
            # face the stick to the tower surface at its middle (a straight stick can't follow it everywhere)
            u = (p1 - p0) / np.linalg.norm(p1 - p0)
            ref = np.cross(_radial((p0 + p1) / 2), u)
        out.append(member_solid(p0, p1, m.sec, ref, grow))
    else:
        for m_i in part.members:
            m = st.members[m_i]
            p0, p1 = st.nodes[m.i], st.nodes[m.j]
            if part.offset:
                p0, p1 = _radial_offset(p0, p1, part.offset)
            if dz:
                p0, p1 = p0 + np.array([0, 0, dz]), p1 + np.array([0, 0, dz])
            out.append(member_solid(p0, p1, m.sec, m.ref, grow))
    if part.kind in ("diag", "leg") and not grow:
        # level-cut ends: trim to the table and to the top of the tower
        slab = cq.Solid.makeBox(1000, 1000, st.height, cq.Vector(-500, -500, 0))
        out = [s.intersect(slab) for s in out]
    return out


def tower_assembly(st: Structure) -> cq.Assembly:
    assy = cq.Assembly(name="tower")
    used = set()
    for pi, part in enumerate(st.parts):
        if part.kind == "gusset":
            continue
        solids = part_solids(st, pi)
        if not solids:
            continue
        name = part.name.replace(" ", "_")
        while name in used:
            name += "_"
        used.add(name)
        shape = solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
        assy.add(shape, name=name, color=cq.Color(*COLORS.get(part.kind, (0.8, 0.6, 0.4))))
    return assy


def save_assembly(assy: cq.Assembly, path: str):
    try:
        assy.export(path)
    except AttributeError:
        assy.save(path)


# ---------------------------------------------------------------------------------------------
# jig
# ---------------------------------------------------------------------------------------------
@dataclass
class JigPart:
    name: str
    shape: cq.Shape
    qty: int
    note: str
    label: str = ""      # short ID engraved on the part and used in the guide (BP, M, SP2, C3-4, CG1, SG, SJ1)
    engraved: bool = False   # its text is already cut in (engrave_label leaves it alone)


def _clearance_solids(st: Structure, zmin: float, zmax: float) -> list[cq.Shape]:
    out = []
    for pi, part in enumerate(st.parts):
        if part.kind == "gusset":
            continue
        zs = [st.nodes[st.members[m].i][2] for m in part.members] + [st.nodes[st.members[m].j][2] for m in part.members]
        if part.kind in ("cap", "base") and part.info.get("z") is not None:
            zs = [part.info["z"]]
        if not zs or max(zs) < zmin - 1 or min(zs) > zmax + 1:
            continue
        out.extend(part_solids(st, pi, grow=FIT))
    return out


def _leg_points(st: Structure, z: float) -> list[np.ndarray]:
    pts = []
    for pi in st.leg_parts:
        for m_i in st.parts[pi].members:
            m = st.members[m_i]
            a, b = st.nodes[m.i], st.nodes[m.j]
            lo, hi = sorted((a[2], b[2]))
            if lo - 1e-9 <= z <= hi + 1e-9:
                s = (z - a[2]) / (b[2] - a[2]) if abs(b[2] - a[2]) > 1e-9 else 0.0
                pts.append(a + s * (b - a))
                break
    pts.sort(key=lambda p: math.atan2(p[1], p[0]))
    return pts


def _bb_overlap(a, b, pad=0.5) -> bool:
    return not (a.xmax + pad < b.xmin or b.xmax + pad < a.xmin or a.ymax + pad < b.ymin or
                b.ymax + pad < a.ymin or a.zmax + pad < b.zmin or b.zmax + pad < a.zmin)


def _cut_all(body: cq.Workplane, tools: list[cq.Shape]) -> cq.Workplane:
    """Subtract many (overlapping) tools robustly: fuse them first, fall back to one by one."""
    bb = body.val().BoundingBox()
    tools = [t for t in tools if _bb_overlap(bb, t.BoundingBox())]
    if not tools:
        return body
    try:
        fused = tools[0].fuse(*tools[1:]).clean() if len(tools) > 1 else tools[0]
        out = body.cut(cq.Workplane().add(fused))
        if out.val().isValid():
            return out
    except Exception:
        pass
    for t in tools:
        try:
            nxt = body.cut(cq.Workplane().add(t))
            if nxt.val().isValid():
                body = nxt
        except Exception:
            pass
    return body


def collar_heights(st: Structure) -> list[float]:
    lv = st.levels
    zs = [(a + b) / 2 for a, b in zip(lv[:-1], lv[1:])]
    out = []
    for z in zs:  # keep at least 12 mm between collars so spacers can be printed
        if not out or z - out[-1] >= COLLAR_T + 7:
            out.append(z)
    out = [z for z in out if z > PLATE_T + 10]
    if len(out) > MAX_COLLARS:  # a few evenly spread collars are enough to line the legs up
        idx = np.linspace(0, len(out) - 1, MAX_COLLARS).round().astype(int)
        out = [out[i] for i in sorted(set(idx))]
    return out


def set_fit(slot: float):
    """Jig slot size from the fit-test coupon: preparation-jig slots = slot, collar notches and pockets a
    little looser (3.3 mm slot -> 0.25 mm clearance round the stick)."""
    global SLOT, FIT
    SLOT = float(slot)
    FIT = max(0.1, (SLOT - 3.0) / 2 + 0.1)


def _keyed(size: float, chamfer: float) -> list[tuple[float, float]]:
    """Square with its +x+y corner cut off, so the mast and every hole on it only fit one way round."""
    h = size / 2
    return [(-h, -h), (h, -h), (h, h - chamfer), (h - chamfer, h), (-h, h)]


def _mast_hole(clear: float, gap: float = 0.2, height: float = 400.0, z0: float = -200.0) -> cq.Workplane:
    """Keyed hole for the mast: `clear` round the flats, at least `gap` across the cut corner."""
    pts = _keyed(MAST + clear, KEY + clear - math.sqrt(2) * gap)
    return cq.Workplane().polyline(pts).close().extrude(height).translate((0, 0, z0))


def _text_cut(body: cq.Workplane, txt: str, size: float, x: float, y: float, z: float,
              plane: str = "XY", depth: float = 0.5, halign: str = "center") -> cq.Workplane:
    """Engrave text into the face at (x, y, z). Never fails: a font problem just leaves the face blank."""
    try:
        wp = (cq.Workplane(cq.Plane((x, y, z), plane[0], plane[1])) if isinstance(plane, tuple)
              else cq.Workplane(plane, origin=(x, y, z)))       # plane: name, or (x direction, normal)
        tool = wp.text(txt, size, -depth, halign=halign, kind="bold")
        out = body.cut(tool)
        if out.val().isValid():
            return out
    except Exception:
        pass
    return body


def jig_stack(st: Structure) -> tuple[float, list[dict]]:
    """Height of the base-plate boss, and everything stacked on the mast, bottom first, with its exact
    heights. Spacer lengths are rounded to 0.1 mm against the running total, so the rounding never adds up."""
    c = collar_heights(st)
    boss = min(14.0, (c[0] - COLLAR_T / 2 - 2) if c else 14.0)
    out, top = [], boss
    for k, zc in enumerate(c, start=1):
        gap = round(zc - COLLAR_T / 2 - top, 1)
        if gap >= 0.8:
            out.append(dict(kind="spacer", k=k, z0=top, z1=top + gap, length=gap))
            top += gap
        out.append(dict(kind="collar", k=k, zc=zc, z0=top, z1=top + COLLAR_T))
        top += COLLAR_T
    return boss, out


def _hub(k: int, ln: float, r_disc: float, angles: list[float]) -> JigPart:
    """Spacer tube with a support disc on top. The collar segments rest on the disc (so they can't tip
    off), and the disc is printed with the number of the segment that goes in each place."""
    t = min(HUB_T, ln)
    body = cq.Workplane().circle(r_disc).extrude(t).translate((0, 0, ln - t))
    if ln - t > 0.2:
        body = body.union(cq.Workplane().rect(SPACER_OD, SPACER_OD).extrude(ln - t + 0.01))
    body = body.cut(_mast_hole(MAST_CLEAR))
    for si, a in enumerate(angles, start=1):
        r = r_disc - 7
        body = _text_cut(body, str(si), 5, r * math.cos(a), r * math.sin(a), ln)
    body = _text_cut(body, f"SP{k}", 4.5, 0, -15, ln)
    body = _text_cut(body, f"under C{k}", 3, 0, -21.5, ln)
    return JigPart(f"spacer_{k}", body.val(), 1,
                   f"Spacer SP{k}: {ln:.1f} mm tall with a support disc on top for collar C{k}. Each number on the "
                   f"disc is where that collar segment goes.", label=f"SP{k}", engraved=True)


def make_jig(st: Structure, spec) -> tuple[list[JigPart], list[str]]:
    set_fit(getattr(spec, "jig_slot", 3.3))
    notes = []
    parts: list[JigPart] = [fit_coupon()]
    hole = MAST + MAST_CLEAR
    boss_top, stack = jig_stack(st)

    # ---- base plate ------------------------------------------------------------------------
    r_foot = max(float(np.hypot(*st.nodes[s][:2])) for s in st.support_nodes)
    for p in st.parts:
        if p.kind == "base" and p.outline and p.info.get("g_ang") is not None:
            r_foot = max(r_foot, float(np.max(np.hypot(p.outline[0][0][:, 0], p.outline[0][0][:, 1]))))
    half = r_foot + 12
    bed = min(spec.bed[0], spec.bed[1]) - 4
    if 2 * half > bed:
        notes.append(f"Base plate is {2 * half:.0f} mm wide — bigger than the printer bed; it is split in two.")
    plate = cq.Workplane().box(2 * half, 2 * half, PLATE_T, centered=(True, True, False)).translate(
        (0, 0, POCKET - PLATE_T))
    boss = cq.Workplane().circle(10).extrude(boss_top - POCKET).translate((0, 0, POCKET))
    plate = plate.union(boss)
    # mast socket: keyed, and a snug fit so the mast stands upright
    plate = plate.cut(_mast_hole(0.2, 0.1, 40, -10))
    # finger holes to push the tower out of the pockets
    for a in (45, 135, 225, 315):
        r = half - 6
        plate = plate.cut(cq.Workplane().circle(3.5).extrude(20).translate(
            (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), -10)))
    plate = _cut_all(plate, _clearance_solids(st, -1, POCKET + 1))
    try:
        plate = plate.faces(">Z").workplane().center(0, -half + 5).text("BASE", 5, -0.6, kind="bold")
    except Exception:
        pass
    # start mark just outside the first foot (strips A1 and B1) so you can find your place round the ring
    try:
        p0 = st.nodes[st.support_nodes[0]]
        r0 = float(np.hypot(p0[0], p0[1]))
        ux, uy = p0[0] / r0, p0[1] / r0
        rr = min(r0 + 9, half - 5)
        ids = _part_ids(st)
        at = [ids[i] for i, p in enumerate(st.parts) if p.kind in ("diag", "leg") and p.members and
              min(np.linalg.norm(st.nodes[n][:2] - p0[:2]) for m in p.members
                  for n in (st.members[m].i, st.members[m].j)) < 1.0]
        mark = " ".join(sorted(at)[:2]) or "1"
        plate = plate.cut(cq.Workplane().text(mark, 4.5, 0.6, kind="bold")
                          .translate((rr * ux, rr * uy, POCKET - 0.6)))
    except Exception:
        pass
    if 2 * half > bed:
        left = plate.intersect(cq.Workplane().box(2 * half, 2 * half, 100).translate((-half, 0, 0)))
        right = plate.intersect(cq.Workplane().box(2 * half, 2 * half, 100).translate((half, 0, 0)))
        parts.append(JigPart("base_plate_left", left.val(), 1, "Glue or tape the halves together."))
        parts.append(JigPart("base_plate_right", right.val(), 1, ""))
    else:
        parts.append(JigPart("base_plate", plate.val(), 1, "Print top side up. Pockets hold the leg feet and base ring."))

    # ---- collars (numbered anticlockwise from the start mark) + spacers with support discs ---------
    seg_angles: dict[int, list[float]] = {}
    r_in: dict[int, float] = {}
    for item in stack:
        if item["kind"] != "collar":
            continue
        ci, zc = item["k"], item["zc"]
        pts = _leg_points(st, zc)
        if len(pts) < 3:
            continue
        r_in[ci] = min(float(np.hypot(p[0], p[1])) for p in pts)
        poly = [(float(p[0]), float(p[1])) for p in pts]
        body = (cq.Workplane().polyline(poly).close().extrude(COLLAR_T)
                .translate((0, 0, zc - COLLAR_T / 2)))
        body = body.cut(_mast_hole(MAST_CLEAR))
        body = _cut_all(body, _clearance_solids(st, zc - COLLAR_T, zc + COLLAR_T))
        # split into segments between the notches so they can be taken out after gluing
        angs = sorted(math.atan2(p[1], p[0]) for p in pts)
        # two notches per segment when there are many legs/strips: still slides out (notches are only
        # half a stick deep and nearly parallel) but halves the number of printed pieces
        group = 2 if len(angs) > 8 else 1
        cuts = []
        for k in range(len(angs)):
            if (k + 1) % group and k != len(angs) - 1:
                continue
            a0, a1 = angs[k], angs[(k + 1) % len(angs)] + (2 * math.pi if k == len(angs) - 1 else 0)
            am = (a0 + a1) / 2
            R = 200
            slot = (cq.Workplane().rect(R, 0.6).extrude(COLLAR_T + 2)
                    .translate((R / 2, 0, zc - COLLAR_T / 2 - 1))
                    .rotate((0, 0, 0), (0, 0, 1), math.degrees(am)))
            cuts.append(slot.val())
        body = _cut_all(body, cuts)
        segs = [s for s in body.solids().vals() if s.Volume() > 20]
        segs.sort(key=lambda s: math.atan2(s.Center().y, s.Center().x) % (2 * math.pi))
        seg_angles[ci] = [math.atan2(s.Center().y, s.Center().x) for s in segs]
        for si, sgm in enumerate(segs, start=1):
            parts.append(JigPart(f"collar_{ci}_seg{si}", sgm.translate(cq.Vector(0, 0, -(zc - COLLAR_T / 2))), 1,
                                 f"Collar {ci} (centre {zc:.1f} mm up), segment {si} of {len(segs)}.",
                                 label=f"C{ci}-{si}"))
    for item in stack:
        if item["kind"] == "spacer":
            k = item["k"]
            parts.append(_hub(k, item["length"], max(14.0, r_in.get(k, 40.0) - 10), seg_angles.get(k, [])))

    # ---- mast --------------------------------------------------------------------------------
    top = stack[-1]["z1"] if stack else boss_top
    mast_len = max(top + 15, st.height * 0.6) - (POCKET - PLATE_T)
    if mast_len > max(spec.bed) * 1.38:
        notes.append("Mast is longer than the printer can fit diagonally; use a 12 mm square stick instead.")
    mast = cq.Workplane().polyline(_keyed(MAST, KEY)).close().extrude(mast_len).edges("|Z").chamfer(0.5)
    parts.append(JigPart("mast", mast.val(), 1, f"{MAST:.0f} mm square with one corner cut off (so everything "
                                               f"on it only fits one way), {mast_len:.0f} mm long. Print lying flat, "
                                               "diagonally on the bed.", label="M"))

    # will the collar segments drop out through the base ring?
    base_parts = [p for p in st.parts if p.kind == "base"]
    if base_parts and base_parts[0].outline and len(base_parts[0].outline[0]) > 1:
        inner = base_parts[0].outline[0][1]
        r_open = float(np.min(np.hypot(inner[:, 0], inner[:, 1])))
        seg_max = max((max(s.shape.BoundingBox().xlen, s.shape.BoundingBox().ylen)
                       for s in parts if s.name.startswith("collar")), default=0)
        if seg_max > 2 * r_open:
            notes.append("Some collar segments are wider than the base ring opening: cut them in half with a knife "
                         "before removing, or remove them before the base ring is glued.")
    cg = cutting_guides(st)
    for k, jp in enumerate(cg, start=1):
        jp.label = f"CG{k}"
    parts.extend(cg)
    parts.extend(sanding_jigs(st))
    mg = marking_gauge(st)
    if mg:
        parts.append(mg)
    if any("scarf" in p.info.get("end_style", []) for p in st.parts):
        parts.append(scarf_guide())
    for jp in parts:
        if jp.name.startswith("base_plate") and not jp.label:
            jp.label = "BP" if jp.name == "base_plate" else ("BP-L" if jp.name.endswith("left") else "BP-R")
        jp.shape = engrave_label(jp)
    return parts, notes


def _part_ids(st: Structure) -> list[str]:
    from .lasercut import part_ids
    return part_ids(st)


# ---- preparation jigs ------------------------------------------------------------------------------
def outer_strip_at(st: Structure, node: int, default: int) -> int:
    """The outermost strip passing through a node: what a ring end there is really glued to."""
    at = [pi for pi, p in enumerate(st.parts) if p.kind in ("diag", "leg") and any(
        node in (st.members[m].i, st.members[m].j) for m in p.members)]
    return max(at, key=lambda pi: st.parts[pi].offset) if at else default


def level_parts(st: Structure) -> list[int]:
    """Straight strips with a level cut at both ends (the diagrid strips)."""
    return [pi for pi, p in enumerate(st.parts) if p.sec is not None and not p.laser
            and tuple(p.info.get("end_style", ())) == ("level", "level") and _straight_chain(st, p) is not None]


def strip_frame(st: Structure, pi: int):
    """Foot point, axis (foot -> top), side-face normal and outer-face normal of a straight strip, with
    its faces turned exactly as in the 3D model."""
    a, b, _ = _straight_chain(st, st.parts[pi])
    if a[2] > b[2]:
        a, b = b, a
    u = _unit(b - a)
    ref = np.cross(_radial((a + b) / 2), u)
    y = _unit(ref - (ref @ u) * u)
    return a, u, y, np.cross(u, y)


def level_angle(st: Structure, pi: int) -> float:
    """Saw angle (deg) in the cutting guide for a level-cut strip lying with its outer (letter) face up and
    its foot to the left; the sign says which way the cut slopes."""
    _, u, y, _ = strip_frame(st, pi)
    return math.degrees(math.atan2(y[2], u[2]))


KERF = 0.5   # mm, razor-saw cut width; the cutting-guide stops allow for it (saw on the waste side of the slot)


def ring_cut_angles(st: Structure, pi: int) -> tuple[float, float] | None:
    """Saw angles (deg) for a half-lap ring piece lying outer (O) face up in the cutting guide, end 1 on the left:
    each end is cut along the strip it sits on, so the angle is that strip's lean seen on the piece's outer face."""
    part = st.parts[pi]
    m = st.members[part.members[0]]
    ends = []
    for node in (m.i, m.j):
        host = outer_strip_at(st, node, None)
        if host is None or st.parts[host].kind not in ("diag", "leg"):
            return None
        h0, h1, _ = strip_line(st, host)
        hu = _unit(h1 - h0)
        z = st.nodes[node][2]
        ends.append((h0 + (z - h0[2]) / hu[2] * hu, hu))
    c0, c1 = ends[0][0], ends[1][0]
    x = _unit(c1 - c0)
    zf = surface_normal(st, (c0 + c1) / 2)
    zf = _unit(zf - (zf @ x) * x)
    y = np.cross(zf, x)
    out = []
    for _, hu in ends:
        d = np.array([hu @ x, hu @ y])
        if d[1] < 0:
            d = -d
        out.append(math.degrees(math.atan2(-d[0], d[1])))
    return out[0], out[1]


def cut_channels(st: Structure) -> list[dict]:
    """What the cutting guides hold: guide 1 = strips, guide 2 = everything else, longest first.
    Every piece whose two ends are (near enough) parallel is cut against a stop: no measuring, no waste.
    Level-cut strips are left a little long for the sanding jig; scarfed or shingled ring pieces are cut square;
    anything with truly different end angles is cut between two slots."""
    groups: dict[tuple, list[int]] = {}
    level = set(level_parts(st))
    ang = {}
    for pi, p in enumerate(st.parts):
        if p.laser or p.sec is None or p.kind == "corner":
            continue
        es = tuple(p.info.get("end_style", ("", "")))
        L = round(p.length, 1)
        if pi in level:
            key = (p.stock, L, "parallel", round(abs(level_angle(st, pi)), 1))
        elif "scarf" in es or "shingle" in es:
            key = (p.stock, L, "square", 0.0)
        else:
            ra = ring_cut_angles(st, pi) if p.kind in ("ring", "base", "cap") and "lap" in es else None
            if ra is not None and abs(ra[0] - ra[1]) <= 2.0:
                ang[pi] = (ra[0] + ra[1]) / 2       # ends parallel within 1 deg: cut against a stop
                key = (p.stock, L, "parallel", round(ang[pi], 1))
            else:
                key = (p.stock, L, round(p.end_angles[0]), round(p.end_angles[1]))
        groups.setdefault(key, []).append(pi)
    out = []
    count = {1: 0, 2: 0}
    for key, pis in sorted(groups.items(), key=lambda kv: (-kv[0][1], str(kv[0]))):
        pis = sorted(pis, key=lambda i: (st.parts[i].group or "", i))
        strips = all(i in level for i in pis)
        g = 1 if strips else 2
        count[g] += 1
        if key[2] == "parallel":
            beta = level_angle(st, pis[0]) if strips else float(np.mean([ang[i] for i in pis]))
        else:
            beta = 0.0
        names = sorted({st.parts[i].group or st.parts[i].name for i in pis})
        short = "STRIPS" if all(nm.startswith("Strip") for nm in names) else "/".join(names).upper()
        L = float(np.mean([st.parts[i].length for i in pis]))
        if strips:
            from .tolerances import SAND_ALLOW
            L += 2 * SAND_ALLOW          # left on for the sanding jig to clean up both ends
        out.append(dict(key=key, parts=pis, n=len(out) + 1, guide=g, ch=count[g],
                        L=L, beta=beta, mode=key[2]
                        if key[2] in ("parallel", "square") else "two", name=short, strips=strips,
                        a0=key[2] if key[2] not in ("parallel", "square") else 90,
                        a1=key[3] if key[2] not in ("parallel", "square") else 90))
    return out


def channel_of(st: Structure) -> dict[int, dict]:
    """Part index -> its cutting-guide channel."""
    return {pi: ch for ch in cut_channels(st) for pi in ch["parts"]}


def needs_roll(st: Structure, pi: int, ch: dict) -> bool:
    """True if this strip is the mirror-image hand of its channel's cut: turn it over before lettering."""
    return ch["mode"] == "parallel" and level_angle(st, pi) * ch["beta"] < 0


def cutting_guides(st: Structure) -> list[JigPart]:
    chans = cut_channels(st)
    out = []
    for g in sorted({c["guide"] for c in chans}):
        out.append(_guide_plate([c for c in chans if c["guide"] == g], g))
    return out


def _guide_plate(chans: list[dict], idx: int) -> JigPart:
    """Cutting guide: one channel per piece, each with a tall saw guide over its slot (keeps the saw square and
    holds the stick down) and a stop exactly one piece-length away; channel number, name, length and count are
    engraved beside each channel."""
    wall, base, margin, top, tower_h = 4.5, 3.0, 12.0, 8.0, 18.0
    w, xs = SLOT, 14.0                      # channel width, saw slot position
    ends = []
    for c in chans:
        b = math.radians(c["beta"])
        if c["mode"] == "two":
            ends.append(5 + 3 + c["L"] + 12)
        else:
            ends.append(xs + c["L"] + abs(math.tan(b)) * w / 2 + 9)
    length = max(max(ends), 70.0)
    width = margin + sum(w + wall for _ in chans)
    body = cq.Workplane().box(length, width, top, centered=(False, False, False))
    # saw-guide tower across every channel at the slot
    body = body.union(cq.Workplane().box(10, width, tower_h, centered=(False, False, False)).translate((xs - 5, 0, 0)))
    y = margin
    notes = []
    for c in chans:
        yc = y + w / 2
        b = math.radians(c["beta"])
        if c["mode"] == "two":
            body = body.cut(cq.Workplane().box(length - 10, w, w + 0.4, centered=(False, False, False))
                            .translate((5, y, base)))
            x1 = 5 + 1.5 + abs(math.tan(math.radians(90 - c["a0"]))) * w / 2
            for xpos, ang in ((x1, c["a0"]), (x1 + c["L"] + 2 * KERF - 0.8, c["a1"])):
                slot = (cq.Workplane().box(0.8, w + 2 * wall + 2, 30).translate((0, 0, 15))
                        .rotate((0, 0, 0), (0, 0, 1), 90 - ang).translate((xpos, yc, base)))
                body = body.cut(slot)
            how = "end against the stop, saw both slots on their waste side"
        else:
            # stick enters from the left under the saw guide; the stop sits one piece-length past the slot,
            # allowing for the saw cut when you saw against the slot's left (waste) side
            xstop = xs + c["L"] + (KERF - 0.4) / math.cos(b)
            t = math.tan(b)
            xl = lambda yy: xstop - t * (yy - yc)
            chan = (cq.Workplane().polyline([(-1, y), (xl(y), y), (xl(y + w), y + w), (-1, y + w)]).close()
                    .extrude(w + 0.4).translate((0, 0, base)))
            body = body.cut(chan)
            # open top beyond the saw guide so you can see and lift the piece
            body = body.cut(cq.Workplane().polyline([(xs + 5, y), (xl(y), y), (xl(y + w), y + w), (xs + 5, y + w)])
                            .close().extrude(30).translate((0, 0, base)))
            slot = (cq.Workplane().box(0.8, w + 2 * wall + 2, 30).translate((0, 0, 15))
                    .rotate((0, 0, 0), (0, 0, 1), c["beta"]).translate((xs, yc, base)))
            body = body.cut(slot)
            how = "push the stick to the stop, saw at the slot on its left (waste) side"
        body = _text_cut(body, f"{c['ch']} {c['name']} {c['L']:.1f} x{len(c['parts'])}", 2.8, xs + 7,
                         y - wall / 2 + 0.1, top, halign="left")
        notes.append(f"channel {c['ch']}: {c['name'].lower()} {c['L']:.1f} mm ×{len(c['parts'])} ({how})")
        y += w + wall
    body = _text_cut(body, f"CG{idx}", 5.5, xs + 7, 4.5, top, halign="left")
    msg = "SAW ON THE LEFT FACE OF THE SLOT" if width > 45 else "SAW LEFT FACE"
    body = _text_cut(body, msg, 2.2, xs - 3.2, width / 2, tower_h, plane=((0, 1, 0), (0, 0, 1)))
    return JigPart(f"cutting_guide_{idx}", body.val(), 1, "; ".join(notes), label=f"CG{idx}", engraved=True)


def scarf_guide(slope: float = 3.0) -> JigPart:
    """Wedge with two open grooves sloping at 1-in-`slope`. A ring piece lies in a groove with its end just
    past the toe; rub the wedge on the sandpaper board until the plastic toe starts to scuff."""
    tan = 1 / slope
    Lx, wall, depth = 45.0, 6.0, 2.6
    W = 2 * SLOT + 3 * wall
    prof = [(0, 0), (Lx, 0), (Lx, Lx * tan + depth), (0, depth)]
    body = cq.Workplane("XZ").polyline(prof).close().extrude(-W)        # XZ normal is -y: extrude to +y
    for yg in (wall, 2 * wall + SLOT):
        groove = [(-1, -1 * tan), (Lx + 1, (Lx + 1) * tan), (Lx + 1, (Lx + 1) * tan + 10), (-1, -tan + 10)]
        body = body.cut(cq.Workplane("XZ").polyline(groove).close().extrude(-SLOT).translate((0, yg, 0)))
    # each groove is named on the side face next to it, the ID on the back
    body = _text_cut(body, "O UP", 4.0, Lx * 0.62, 0, 7.0, plane="XZ")
    body = _text_cut(body, "O DOWN", 4.0, Lx * 0.62, W, 7.0, plane=((-1, 0, 0), (0, 1, 0)))
    body = _text_cut(body, "SG", 4.0, Lx, W / 2, 6.0, plane="YZ")
    return JigPart("scarf_sanding_guide", body.val(), 1, label="SG", engraved=True, note=
                   "Lay a ring piece in a groove with its end 1 mm past the thin front edge, press it down with a "
                   "finger, and rub the wedge on the sandpaper board until the plastic either side of the piece "
                   "starts to scuff: the end is then a clean 1-in-3 scarf (about 9 mm long). Groove 'O UP' for "
                   "end 1, groove 'O DOWN' for end 2.")


def end_angle_groups(st: Structure, exclude_level: bool = False) -> dict[int, list[int]]:
    """Distinct angled (non-scarf, non-square) end cuts: degrees off square -> parts that need them."""
    out: dict[int, list[int]] = {}
    level = set(level_parts(st)) if exclude_level else set()
    for pi, p in enumerate(st.parts):
        if p.sec is None or pi in level:
            continue
        for a, style in zip(p.end_angles, p.info.get("end_style", ["", ""])):
            off = int(round(90 - a))
            if style in ("level", "lap", "mitre") and off >= 1:
                out.setdefault(off, [])
                if pi not in out[off]:
                    out[off].append(pi)
    return dict(sorted(out.items()))


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def level_poses(st: Structure) -> list[dict]:
    """Every way a level-cut strip end sits in the strip-end sanding jig: for each strip set and each end,
    the stick's axis and side-face normal, turned so the end is flat on the table and the outer (letter)
    face looks towards the front of the jig (-y). This makes the exact compound angle, lean and inward
    tilt together."""
    reps: dict[str, int] = {}
    for pi in level_parts(st):
        reps.setdefault(st.parts[pi].group or st.parts[pi].name, pi)
    out = []
    for g, pi in sorted(reps.items()):
        _, u, y, n = strip_frame(st, pi)
        for end in ("foot", "top"):
            flip = np.eye(3) if end == "foot" else np.diag([1.0, -1.0, -1.0])
            u1, y1, n1 = flip @ u, flip @ y, flip @ n
            R = _rz(-math.pi / 2 - math.atan2(n1[1], n1[0]))
            out.append(dict(group=g, letter=g.split()[-1][:2].upper(), end=end, u=R @ u1, y=R @ y1, pi=pi))
    return out


def _level_sanding_jig(st: Structure, n: int) -> JigPart | None:
    poses = level_poses(st)
    if not poses:
        return None
    H = 22.0
    letters = sorted({p["letter"] for p in poses})
    rows = {L: (-9.0 if k == 0 else 9.0) for k, L in enumerate(letters)}
    body = cq.Workplane().box(76, 44, H, centered=(True, True, False))
    sec = st.parts[poses[0]["pi"]].sec
    for p in poses:
        x = -15.0 if p["end"] == "foot" else 15.0
        c = np.array([x, rows[p["letter"]], 0.0])
        tun = member_solid(c - p["u"] * 60, c + p["u"] * 60, sec, p["y"], grow=(SLOT - sec.w) / 2)
        body = body.cut(cq.Workplane().add(tun))
        # label next to where the stick comes out of the top
        u = p["u"] if p["u"][2] > 0 else -p["u"]
        ex = c + u * (H / u[2])
        ly = 17.0 if rows[p["letter"]] > 0 else -17.0
        body = _text_cut(body, f"{p['letter']} {p['end'].upper()}", 3.2, float(ex[0]), ly, H)
    off = 90 - st.parts[poses[0]["pi"]].end_angles[0]
    body = _text_cut(body, f"SJ{n}  LETTER FACE THIS SIDE", 3.2, 0, -22, H / 2, plane="XZ")
    return JigPart(f"end_sanding_jig_{int(round(off))}deg", body.val(), 1,
                   f"Strip ends, {off:.1f}° off square in total (the lean and the inward tilt together). Four slots: "
                   f"{', '.join(p['letter'] + ' ' + p['end'] for p in poses)}. Letter face to the front.",
                   label=f"SJ{n}", engraved=True)


def sanding_jigs(st: Structure) -> list[JigPart]:
    """Strip ends: one block with a slot for every strip set and end, at the exact compound angle.
    Other angled ends (other tower types): one block per angle with two simple tilted slots."""
    out = []
    lj = _level_sanding_jig(st, 1)
    if lj:
        out.append(lj)
    groups = end_angle_groups(st, exclude_level=True)
    if groups:
        # one block for every other angled end (e.g. ring ends): a pair of slots per angle, leaning both ways
        n = len(out) + 1
        offs = list(groups)
        pitch = 10.0
        width = pitch * len(offs) + 8
        body = cq.Workplane().box(width, 34, 22, centered=(True, True, False))
        for k, off in enumerate(offs):
            x = -width / 2 + 4 + pitch * (k + 0.5)
            for y, sgn in ((-7, 1), (7, -1)):     # leaning one way, then the other
                chan = (cq.Workplane().box(SLOT, SLOT, 90).rotate((0, 0, 0), (0, 1, 0), sgn * off)
                        .translate((x, y, 11)))
                body = body.cut(chan)
            body = _text_cut(body, f"{off}", 3.2, x, 13.5, 22)
            body = _text_cut(body, f"{off}", 3.2, x, -13.5, 22)
        body = _text_cut(body, f"SJ{n}  END ANGLE IN DEGREES OFF SQUARE", 2.8, 0, -17, 11, plane="XZ")
        names = sorted({st.parts[i].group for pis in groups.values() for i in pis})
        out.append(JigPart("end_sanding_jig_rings", body.val(), 1,
                           f"Sands the angled ends of {', '.join(names)}. Each angle (degrees off square) has two "
                           f"slots, one leaning each way: use the one that matches the cut end. Push the piece down "
                           f"until it touches the sandpaper, hold it still and rub until the plastic scuffs.",
                           label=f"SJ{n}", engraved=True))
    return out


def crossing_marks(st: Structure, pi: int) -> list[tuple[float, str]]:
    """Distances along a level-cut strip from its foot (centre line) to each crossing level, labelled with the
    level number and an R where a ring goes."""
    a, u, _, _ = strip_frame(st, pi)
    ring_lv = ring_levels(st)
    out = []
    for k, z in enumerate(st.levels[1:-1], start=1):
        out.append(((z - a[2]) / u[2], f"{k}R" if k in ring_lv else str(k)))
    return out


def ring_levels(st: Structure) -> dict[int, str]:
    """Crossing level number -> the ring group at that level."""
    out = {}
    for g in sorted({p.group for p in st.parts if p.kind == "ring" and p.group}):
        zs = [st.nodes[st.joints[j].node][2] for j, jt in enumerate(st.joints)
              if st.parts[jt.parts[0]].group == g and jt.kind != "scarf"]
        if zs:
            z = float(np.mean(zs))
            k = int(np.argmin([abs(z - lv) for lv in st.levels[1:-1]])) + 1
            out[k] = g
    return out


def marking_gauge(st: Structure) -> JigPart | None:
    """Strip-length template: lay a strip in its channel, letter face up, foot against the angled stop, and
    pencil a tick through each notch: where every crossing (and ring) goes. The END notch checks the length."""
    reps: dict[str, int] = {}
    for pi in level_parts(st):
        reps.setdefault(st.parts[pi].group or st.parts[pi].name, pi)
    if not reps:
        return None
    w, wall, margin, top, base, x0 = SLOT, 9.0, 8.0, 7.0, 3.0, 14.0
    L = max(st.parts[pi].length for pi in reps.values())
    length = x0 + L + 9
    width = 2 * margin + len(reps) * w + (len(reps) - 1) * wall
    body = cq.Workplane().box(length, width, top, centered=(False, False, False))
    y = margin
    for g, pi in sorted(reps.items()):
        yc = y + w / 2
        t = math.tan(math.radians(level_angle(st, pi)))
        xl = lambda yy: x0 - t * (yy - yc)
        chan = (cq.Workplane().polyline([(xl(y), y), (length + 1, y), (length + 1, y + w), (xl(y + w), y + w)])
                .close().extrude(10).translate((0, 0, base)))
        body = body.cut(chan)
        marks = crossing_marks(st, pi) + [(st.parts[pi].length, "END")]
        from .tolerances import CUT_TOL
        for s in (st.parts[pi].length - CUT_TOL, st.parts[pi].length + CUT_TOL):   # length tolerance band
            body = body.cut(cq.Workplane().box(0.5, w + 1.6, 10, centered=(True, True, False))
                            .translate((x0 + s, yc, base + 2.5)))
        for s, lab in marks:
            body = body.cut(cq.Workplane().box(0.8, w + 3.0, 10, centered=(True, True, False))
                            .translate((x0 + s, yc, base + 1.0)))
            body = _text_cut(body, lab, 3.0, x0 + s, y + w + 3.2, top)
        body = _text_cut(body, g.split()[-1].upper(), 5.0, 6.0, yc, top)
        y += w + wall
    body = _text_cut(body, "MG1  FOOT AT THE STOP, LETTER UP", 3.0, x0 + 30, margin / 2 - 0.2, top, halign="left")
    return JigPart("marking_gauge", body.val(), 1,
                   "Lay a cut strip in its channel (A or B), letter face up, foot against the angled stop. Pencil a "
                   "tick through every notch: the numbers are the crossing levels, R = a ring goes there. The strip "
                   "should end at the END notch.", label="MG1", engraved=True)


def fit_coupon() -> JigPart:
    """Print this first: which slot does a real stick slide into with light friction?"""
    sizes = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6]
    pitch, x0 = 9.0, 13.0
    body = cq.Workplane().box(x0 + pitch * len(sizes), 22, 6, centered=(False, False, False))
    for k, s in enumerate(sizes):
        x = x0 + pitch * k + pitch / 2
        body = body.cut(cq.Workplane().box(s, 15, 4, centered=(True, False, False)).translate((x, 8, 2)))
        body = _text_cut(body, f"{s:.1f}", 2.6, x, 4, 6)
    body = _text_cut(body, "FT", 4.0, 6, 11, 6)
    return JigPart("fit_test_coupon", body.val(), 1,
                   "Print first. Try a stick in each slot: pick the smallest one it slides into with light "
                   "friction and set Jig slot size in the app to that number (default 3.3).",
                   label="FT", engraved=True)


def engrave_label(jp: JigPart) -> cq.Shape:
    """Cut the part's ID into a visible face (top face of each part)."""
    if not jp.label or jp.engraved or jp.name.startswith("spacer"):
        return jp.shape
    try:
        wp = cq.Workplane().add(jp.shape)
        bb = jp.shape.BoundingBox()
        if jp.name.startswith("collar"):
            top = max(jp.shape.Faces(), key=lambda f: f.Center().z)
            c = top.Center()
            text = jp.label               # "C3-4"
            size = 3.0
        elif jp.name == "mast":
            size = 6.0
            return wp.cut(cq.Workplane("XY", origin=(-1, -1, bb.zmax)).text(jp.label, size, -0.6)).val()
        else:
            top = max(jp.shape.Faces(), key=lambda f: (round(f.Center().z, 2), f.Area()))
            c = top.Center()
            text = jp.label
            size = 5.0
            if jp.name.startswith("base_plate"):
                size = 6.0
                c = cq.Vector(bb.xmin + 14, bb.ymax - 8, bb.zmax)
                ztop = max(f.Center().z for f in jp.shape.Faces() if abs(f.Center().x - c.x) < 30
                           and abs(f.Center().y - c.y) < 30 and f.normalAt().z > 0.9)
                c = cq.Vector(c.x, c.y, min(ztop, bb.zmax))
            elif jp.name.startswith("end_sanding"):
                c = cq.Vector(bb.xmax - 12, bb.ymin + 5, bb.zmax)
        tool = cq.Workplane("XY", origin=(c.x, c.y, c.z)).text(text, size, -0.5)
        out = wp.cut(tool).val()
        return out if out.isValid() and out.Volume() < jp.shape.Volume() else jp.shape
    except Exception:
        return jp.shape


def export_shape(shape: cq.Shape, step_path: str, stl_path: str | None = None):
    cq.exporters.export(cq.Workplane().add(shape), step_path)
    if stl_path:
        cq.exporters.export(cq.Workplane().add(shape), stl_path, tolerance=0.05, angularTolerance=0.2)
