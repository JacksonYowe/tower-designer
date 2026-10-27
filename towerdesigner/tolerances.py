"""Build tolerances: how accurate each cut and placement has to be, how the jig allows for it, and a check
that the parts of the 3D model actually fit together (rings on strips, laps, crossings, jig, pole)."""
from __future__ import annotations

import math

import numpy as np

from . import cad
from .structure import Structure, bore_clearance

STICK_VARIATION = 0.15   # mm, typical size variation of supplied 3 x 3 sticks (either way)
PRINT_TOL = 0.15         # mm, typical error of a Bambu print on slots and holes
CUT_TOL = 0.3            # mm, length error cutting against a stop with a razor saw
SAND_ALLOW = 0.15        # mm, extra length left on each strip end for the sanding jig to take off
MARK_TOL = 0.5           # mm, pencil tick through a marking-gauge notch
MIN_HEIGHT_MARGIN = 0.5  # mm, the tower must still be over the minimum height at the worst tolerance
GLUE_GAP = 0.1           # mm, a gap no thicker than a glue film still counts as touching
MIN_CONTACT = 0.6        # at least this share of every glue face must touch
PEG_CLOSE = 0.4          # mm, a gap a peg closes by bending a long strip slightly
PRESS_BAND = 0.2         # mm, what a peg or finger pressure squeezes flat (balsa crushes this much)


def rows(st: Structure, spec, jig_parts) -> list[dict]:
    """One row per thing you cut or place: nominal size, tolerance (mm or degrees), how the jig allows for it,
    and how to check it."""
    from .families import RING_END_ALLOWANCE
    out = []
    slot = cad.SLOT
    side = (slot - 3.0) / 2
    out.append(dict(item="Stick size", nominal="3.0 × 3.0 mm", tol=f"±{STICK_VARIATION:.2f} mm (typical)",
                    jig=f"Every jig slot is {slot:.1f} mm ({side:.2f} mm clear each side). Set it from the fit-test "
                        "coupon FT, which also cancels out your printer's error.",
                    check="FT: the stick slides in with light friction."))
    out.append(dict(item="Printed jig slots and holes", nominal="as designed", tol=f"±{PRINT_TOL:.2f} mm",
                    jig=f"Collar notches and base pockets have {cad.FIT:.2f} mm clearance each side; the mast hole "
                        f"{cad.MAST_CLEAR / 2:.1f} mm each side; the mast socket 0.1 mm each side.",
                    check="Step 1 dry stack: every strip drops into its notch without forcing."))
    level = cad.level_parts(st)
    if level:
        L = st.parts[level[0]].length
        tilt = 90 - st.parts[level[0]].end_angles[0]
        tunnel = 22.0 / math.cos(math.radians(tilt))
        play = math.degrees(math.atan(2 * side / tunnel))
        out.append(dict(item="Strip length (cut)", nominal=f"{L + 2 * SAND_ALLOW:.1f} mm",
                        tol=f"±{CUT_TOL:.1f} mm",
                        jig=f"The cutting-guide stop is set {2 * SAND_ALLOW:.1f} mm long so the sanding jig can clean "
                            f"both ends.",
                        check="No measuring: the stop sets it."))
        out.append(dict(item="Strip length (after sanding)", nominal=f"{L:.1f} mm", tol=f"±{CUT_TOL:.1f} mm",
                        jig="Marking gauge MG1 has an END notch with a small tick 0.3 mm either side.",
                        check="The strip's top end lands between the two small ticks."))
        out.append(dict(item="Strip end angle", nominal=f"level cut, {tilt:.1f}° off square in total (lean + inward tilt)",
                        tol=f"±{play:.1f}° (≤ {3 * math.tan(math.radians(play)):.2f} mm across the end)",
                        jig="Sanding jig SJ1 holds the strip at the exact compound angle; the slot clearance sets "
                            "the tolerance.",
                        check="The strip stands on the table at its lean without rocking."))
        tw = cad.strip_twist(st, level[0])
        out.append(dict(item="Strip twist", nominal=f"{tw:.0f}° from foot to top", tol="±3°",
                        jig="Every collar notch is angled to the twist at its height, with "
                            f"{cad.FIT:.2f} mm clearance; damp strips take the twist without strain.",
                        check="Each strip lies flat in every notch, no edge lifting."))
        out.append(dict(item="Crossing gap before pegging", nominal="0 mm", tol=f"up to {PEG_CLOSE:.1f} mm",
                        jig="A straight strip can't sit exactly on every curved crossing; the 3D check shows gaps of "
                            "up to about 0.3 mm. A peg bends the strip that little to close it.",
                        check="With the peg on, no light between the faces (gap under 0.1 mm)."))
        out.append(dict(item="Crossing marks", nominal="at each level", tol=f"±{MARK_TOL:.1f} mm",
                        jig="MG1 notches are 0.8 mm wide: put the pencil in the middle.",
                        check="A and B ticks line up at every crossing in the dry fit."))
    ring = [p for p in st.parts if p.info.get("offsets")]
    if ring:
        w = ring[0].sec.size_y
        out.append(dict(item="Ring piece length", nominal="see cut list", tol=f"±{2 * RING_END_ALLOWANCE:.1f} mm "
                        f"(±{RING_END_ALLOWANCE:.1f} each end)",
                        jig=f"Each piece is cut {2 * RING_END_ALLOWANCE:.1f} mm longer than the crossing needs, so "
                            "even a piece cut short still covers its lap fully. Square cuts in the cutting guide.",
                        check="Each end runs past the strip it crosses by at least 2 mm."))
        out.append(dict(item="Ring height", nominal="at a crossing level", tol=f"±{MARK_TOL:.1f} mm",
                        jig="The crossing (pencil tick) sets it; no measuring.",
                        check="The ring looks level against the ticks all round."))
    lap_ring = [p for p in st.parts if p.kind in ("ring", "base", "cap") and "lap" in p.info.get("end_style", [])]
    if lap_ring:
        out.append(dict(item="Ring piece length", nominal="see cut list (strip centre to strip centre)",
                        tol=f"±{CUT_TOL:.1f} mm",
                        jig="The cutting guide cuts both ends at their angle against a stop, so neighbouring pieces "
                            "meet on the middle of each strip. A short piece leaves at most "
                            f"{2 * CUT_TOL:.1f} mm between the two ends; each end is still glued over its half of "
                            "the strip. A long piece won't sit: sand its end lightly in the ring-end sanding jig.",
                        check="Both ends sit on the middle of their strips; the gap between neighbouring ends is "
                              f"no more than {2 * CUT_TOL:.1f} mm."))
        out.append(dict(item="Ring end angle", nominal="see cut list", tol="±1°",
                        jig="The cutting-guide slot sets the angle; the ring-end sanding jig cleans it up.",
                        check="Two neighbouring ends meet with no light between them."))
        out.append(dict(item="Ring height", nominal="at a crossing level", tol=f"±{MARK_TOL:.1f} mm",
                        jig="The crossing (pencil tick) sets it; no measuring.",
                        check="The ring looks level against the ticks all round."))
    H = st.height
    out.append(dict(item="Tower height", nominal=f"{H:.1f} mm", tol=f"−{CUT_TOL + 0.2:.1f} / +{CUT_TOL:.1f} mm",
                    jig=f"Strips are cut to full height; the top is only lightly sanded flat. Worst case is "
                        f"{H - CUT_TOL - 0.2:.1f} mm, still over the {spec.min_height:.0f} mm minimum.",
                    check="Measure at three places round the top."))
    gap = min(bore_clearance(st.nodes[m.i], st.nodes[m.j], m.sec.half_extent) for m in st.members) - spec.pole_d / 2
    out.append(dict(item="Gap to the pole", nominal=f"{gap:.1f} mm all round", tol=f"−{spec.sway:.1f} mm "
                    "(if the tower is built leaning)",
                    jig="The collars hold every strip at its radius.",
                    check=f"The pole slides through without touching; at least {max(0.0, gap - spec.sway):.1f} mm "
                          "even with a lean."))
    out.append(dict(item="Base ring in its groove", nominal=f"{cad.FIT:.2f} mm clear all round", tol="±0.15 mm (print)",
                    jig=f"The groove is the ring's own shape grown by {cad.FIT:.2f} mm, so the ring never touches the "
                        "plastic; the strips hold it.",
                    check="The ring isn't touching the base plate anywhere."))
    return out


def cut_tolerance(st: Structure, pis: list[int]) -> str:
    """Short tolerance text for one cut-list row."""
    from .families import RING_END_ALLOWANCE
    p = st.parts[pis[0]]
    es = p.info.get("end_style", [])
    if "level" in es:
        return f"cut {p.length + 2 * SAND_ALLOW:.1f}, sand to {p.length:.1f} ±{CUT_TOL:.1f}"
    if p.info.get("offsets"):
        return f"±{RING_END_ALLOWANCE:.1f} each end"
    return f"±{CUT_TOL:.1f} (end gap ≤ {2 * CUT_TOL:.1f})"


def face_angle(st: Structure, a: int, b: int, node: int) -> float:
    """Angle (degrees) between two crossing strips' facing faces at a node, with both twisted to the surface."""
    z = st.nodes[node][2]
    fs = []
    for i in (a, b):
        q0, q1, _ = cad.strip_line(st, i)
        u = (q1 - q0) / np.linalg.norm(q1 - q0)
        p = q0 + (z - q0[2]) / u[2] * u
        n = cad.surface_normal(st, p)
        f = n - (n @ u) * u
        fs.append(f / np.linalg.norm(f))
    return float(np.degrees(np.arccos(np.clip(fs[0] @ fs[1], -1, 1))))


def fit_report(st: Structure, jig_parts=None) -> list[tuple[str, str, bool]]:
    """Check the 3D model the way a builder would: for every glued pair, how much of the glue face really touches
    (within a glue film of GLUE_GAP), how deep any overlap is, and whether the jig clears the tower.
    Returns (check, result, ok)."""
    sol, grown = {}, {}

    def solid(i):
        if i not in sol:
            sol[i] = cad.part_solids(st, i)[0]
        return sol[i]

    def thick(i):
        if i not in grown:
            grown[i] = cad.part_solids(st, i, grow=GLUE_GAP)[0]
        return grown[i]

    def vol(a, b):
        if not cad._bb_overlap(a.BoundingBox(), b.BoundingBox(), 0.2):
            return 0.0
        try:
            return a.intersect(b).Volume()
        except Exception:
            return 0.0

    groups: dict[str, list] = {}
    for jt in st.joints:
        a, b = jt.parts[0], jt.parts[1]
        pa, pb = st.parts[a], st.parts[b]
        if pa.sec is None or pb.sec is None:
            continue
        if jt.kind == "lap" and pa.kind == "diag" and pb.kind == "diag":
            key = "Crossings, feet and tops: B strip glued to A strip"
        elif jt.kind == "lap":
            key = "Ring pieces glued to the next piece"
        elif jt.kind in ("lap end", "scarf end"):
            b = cad.outer_strip_at(st, jt.node, b)
            key = "Ring ends glued to their strip"
        else:
            continue
        ov = vol(solid(a), solid(b))
        gap = 0.0
        if ov < 1e-3:
            from OCP.BRepExtrema import BRepExtrema_DistShapeShape
            d = BRepExtrema_DistShapeShape(solid(a).wrapped, solid(b).wrapped)
            d.Perform()
            gap = d.Value()
        # once pegged (the strip bends to close any gap), how much of the face is within PRESS_BAND?
        g0 = cad.part_solids(st, a, grow=gap)[0] if gap > 1e-3 else solid(a)
        g1 = cad.part_solids(st, a, grow=gap + PRESS_BAND)[0]
        touch = max(0.0, vol(g1, solid(b)) - vol(g0, solid(b))) / PRESS_BAND
        nominal = max(jt.area, 1e-6)
        angle = face_angle(st, a, b, jt.node) if key.startswith("Crossings") else 0.0
        groups.setdefault(key, []).append((touch / nominal, ov / nominal, gap, angle))
    out = []
    for key, vals in groups.items():
        worst = min(v[0] for v in vals)
        mean = sum(v[0] for v in vals) / len(vals)
        depth = max(v[1] for v in vals)
        gmax = max(v[2] for v in vals)
        amax = max(v[3] for v in vals)
        ok = worst >= MIN_CONTACT and depth <= 0.5 and gmax <= PEG_CLOSE and amax <= 2.0
        txt = (f"{len(vals)} joints: glue face touching {100 * min(mean, 1):.0f} % on average, worst "
               f"{100 * min(worst, 1):.0f} % (within {PRESS_BAND:.1f} mm, pressed); deepest overlap {depth:.2f} mm")
        if amax > 0:
            txt += f"; faces within {amax:.1f}° of parallel"
        if gmax > GLUE_GAP / 2:
            txt += f"; largest gap before pegging {gmax:.2f} mm (the strip bends to close it)"
        out.append((key, txt, ok))
    # half-lap rings: neighbouring pieces meet end to end on each strip
    ends: dict[int, list[int]] = {}
    for i, p in enumerate(st.parts):
        if p.kind in ("ring", "base", "cap") and "lap" in p.info.get("end_style", []) and p.sec is not None:
            m = st.members[p.members[0]]
            for nd in (m.i, m.j):
                ends.setdefault(nd, []).append(i)
    pairs = [ps for ps in ends.values() if len(ps) == 2]
    if pairs:
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        gaps = []
        for a, b in pairs:
            d = BRepExtrema_DistShapeShape(solid(a).wrapped, solid(b).wrapped)
            d.Perform()
            gaps.append((d.Value(), vol(solid(a), solid(b))))
        g, v = max(x[0] for x in gaps), max(x[1] for x in gaps)
        out.append(("Neighbouring ring pieces meet on each strip", f"{len(gaps)} joints: largest gap {g:.2f} mm, "
                    f"largest overlap {v:.2f} mm³", g <= GLUE_GAP and v <= 1.0))
    if jig_parts is not None:
        from .stepmodels import jig_layout
        lay = jig_layout(st, jig_parts)
        tower = [solid(i) for i, p in enumerate(st.parts) if p.sec is not None]
        clash = 0
        for name, js in lay.items():
            jb = js.BoundingBox()
            for t in tower:
                if cad._bb_overlap(jb, t.BoundingBox(), 1.0) and js.intersect(t).Volume() > 0.01:
                    clash += 1
        out.append(("Jig never overlaps the tower", f"{clash} overlaps", clash == 0))
    return out
