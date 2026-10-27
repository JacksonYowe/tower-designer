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
        out.append(dict(item="Strip end angle", nominal=f"level cut (~{tilt:.0f}° plus the inward tilt)",
                        tol=f"±{play:.1f}° (≤ {3 * math.tan(math.radians(play)):.2f} mm across the end)",
                        jig="Sanding jig SJ1 holds the strip at the exact compound angle; the slot clearance sets "
                            "the tolerance.",
                        check="The strip stands on the table at its lean without rocking."))
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
    out.append(dict(item="Base ring in its groove", nominal="0.5 mm above the plate", tol="±0.2 mm",
                    jig="The groove is cut 0.5 mm deeper than the ring, so it never touches the plastic.",
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


def fit_report(st: Structure, jig_parts=None) -> list[tuple[str, str, bool]]:
    """Check the 3D model: does every glued pair touch, does nothing overlap that shouldn't, does the jig clear
    the tower? Returns (check, result, ok)."""
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    def dist(a, b):
        d = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped)
        d.Perform()
        return d.Value()

    def overlap_depth(a, b):
        """Rough interference depth (mm): overlap volume ÷ the smaller face it could spread over."""
        if not cad._bb_overlap(a.BoundingBox(), b.BoundingBox(), 0):
            return 0.0
        try:
            v = a.intersect(b).Volume()
        except Exception:
            return 0.0
        return v / 9.0

    sol = {i: cad.part_solids(st, i) for i, p in enumerate(st.parts) if p.sec is not None}
    one = {i: s[0] for i, s in sol.items() if s}
    out = []
    groups = {}
    for j, jt in enumerate(st.joints):
        a, b = jt.parts[0], jt.parts[1]
        if a not in one or b not in one:
            continue
        pa, pb = st.parts[a], st.parts[b]
        if jt.kind == "lap" and pa.kind == "diag" and pb.kind == "diag":
            key = "Crossings: B strip touches A strip"
        elif jt.kind == "lap":
            key = "Ring laps: each piece touches the next"
        elif jt.kind in ("lap end", "scarf end"):
            # the ring end is really glued to the outer (B) strip at that crossing
            b = cad.outer_strip_at(st, jt.node, b)
            key = "Ring ends: piece touches its strip"
        else:
            continue
        g = dist(one[a], one[b])
        dpt = overlap_depth(one[a], one[b])
        groups.setdefault(key, []).append((g, dpt))
    # half-lap rings: neighbouring pieces meet end to end on each strip
    ends: dict[int, list[int]] = {}
    for i, p in enumerate(st.parts):
        if p.kind in ("ring", "base", "cap") and "lap" in p.info.get("end_style", []) and i in one:
            m = st.members[p.members[0]]
            for nd in (m.i, m.j):
                ends.setdefault(nd, []).append(i)
    for nd, ps in ends.items():
        if len(ps) == 2:
            a, b = ps
            groups.setdefault("Ring pieces meet end to end on each strip", []).append(
                (dist(one[a], one[b]), overlap_depth(one[a], one[b])))
    for key, vals in groups.items():
        worst_gap = max(v[0] for v in vals)
        worst_in = max(v[1] for v in vals)
        ok = worst_gap <= GLUE_GAP and worst_in <= 0.5
        out.append((key, f"{len(vals)} joints: largest gap {worst_gap:.2f} mm, deepest overlap {worst_in:.2f} mm", ok))
    if jig_parts is not None:
        from .stepmodels import jig_layout
        lay = jig_layout(st, jig_parts)
        clash, near = 0, 99.0
        for name, js in lay.items():
            jb = js.BoundingBox()
            for t in one.values():
                if not cad._bb_overlap(jb, t.BoundingBox(), 1.0):
                    continue
                if js.intersect(t).Volume() > 0.01:
                    clash += 1
        out.append(("Jig never overlaps the tower", f"{clash} overlaps", clash == 0))
    return out
