"""Build steps: which pieces and which jig files each step uses, plus a 3D model and picture of each.

Every step gets  steps/step_NN_<name>.step  (open in Onshape: jig in blue, pieces already built in
brown, pieces added in this step in orange) and a PNG used in the assembly guide.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

import cadquery as cq

from . import cad
from .structure import Structure



@dataclass
class Step:
    key: str
    title: str
    parts: list[int]                  # tower pieces added in this step
    jig: list[str]                    # jig files (without extension) on the bench in this step
    actions: list[str]                # numbered instructions
    checks: list[str] = field(default_factory=list)
    mistakes: list[str] = field(default_factory=list)
    model: str = ""                   # STEP file written for this step (relative to the export folder)
    new_joints: list = field(default_factory=list)
    picture: str = ""                 # PNG of the step


def jig_layout(st: Structure, jig_parts) -> dict[str, cq.Shape]:
    """Every jig file placed where it sits during the build (name -> shape in tower coordinates)."""
    by = {jp.name: jp for jp in jig_parts}
    placed = {}
    for name in by:
        if name.startswith("base_plate"):
            placed[name] = by[name].shape
    if "mast" in by:
        placed["mast"] = by["mast"].shape.translate(cq.Vector(0, 0, cad.POCKET - cad.PLATE_T))
    _, stack = cad.jig_stack(st)
    for item in stack:
        if item["kind"] == "spacer":
            nm = f"spacer_{item['k']}"
            if nm in by:
                placed[nm] = by[nm].shape.translate(cq.Vector(0, 0, item["z0"]))
        else:
            for name, jp in by.items():
                if name.startswith(f"collar_{item['k']}_seg"):
                    placed[name] = jp.shape.translate(cq.Vector(0, 0, item["zc"] - cad.COLLAR_T / 2))
    return placed


def jig_files_in_use(layout: dict) -> list[str]:
    return sorted({n.split("#")[0] for n in layout}, key=_natural)


def _natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def lean_side(st: Structure, group: str) -> str:
    """Which way a strip set leans going up, seen from outside the tower: 'right' or 'left'."""
    for pi in cad.level_parts(st):
        if st.parts[pi].group == group:
            a, u, _, _ = cad.strip_frame(st, pi)
            t = np.cross([0, 0, 1.0], cad._radial(a))       # anticlockwise seen from above = to your right
            return "right" if u @ t > 0 else "left"
    return "right"


def ring_height(st: Structure, group: str) -> tuple[int, float] | None:
    """(crossing level, height above the table) of a ring group."""
    for k, g in cad.ring_levels(st).items():
        if g == group:
            return k, float(st.levels[k])
    return None


def plan_steps(st: Structure, jig_parts) -> list[Step]:
    layout = jig_layout(st, jig_parts)
    jig_all = jig_files_in_use(layout)
    by = defaultdict(list)
    for i, p in enumerate(st.parts):
        by[p.group].append(i)
    from .lasercut import part_ids
    ids = part_ids(st)
    names = {jp.name: jp.label for jp in jig_parts}
    n_col = len(cad.collar_heights(st))
    mark = "A1 B1" if st.family.startswith("diagrid") else "S1"
    steps = [Step("jig", "Set up the jig", [], jig_all, [
        "Lay a sheet of baking paper on a flat bench and tape the base plate <b>BP</b> down on it, pockets facing up.",
        "Push the mast <b>M</b> into the socket in the middle of the boss. One corner of the mast is cut off, and so "
        "is one corner of the socket: line them up and it goes in only one way. Push it down until it stops. If it "
        "wobbles, wrap one layer of tape round its foot and push it in again.",
        f"Build the stack on the mast from the bottom up, using the stack table in section 8: <b>SP1</b>, then the "
        f"segments of collar <b>C1</b>, then <b>SP2</b>, then <b>C2</b>, and so on up to <b>C{n_col}</b>. Each spacer "
        "has the same cut corner in its hole, so it only goes on one way, with its disc on top.",
        "Before you cover a spacer's disc, read the numbers printed on it: each number is where that collar segment "
        "goes. Put segment <b>C1-1</b> on the <b>1</b>, <b>C1-2</b> on the <b>2</b>, and so on. Engraved ID on top, "
        "notches facing outwards, inner end touching the mast. The numbers run anticlockwise (seen from above) "
        f"from the <b>{mark}</b> mark on the base plate.",
        "Put packing tape over the base-plate pockets and the base-ring groove, so glue can't bond the tower to the "
        "plate. (No glue goes near the collars, so they don't need tape.)",
        f"Cut {2 * len(by.get('Strip A', [])) or 40} tabs of masking tape, about 6 mm × 30 mm, and stick them along "
        "the edge of the bench ready for Step 2.",
    ], checks=["The mast is upright: sight it against a door frame from two sides.",
               "Every collar is complete and flat, and each segment sits on its own number.",
               "Every segment's ID is on top, and the IDs read C1-…, C2-… from the bottom up."],
        mistakes=["Stacking the spacers in the wrong order puts the collars at the wrong heights, so the strips won't "
                  "follow their notches. The disc on each spacer says which collar goes on it (\"under C3\")."])]
    scarf = st.params.get("ring_joint") == "scarf"
    meet = ("Neighbouring pieces overlap on their 9 mm scarf, centred on a crossing. Hold each piece with its "
            "<b>O</b> face outwards and its <b>end 2</b> (the end where you can see the sloping scarf face) to the "
            "right. Work to the right: each new piece's end 1 lies on top of the last piece's end 2. Glue the strip "
            "patch and the sloping faces." if scarf
            else "Neighbouring pieces meet end to end on a B strip, each covering half its width.")
    close = (" Leave the <b>first</b> piece's end 1 unglued until the end; then slide the last piece's end 2 "
             "underneath it and glue both together." if scarf else "")
    if st.family.startswith("diagrid"):
        def rng(group):
            ps = sorted(by.get(group, []), key=lambda i: _natural(ids[i]))
            return f"{ids[ps[0]]}–{ids[ps[-1]]}" if ps else ""
        n_str = len(by.get("Strip A", []))
        la, lb = lean_side(st, "Strip A"), lean_side(st, "Strip B")
        rings = sorted((g for g in by if g.startswith("Ring")), key=lambda g: int(g.split()[1]))
        mg = names.get("marking_gauge", "MG1")
        tabs = ", ".join(f"C{k}" for k in sorted({2, max(2, n_col - 1)}))
        steps += [
            Step("strips_a", "Inner strips (set A)", by.get("Strip A", []), jig_all, [
                f"Take the A strips ({rng('Strip A')}). Each has a letter <b>A</b> on its outer face and a foot end "
                f"marked <b>F</b> (Step 0).",
                f"Seen from outside, the A strips lean to the <b>{la}</b> as they go up (the B strips lean the other "
                "way).",
                f"Each pocket in the base plate is a V: two strips share each foot. <b>A1</b> goes in the pocket at the "
                f"<b>{mark}</b> mark, <b>A2</b> in the next pocket to the right (anticlockwise seen from above), and "
                "so on round.",
                "Stand the strip with its foot in the <b>inner</b> arm of the V, letter face outwards, lean it to the "
                f"{la}, and press it into the notch of every collar it passes, from C1 up to C{n_col}.",
                f"Hold it with a masking-tape tab at collars {tabs}: tape over the strip with the ends pressed onto "
                "the top and bottom of the collar. The tabs sit between the B strips' paths, so the B strips never lie "
                "on them.",
                "Check the angled foot sits flat in its pocket and the top end is 211 mm above the table (measure one).",
                "No glue yet. This step only sets out the inner layer.",
            ], checks=[f"All {n_str} A strips are in pockets and notches, letter faces outwards.",
                       "The strips spiral evenly: the gaps between neighbours look the same all the way up.",
                       "The pencil ticks on every strip are at the same heights all round."],
                mistakes=["A strip in the wrong pocket makes every crossing above it miss. Count the pockets from "
                          "the mark."]),
            Step("strips_b", "Outer strips (set B) and the crossings", by.get("Strip B", []), jig_all, [
                f"Take the B strips ({rng('Strip B')}). They lean to the <b>{lb}</b> going up, seen from outside, "
                "and sit in the <b>outer</b> arm of each V pocket, letter face outwards. B1 shares A1's pocket.",
                "<b>Dry fit first.</b> Stand B1 in its pocket and lay it over the A strips. Each pencil tick on B1 "
                "should sit right over the crossing with an A strip (the A strip's tick at the same height). Take it "
                "out again.",
                "<b>Glue in batches of three strips.</b> Balsa cement needs two coats. Brush a thin coat on each B "
                "strip's <b>inner</b> face (opposite the letter) at its six ticks, and on the A strips' outer face at "
                "the six crossings each B strip will touch. Wait 5–10 min until it's tacky.",
                "Add a second thin coat on the A side, stand the B strip in its pocket and lay it on. Press each "
                "crossing for 30 s, working from the bottom up. If a crossing springs apart, clip a small spring peg "
                "across it (one jaw inside the A strip, one outside the B strip) until it grabs.",
                f"Go round to the right in order: B1–B3, then B4–B6, and so on. That's {6 * n_str} crossings, "
                "about 2 hours. Split it over two sessions if you like; stop after a complete batch.",
                "Wipe off any glue that squeezes out. It adds mass and no strength.",
            ], checks=["Every crossing has glue and the faces touch fully.",
                       "B strips sit tight against A; no gaps over 0.5 mm.",
                       "All the masking-tape tabs are still on the outside of the A strips and free of glue."],
                mistakes=["Gluing a crossing before the strip is fully seated in its pocket locks in a crooked strip.",
                          "Coating only one face with balsa cement: it soaks in and the joint stays weak."]),
            Step("base", "Base ring", by.get("Base ring", []), jig_all, [
                f"Take the base-ring pieces ({rng('Base ring')}). Each one runs from one foot to the next.",
                "Lay a piece on the outside of the B strips so it rests in the shallow groove in the base plate. The "
                "groove holds it 0.5 mm above the table, so it never touches the jig.",
                meet + " Glue each end on its strip (dot colour = glue, see the picture). Start at the "
                f"<b>{mark}</b> mark." + close,
            ], checks=["The base ring sits in its groove all the way round; every piece touches both strips."]),
        ]
        for g in rings:
            lvh = ring_height(st, g)
            where = (f"crossing level {lvh[0]}, <b>{lvh[1]:.0f} mm</b> above the table, where the strips' "
                     f"<b>{lvh[0]}R</b> pencil ticks are" if lvh else "one crossing level")
            steps.append(Step(g.lower().replace(" ", "_"), f"{g}: {lvh[1]:.0f} mm up" if lvh else g, by[g], jig_all, [
                f"Take the {g} pieces ({rng(g)}). This ring goes at {where}.",
                "Each piece runs from one crossing to the next, lying on the outside of the B strips. " + meet,
                "Centre each scarf on the crossing (the tick). You don't need to measure heights: the crossings set "
                "them.",
                f"Start at the strips above the <b>{mark}</b> mark and go round to the right using the glue map and "
                "table on this page." + close,
            ], checks=["The ring is level; every piece sits flat on the crossings."]))
        steps.append(Step("cap", "Top ring", by.get("Top cap", []), jig_all, [
            f"Take the top-ring pieces ({rng('Top cap')}).",
            "Lay each piece on the outside of the B strips, with its top flush with the strip tops. " + meet +
            " Glue both ends (dot colour = glue).",
            f"Start at the strips above the <b>{mark}</b> mark and go round to the right, using the glue map and "
            "table on this page." + close,
        ], checks=["The top ring is flush with the strip ends all the way round."]))
    else:
        for s in sorted({p.step for p in st.parts}):
            ps = [i for i, p in enumerate(st.parts) if p.step == s]
            steps.append(Step(f"step{s}", f"Build step {s}", ps, jig_all,
                              ["Place the pieces into the jig and glue each shaped end as the cut list shows."]))
    steps.append(Step("remove", "Cure, remove the jig and finish", [], [], [
        "Leave everything for <b>24 hours</b> so every glue reaches full strength.",
        "Peel off every masking-tape tab.",
        "Lift the tower off the base plate, mast and all: push the feet up gently through the finger holes, a little "
        "at a time all round, until they come out of the pockets. Hold the tower by its rings, never by one strip.",
        "Hold the tower over a box and pull the mast <b>M</b> straight up out of the top.",
        "Now take the jig out <b>from the bottom up</b>, in the reverse of how you stacked it: slide <b>SP1</b> down "
        "and out; then push each <b>C1</b> segment about 3 mm inwards off its strips, tip it and lower it out of the "
        f"bottom; then <b>SP2</b>, then the <b>C2</b> segments, and so on up to <b>C{n_col}</b>. Every part comes "
        "out through the wide bottom of the tower.",
        "Never pull a segment outwards or upwards through the strips. If one sticks, push it inwards firmly or cut "
        "it with a knife: it's only a jig part.",
        "Check the tower on a flat table. If it rocks, lay the sandpaper board on the table, rest a flat book on the "
        "top ring, and slide the tower in short straight strokes (2–3 cm, no twisting) with light pressure. Check "
        "after every 5 strokes.",
        "Do the same for the top, upside down, so the load plate will bear on every strip at once.",
        "Weigh the finished tower.",
    ], checks=["It weighs no more than the limit.", "It stands upright without rocking.",
               "The pole slides through without touching."],
        mistakes=["Sanding in circles or pressing hard snaps feet. Short straight strokes, light pressure."]))
    return steps


# glue colours (also used in the guide): red = balsa cement, blue = PVA, purple = epoxy, green = superglue
GLUE_RGB = {"cement": (0.85, 0.10, 0.10), "pva": (0.10, 0.40, 0.95), "epoxy": (0.60, 0.10, 0.85),
            "ca": (0.15, 0.60, 0.25)}
RGB_JIG, RGB_DONE, RGB_NEW = (0.55, 0.72, 0.98), (0.80, 0.66, 0.48), (1.0, 0.62, 0.10)
RGB_A, RGB_B, RGB_RING = (0.86, 0.72, 0.52), (0.62, 0.45, 0.28), (0.98, 0.84, 0.35)
DOT_R = 1.6


def joint_point(st: Structure, j: int) -> np.ndarray:
    """Where the glue actually goes for joint j (between the layers that touch)."""
    jt = st.joints[j]
    pos = st.nodes[jt.node].astype(float).copy()
    parts = [st.parts[p] for p in jt.parts[:2]]
    offs = [p.offset for p in parts]
    if jt.kind == "lap":
        r_off = sum(offs) / 2                      # between the two crossing strips
    else:
        r_off = max(offs) - 1.5 if max(offs) > 0 else 0.0   # under the lapping piece
    rad = np.array([pos[0], pos[1], 0.0])
    rad /= np.linalg.norm(rad) or 1.0
    pos += rad * r_off
    pos[2] += max((p.info.get("dz", 0.0) for p in parts), key=abs)
    return pos


def glue_dots(st: Structure, joints, joint_glue) -> dict[str, list]:
    out = defaultdict(list)
    for j in joints:
        p = joint_point(st, j)
        out[joint_glue[j]].append(cq.Solid.makeSphere(DOT_R).translate(cq.Vector(*p)))
    return out


def _cq_color(rgb):
    return cq.Color(*rgb)


def _assembly(shapes_done, shapes_new, jig_shapes, dots):
    a = cq.Assembly(name="step")
    for i, s in enumerate(jig_shapes):
        a.add(s, name=f"jig_{i}", color=_cq_color(RGB_JIG))
    if shapes_done:
        a.add(cq.Compound.makeCompound(shapes_done), name="built", color=_cq_color(RGB_DONE))
    if shapes_new:
        a.add(cq.Compound.makeCompound(shapes_new), name="this_step", color=_cq_color(RGB_NEW))
    for g, ds in dots.items():
        if ds:
            a.add(cq.Compound.makeCompound(ds), name=f"glue_{g}", color=_cq_color(GLUE_RGB[g]))
    return a


def _export(shapes, path):
    if not shapes:
        return ""
    cq.exporters.export(cq.Workplane().add(cq.Compound.makeCompound(shapes)), path)
    return path


def write_step_models(st: Structure, jig_parts, steps: list[Step], outdir: str, joint_glue=None,
                      render: bool = True):
    """steps/step_NN_<key>.step (+ png) for every step, plus ring close-ups and a glue map."""
    from . import render as rnd
    sd = os.path.join(outdir, "steps")
    os.makedirs(sd, exist_ok=True)
    tmp = os.path.join(sd, ".render")
    os.makedirs(tmp, exist_ok=True)
    joint_glue = joint_glue or ["pva"] * len(st.joints)
    layout = jig_layout(st, jig_parts)
    part_shapes = {i: cad.part_solids(st, i) for i in range(len(st.parts)) if st.parts[i].sec is not None}
    done: set[int] = set()
    joints_done: set[int] = set()
    for n, stp in enumerate(steps, start=1):
        pset = set(stp.parts)
        new_j = {j for j, jt in enumerate(st.joints) if all(p in done | pset for p in jt.parts)} - joints_done
        jig_shapes = [s for k, s in layout.items() if k.split("#")[0] in stp.jig]
        done_shapes = [s for i in sorted(done) for s in part_shapes.get(i, [])]
        new_shapes = [s for i in stp.parts for s in part_shapes.get(i, [])]
        dots = glue_dots(st, sorted(joints_done | new_j), joint_glue)
        name = f"step_{n:02d}_{stp.key}"
        cad.save_assembly(_assembly(done_shapes, new_shapes, jig_shapes, dots), os.path.join(sd, name + ".step"))
        stp.model = f"steps/{name}.step"
        stp.new_joints = sorted(new_j)
        if render:
            items = [{"file": _export(jig_shapes, os.path.join(tmp, name + "_jig.step")), "rgb": RGB_JIG},
                     {"file": _export(done_shapes, os.path.join(tmp, name + "_done.step")), "rgb": RGB_DONE},
                     {"file": _export(new_shapes, os.path.join(tmp, name + "_new.step")), "rgb": RGB_NEW}]
            for g, ds in dots.items():
                items.append({"file": _export(ds, os.path.join(tmp, f"{name}_glue_{g}.step")), "rgb": GLUE_RGB[g]})
            png = os.path.join(sd, name + ".png")
            if rnd.render_spec_subprocess({"items": items, "distance": 340, "elevation": 20, "azimuth": 35,
                                           "center": [0, 0, 100]}, png):
                stp.picture = png
        done |= pset
        joints_done |= new_j
    extras = {}
    if render:
        extras = _details(st, part_shapes, layout, joint_glue, sd, tmp, rnd)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return steps, extras


def _details(st, part_shapes, layout, joint_glue, sd, tmp, rnd) -> dict:
    """Close-ups of how the rings attach, and a map of every glue point."""
    out = {}
    # glue map: whole tower, every glue dot
    all_parts = [s for i in part_shapes for s in part_shapes[i]]
    dots = glue_dots(st, range(len(st.joints)), joint_glue)
    items = [{"file": _export(all_parts, os.path.join(tmp, "map_tower.step")), "rgb": RGB_DONE}]
    for g, ds in dots.items():
        items.append({"file": _export(ds, os.path.join(tmp, f"map_{g}.step")), "rgb": GLUE_RGB[g]})
    if rnd.render_spec_subprocess({"items": items, "distance": 330, "elevation": 16, "azimuth": 35,
                                   "center": [0, 0, 105]}, os.path.join(sd, "glue_map.png")):
        out["glue_map"] = os.path.join(sd, "glue_map.png")

    def layer_rgb(p):
        if p.kind in ("ring", "base", "cap"):
            return RGB_RING
        return RGB_B if p.offset else RGB_A

    def closeup(key, joint_kind, part_kind, elev, dist, with_jig=False):
        kinds = (joint_kind, "scarf end") if joint_kind == "lap end" else (joint_kind,)
        cands = [j for j, jt in enumerate(st.joints) if jt.kind in kinds and
                 (part_kind is None or st.parts[jt.parts[0]].kind == part_kind)]
        if not cands:
            return
        # pick the joint facing the camera most squarely (largest x)
        j = max(cands, key=lambda k: joint_point(st, k)[0])
        c = joint_point(st, j)
        near = [i for i, p in enumerate(st.parts) if p.sec is not None and any(
            np.linalg.norm(st.nodes[m.i] - st.nodes[st.joints[j].node]) < 40 or
            np.linalg.norm(st.nodes[m.j] - st.nodes[st.joints[j].node]) < 40 for m in (st.members[k] for k in p.members))]
        groups = defaultdict(list)
        for i in near:
            groups[layer_rgb(st.parts[i])].extend(part_shapes[i])
        items = [{"file": _export(sh, os.path.join(tmp, f"{key}_{n}.step")), "rgb": rgb}
                 for n, (rgb, sh) in enumerate(groups.items())]
        jn = [k for k in range(len(st.joints)) if np.linalg.norm(joint_point(st, k) - c) < 25]
        for g, ds in glue_dots(st, jn, joint_glue).items():
            items.append({"file": _export(ds, os.path.join(tmp, f"{key}_glue_{g}.step")), "rgb": GLUE_RGB[g]})
        if with_jig:
            items.append({"file": _export([s for k, s in layout.items() if k.startswith("base_plate")],
                                          os.path.join(tmp, f"{key}_jig.step")), "rgb": RGB_JIG})
        az = float(np.degrees(np.arctan2(c[1], c[0])))
        png = os.path.join(sd, f"detail_{key}.png")
        if rnd.render_spec_subprocess({"items": items, "distance": dist, "elevation": elev, "azimuth": az,
                                       "center": list(map(float, c)), "size": [800, 700]}, png):
            out[key] = png

    closeup("crossing", "lap", None, 10, 42)
    closeup("ring", "lap end", "ring", 12, 48)
    closeup("base", "lap end", "base", 14, 55, with_jig=True)
    closeup("cap", "lap end", "cap", 25, 48)
    return out


def attach_existing(st: Structure, steps: list[Step], outdir: str) -> list[Step]:
    """Fill in model/picture paths and joints from an earlier export (rebuild the guide without re-rendering)."""
    done: set[int] = set()
    joints_done: set[int] = set()
    for n, stp in enumerate(steps, start=1):
        pset = set(stp.parts)
        new_j = {j for j, jt in enumerate(st.joints) if all(p in done | pset for p in jt.parts)} - joints_done
        name = f"step_{n:02d}_{stp.key}"
        stp.model = f"steps/{name}.step"
        png = os.path.join(outdir, "steps", name + ".png")
        stp.picture = png if os.path.exists(png) else ""
        stp.new_joints = sorted(new_j)
        done |= pset
        joints_done |= new_j
    return steps


PREP_JIGS = ("fit_test", "cutting_guide", "end_sanding", "marking_gauge", "scarf_sanding")


def render_prep_jigs(jig_parts, sd: str) -> str:
    """Picture of the jigs used to prepare the parts (cutting guide, end-sanding jigs, scarf guide)."""
    from . import render as rnd
    tmp = os.path.join(sd, ".prep")
    os.makedirs(tmp, exist_ok=True)
    picks = [jp for jp in jig_parts if jp.name.startswith(PREP_JIGS)]
    items, x = [], 0.0
    for jp in picks:
        bb = jp.shape.BoundingBox()
        sh = jp.shape.translate(cq.Vector(-bb.xmin, x - bb.ymin, -bb.zmin))
        x += bb.ylen + 12
        p = os.path.join(tmp, jp.name + ".step")
        cq.exporters.export(cq.Workplane().add(sh), p)
        items.append({"file": p, "rgb": list(RGB_JIG)})
    out = os.path.join(sd, "prep_jigs.png")
    ok = rnd.render_spec_subprocess({"items": items, "distance": 360, "elevation": 55, "azimuth": -90,
                                     "center": [110, x / 2, 0], "size": [1200, 800]}, out)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return out if ok else ""
