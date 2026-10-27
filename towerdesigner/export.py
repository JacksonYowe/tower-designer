"""Write every output for one design into a folder."""
from __future__ import annotations

import json
import os

import cadquery as cq

from . import cad, lasercut
from .analysis import analyse
from .families import build
from .instructions import make_html
from .spec import Spec


def _plain(o):
    """numpy numbers and other odd types -> plain JSON values."""
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def save_design(path: str, family: str, params: dict, spec: Spec, summary: dict | None = None):
    """Write safely: build the whole file first, then swap it in, so a failed save never leaves it empty."""
    text = json.dumps(dict(family=family, params=params, spec=spec.to_dict(), summary=summary or {}),
                      indent=2, default=_plain)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def load_design(path: str) -> tuple[str, dict, Spec]:
    with open(path) as f:
        d = json.load(f)
    return d["family"], d["params"], Spec.from_dict(d["spec"])


def export_all(family: str, params: dict, spec: Spec, outdir: str, progress=None) -> dict:
    say = progress or (lambda msg: None)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "jig"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "laser"), exist_ok=True)
    say("Analysing")
    st = build(family, params, spec)
    res = analyse(st, spec, accurate=True)
    files = {}

    save_design(os.path.join(outdir, "design.json"), family, st.params, spec,
                dict(load_kg=res.load_kg, mass_g=res.mass.total, governing=res.governing, glue=res.glue))
    files["design.json"] = "re-open this design in the app"

    say("Building tower 3D model")
    assy = cad.tower_assembly(st)
    cad.save_assembly(assy, os.path.join(outdir, "tower.step"))
    files["tower.step"] = "the whole tower, every piece a separate part (import into Onshape)"
    try:
        cq.exporters.export(cq.Workplane().add(assy.toCompound()), os.path.join(outdir, "tower.stl"),
                            tolerance=0.05, angularTolerance=0.2)
        files["tower.stl"] = "tower as one mesh (for quick viewing)"
    except Exception:
        pass

    say("Building jig")
    jig_parts, jig_notes = cad.make_jig(st, spec)
    jassy = cq.Assembly(name="jig")
    for jp in jig_parts:
        cad.export_shape(jp.shape, os.path.join(outdir, "jig", f"{jp.name}.step"),
                         os.path.join(outdir, "jig", f"{jp.name}.stl"))
    # assembled jig (collars and spacers in place) for checking in Onshape
    from .stepmodels import jig_layout
    for name, shape in jig_layout(st, jig_parts).items():
        jassy.add(shape, name=name.replace("#", "_"), color=cq.Color(0.2, 0.5, 0.9))
    cad.save_assembly(jassy, os.path.join(outdir, "jig", "jig_assembly.step"))
    files["jig/*.stl"] = "print these (one file per jig part)"
    files["jig/*.step"] = "same jig parts as STEP, for Onshape"
    files["jig/jig_assembly.step"] = "the jig assembled around the tower position"

    say("Nesting laser parts")
    pieces = lasercut.collect_pieces(st)
    sheets, problems = lasercut.nest(pieces)
    for k, sh in enumerate(sheets, start=1):
        base = os.path.join(outdir, "laser", f"sheet{k}_{sh.name}")
        lasercut.write_dxf(sh, base + ".dxf")
        lasercut.write_svg(sh, base + ".svg")
    if sheets:
        files["laser/*.dxf, *.svg"] = "laser cutter files, one per sheet (grain along the length)"

    if st.family.endswith("3"):
        say("Writing the assembly guide (PDF)")
        from . import guide, render
        imgs = {}
        for what in ("tower", "jig"):
            png = os.path.join(outdir, f"picture_{what}.png")
            if render.render_subprocess(outdir, what, png):
                imgs[what] = png
        say("Building a 3D model of every step")
        from . import stepmodels
        steps = stepmodels.plan_steps(st, jig_parts)
        steps, extras = stepmodels.write_step_models(st, jig_parts, steps, outdir, res.joint_glue)
        imgs.update({f"detail_{k}" if k != "glue_map" else k: v for k, v in extras.items()})
        prep = stepmodels.render_prep_jigs(jig_parts, os.path.join(outdir, "steps"))
        if prep:
            imgs["prep"] = prep
        files["steps/*.step"] = "3D model of every build step (blue jig, tan built, orange new)"
        gh = os.path.join(outdir, "assembly_guide.html")
        with open(gh, "w") as f:
            say("Checking that every part fits")
            from . import tolerances
            fit = tolerances.fit_report(st, jig_parts)
            f.write(guide.make_guide_html(st, res, spec, jig_parts, imgs, steps, fit=fit))
        if guide.write_pdf(gh, os.path.join(outdir, "Assembly guide.pdf")):
            files["Assembly guide.pdf"] = "how to build it: grain, glue per joint, cuts, jig, build order"
        files["assembly_guide.html"] = "the same guide as a web page"

    # the wood and glue test kits travel with the build pack
    import shutil
    app = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for kit in ("Wood test kit", "Glue test kit"):
        src = os.path.join(app, kit)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(outdir, "test kits", kit), dirs_exist_ok=True)
            files[f"test kits/{kit}/"] = "print and use before building (guide section 5)"

    say("Writing instructions")
    files["instructions.html"] = "this page"
    page = make_html(st, res, spec, sheets, problems, jig_parts, jig_notes, files)
    with open(os.path.join(outdir, "instructions.html"), "w") as f:
        f.write(page)
    say("Done")
    return dict(outdir=outdir, result=res, structure=st, sheets=sheets, jig=jig_parts, notes=jig_notes + problems)
