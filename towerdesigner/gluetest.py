"""Glue test: glue and weigh a few standard joints per glue, then set each glue's hand factor.

Standard test joint (same kind as a brace end on a leg):
  * host  : 100 mm of Boyle 5 x 5 mm square rod
  * strips: 5 x laser-cut 2.5 x 3 x 30 mm strips, each butt-glued by its END onto the top of the host
            (end grain into side grain, square, 17.5 mm apart)
Weigh the host and its 5 strips before gluing and again once fully dry. The gain / 5 is the dried
glue per joint; dividing by what the model predicts (with factor 1) gives that glue's hand factor.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import lasercut
from .materials import GLUES, Glue, GlueAmount, Wood, glue_amount
from .spec import Spec

N_JOINTS = 5
STRIP = (2.5, 3.0, 30.0)          # thickness, width, length mm
HOST_LEN = 100.0
PITCH = 17.5
TEST_AREA = STRIP[0] * STRIP[1]
TEST_PERIMETER = 2 * (STRIP[0] + STRIP[1])
GLUE_FILE = os.path.expanduser("~/.towerdesigner/glue_factors.json")


def load_glue_factors() -> dict:
    """Per-glue hand factors saved from your last glue test (empty if you haven't done one)."""
    try:
        import json
        with open(GLUE_FILE) as f:
            return {k: float(v) for k, v in json.load(f).items() if k in GLUES}
    except (OSError, ValueError):
        return {}


def save_glue_factors(d: dict):
    import json
    os.makedirs(os.path.dirname(GLUE_FILE), exist_ok=True)
    with open(GLUE_FILE, "w") as f:
        json.dump(d, f, indent=2)


CURE = {"pva": "24 h", "ca": "1 h", "epoxy": "24 h", "cement": "12 h"}
SHORT = {"pva": "PV", "ca": "CA", "epoxy": "EP", "cement": "BC"}


def predicted(glue: Glue, wood: Wood, factor: float = 1.0) -> GlueAmount:
    """Glue the model expects in ONE standard test joint."""
    return glue_amount(TEST_AREA, TEST_PERIMETER, TEST_AREA, glue, wood, factor)


@dataclass
class Measurement:
    glue: str
    joints: int
    before: float     # g, host + strips before gluing
    after: float      # g, after fully dry

    @property
    def per_joint(self) -> float:
        return (self.after - self.before) / max(self.joints, 1)


def evaluate(m: Measurement, spec: Spec) -> dict:
    wood = spec.wood_props(False)
    raw = predicted(GLUES[m.glue], wood, 1.0).dry
    now = predicted(GLUES[m.glue], wood, spec.factors()[m.glue]).dry
    gain = m.after - m.before
    warn = ""
    if gain <= 0:
        warn = "No mass gain — check the numbers."
    elif gain < 0.03:
        warn = "Gain under 0.03 g: a 0.01 g scale can't resolve this well. Glue more joints or use a 0.001 g scale."
    factor = m.per_joint / raw if gain > 0 and raw > 0 else None
    return dict(per_joint_mg=m.per_joint * 1000, model_now_mg=now * 1000, model_raw_mg=raw * 1000,
                factor=factor, warning=warn)


# ---------------------------------------------------------------------------------------------
# test kit
# ---------------------------------------------------------------------------------------------
def _strip_pieces(glues: list[str], spares: int = 1) -> list[lasercut.Piece]:
    t, w, L = STRIP
    rect = np.array([[0, 0], [L, 0], [L, w], [0, w]], float)
    out = []
    for g in glues:
        for k in range(1, N_JOINTS + spares + 1):
            out.append(lasercut.Piece(-1, f"{SHORT[g]}{k}", [rect.copy()], t))
    return out


def _jig():
    """Base with a channel for the host rod, and a comb that holds the strips square."""
    import cadquery as cq
    L, W = HOST_LEN + 24, 26.0
    base = cq.Workplane().box(L, W, 8, centered=(True, True, False))
    base = base.cut(cq.Workplane().box(HOST_LEN + 1, 5.3, 4, centered=(True, True, False)).translate((0, 0, 4)))
    # finger notches to lift the rod out, and drip holes under every joint
    for x in (-HOST_LEN / 2 - 2, HOST_LEN / 2 + 2):
        base = base.cut(cq.Workplane().box(4, 12, 6, centered=(True, True, False)).translate((x, 0, 3)))
    xs = [(k - (N_JOINTS - 1) / 2) * PITCH for k in range(N_JOINTS)]
    for x in xs:
        base = base.cut(cq.Workplane().box(4, 3, 10).translate((x, 0, 2)))
    post_x = L / 2 - 5
    for x in (-post_x, post_x):
        post = cq.Workplane().box(8, 8, 14, centered=(True, True, False)).translate((x, W / 2 - 4, 8))
        peg = cq.Workplane().circle(1.4).extrude(3).translate((x, W / 2 - 4, 22))
        base = base.union(post).union(peg)
    # comb: same coordinates as the base; slots over the rod (y = 0), peg holes over the posts
    py = W / 2 - 4
    comb = cq.Workplane().box(L, 16, 2.4, centered=(True, True, False)).translate((0, 5, 0))
    for x in xs:
        comb = comb.cut(cq.Workplane().box(STRIP[0] + 0.4, STRIP[1] + 0.4, 10).translate((x, 0, 0)))
    for x in (-post_x, post_x):
        comb = comb.cut(cq.Workplane().circle(1.65).extrude(10).translate((x, py, -1)))
    return base.val(), comb.val()


def export_kit(outdir: str, spec: Spec, glues: list[str] | None = None) -> list[str]:
    """Laser file for the test strips, a printable jig, and the procedure + results sheet."""
    import cadquery as cq
    glues = glues or list(GLUES)
    os.makedirs(outdir, exist_ok=True)
    files = []
    sheets, _ = lasercut.nest(_strip_pieces(glues))
    for k, sh in enumerate(sheets, start=1):
        base = os.path.join(outdir, f"glue_test_strips_{k}")
        lasercut.write_dxf(sh, base + ".dxf")
        lasercut.write_svg(sh, base + ".svg")
        files += [base + ".dxf", base + ".svg"]
    try:
        b, c = _jig()
        for name, shp in (("glue_test_jig_base", b), ("glue_test_jig_comb", c)):
            for ext in ("stl", "step"):
                p = os.path.join(outdir, f"{name}.{ext}")
                cq.exporters.export(cq.Workplane().add(shp), p)
                files.append(p)
    except Exception as e:  # the kit is still usable without the jig
        files.append(f"(jig not generated: {e})")
    page = os.path.join(outdir, "glue_test.html")
    with open(page, "w") as f:
        f.write(procedure_html(spec, glues))
    files.append(page)
    return files


def procedure_html(spec: Spec, glues: list[str]) -> str:
    wood = spec.wood_props(False)
    fac = spec.factors()
    rows = ""
    for g in glues:
        gl = GLUES[g]
        p = predicted(gl, wood, fac[g])
        rows += (f"<tr><td><b>{gl.name}</b><br><small>{gl.product}</small></td><td>{SHORT[g]}1–{SHORT[g]}{N_JOINTS}</td>"
                 f"<td>{CURE[g]}</td><td>{p.wet * 1000:.0f} mg wet → {p.dry * 1000:.0f} mg dry</td>"
                 f"<td class=box></td><td class=box></td><td class=box></td></tr>")
    t, w, L = STRIP
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Glue test</title><style>
body{{font:15px/1.5 -apple-system,Helvetica,sans-serif;max-width:900px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:6px;vertical-align:top}}
th{{background:#f4efe8}}.box{{min-width:70px}}li{{margin-bottom:6px}}</style></head><body>
<h1>Glue test — how much glue does <i>your</i> hand leave?</h1>
<p>The app estimates glue per joint from each glue's viscosity and the joint's size, then multiplies by a
<b>hand factor</b> for how you actually apply it. This test measures that factor for each glue, so the tower's
mass budget is based on your gluing, not a guess.</p>
<h2>You need</h2>
<ul><li>Per glue: one {HOST_LEN:.0f} mm length of Boyle 5 × 5 mm square rod (the "host") and {N_JOINTS} strips
{t:g} × {w:g} × {L:g} mm — laser cut from <code>glue_test_strips_1.dxf</code> (labelled {', '.join(SHORT[g] + '1…' for g in glues)};
one spare each).</li>
<li>The printed jig: <code>glue_test_jig_base.stl</code> and <code>glue_test_jig_comb.stl</code> (optional but keeps every joint square).</li>
<li>A scale reading to 0.01 g (0.001 g is better), the glues, a toothpick per glue, a pencil.</li></ul>
<h2>Method</h2>
<ol>
<li><b>Label</b> each host rod with its glue (pencil). Keep each glue's host and strips together.</li>
<li><b>Weigh before:</b> put one host and its {N_JOINTS} strips on the scale together. Write the mass in the table. Do this for every glue.</li>
<li><b>Set up:</b> lay the host in the jig channel and sit the comb on its pegs.</li>
<li><b>Glue</b> each strip: put glue on one END of the strip exactly the way you will on the real tower
(same toothpick, same size dab, same wiping). Push it down through the comb onto the host and hold it a few seconds.
Don't add extra just because it's a test — the point is to copy your real habit.</li>
<li><b>Wipe</b> squeeze-out the way you would on the tower (or don't, if you won't).</li>
<li><b>Cure</b> fully (times below). Lift the comb off carefully. PVA and balsa cement lose water/solvent for a day —
weigh again the next day and use the lower reading.</li>
<li><b>Weigh after:</b> the host with its {N_JOINTS} glued strips. Write it down.</li>
<li><b>Enter</b> the before and after masses in the app: <i>Wood and glue → Glue test…</i>, then click <b>Apply</b>.
The app sets each glue's factor and every design is re-weighed with your real glue amounts.</li>
</ol>
<p><small>Why the strip end goes onto the side of the rod: that is the most common joint in the tower (a brace or ring
ending on a leg), and the end grain soaks up glue the way it will on the real thing. Do all glues on the same day,
with the same wood, so they compare fairly.</small></p>
<h2>Results</h2>
<table><tr><th>Glue</th><th>Strips</th><th>Cure</th><th>App currently expects per joint</th><th>Before (g)</th><th>After (g)</th><th>Gain ÷ {N_JOINTS} (mg)</th></tr>
{rows}</table>
<p><small>Standard joint: {TEST_AREA:.1f} mm² glued, {TEST_PERIMETER:.0f} mm of edge, all end grain on the strip side.</small></p>
</body></html>"""
