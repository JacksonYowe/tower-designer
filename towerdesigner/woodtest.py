"""Wood test: measure density, crushing strength and stiffness of supplied 3 x 3 sticks.

1. Density - weigh several sticks of known length.
2. Crush   - short samples (12 mm long, too short to buckle) pressed on a scale until they crush.
             Crushing strength f_c = peak force / area.
3. Buckle  - long samples (150 mm) pressed on a scale until they bow sideways. The load stops rising
             at the buckling load; Euler: E = P (K L)^2 / (pi^2 I).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .materials import WOODS
from .spec import Spec

G = 9.81
CRUSH_LEN = 12.0      # mm: 4x the width, so it crushes instead of buckling
BUCKLE_LEN = 150.0    # mm
K_CAPS = 0.65         # effective length factor with the printed square-socket end caps
K_PINNED = 1.0        # ends sanded to a rounded point
WOOD_FILE = os.path.expanduser("~/.towerdesigner/wood_test.json")


@dataclass
class WoodTest:
    side: float = 3.0                 # mm, measured stick size (calipers), square
    n_sticks: int = 5
    stick_len: float = 900.0          # mm each
    stick_mass: float = 0.0           # g, all sticks together
    crush_len: float = CRUSH_LEN
    crush_kg: list[float] = field(default_factory=list)
    buckle_len: float = BUCKLE_LEN
    buckle_K: float = K_CAPS
    buckle_kg: list[float] = field(default_factory=list)

    @property
    def area(self) -> float:
        return self.side ** 2

    def density(self) -> float:
        """kg/m^3"""
        if self.stick_mass <= 0 or self.n_sticks <= 0 or self.stick_len <= 0:
            return 0.0
        vol = self.n_sticks * self.stick_len * self.area      # mm^3
        return self.stick_mass / vol * 1e6                     # g/mm^3 -> kg/m^3

    def fc(self) -> float:
        """MPa, mean crushing strength."""
        vals = [v for v in self.crush_kg if v > 0]
        return float(np.mean(vals)) * G / self.area if vals else 0.0

    def E(self) -> float:
        """MPa, stiffness along the grain from the buckling loads."""
        vals = [v for v in self.buckle_kg if v > 0]
        if not vals:
            return 0.0
        I = self.side ** 4 / 12
        return float(np.mean(vals)) * G * (self.buckle_K * self.buckle_len) ** 2 / (math.pi ** 2 * I)

    def checks(self) -> list[str]:
        out = []
        rho, fc, E = self.density(), self.fc(), self.E()
        crush = [v for v in self.crush_kg if v > 0]
        if len(crush) >= 2 and np.std(crush) / np.mean(crush) > 0.25:
            out.append("Crush results vary by more than 25 % — check the ends are square and test a few more.")
        if rho and fc and not 0.04 <= fc / rho <= 0.11:
            out.append(f"Crushing strength {fc:.1f} MPa is unusual for {rho:.0f} kg/m³ wood (expect "
                       f"{0.055 * rho:.0f}–{0.085 * rho:.0f} MPa). Re-check the readings.")
        if rho and E and not 10 <= E / rho <= 40:
            out.append(f"Stiffness {E:.0f} MPa is unusual for {rho:.0f} kg/m³ wood (expect "
                       f"{15 * rho:.0f}–{30 * rho:.0f} MPa). Were the long sticks straight and the ends seated?")
        if fc and self.crush_len > 5 * self.side:
            out.append("Crush samples longer than 5× their width may buckle instead of crushing.")
        return out

    def apply(self, spec: Spec) -> Spec:
        spec.wood = "teacher"
        spec.measured_density = round(self.density(), 1)
        spec.measured_fc = round(self.fc(), 2)
        spec.measured_E = round(self.E(), 0)
        return spec

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def load_wood_test() -> WoodTest | None:
    try:
        with open(WOOD_FILE) as f:
            return WoodTest(**json.load(f))
    except (OSError, ValueError, TypeError):
        return None


def save_wood_test(t: WoodTest):
    os.makedirs(os.path.dirname(WOOD_FILE), exist_ok=True)
    with open(WOOD_FILE, "w") as f:
        json.dump(t.to_dict(), f, indent=2)


def apply_saved(spec: Spec) -> Spec:
    t = load_wood_test()
    if t and t.density() > 0 and t.fc() > 0:
        t.apply(spec)
    return spec


# ---------------------------------------------------------------------------------------------
# kit: printable helpers + method sheet
# ---------------------------------------------------------------------------------------------
def _jigs(side: float = 3.0):
    import cadquery as cq
    hole = side + 0.25
    # squaring block: sample sits in a hole exactly CRUSH_LEN deep; sand both ends flush
    sq = cq.Workplane().box(40, 16, CRUSH_LEN, centered=(True, True, False))
    for x in (-12, -4, 4, 12):
        sq = sq.cut(cq.Workplane().rect(hole, hole).extrude(CRUSH_LEN + 2).translate((x, 0, -1)))
    # crush pads: shallow square pocket centres the sample on the scale and under the press block
    pad = cq.Workplane().box(30, 30, 4, centered=(True, True, False))
    pad = pad.cut(cq.Workplane().rect(hole + 0.2, hole + 0.2).extrude(1.0).translate((0, 0, 3)))
    # buckling end caps: square socket 4 mm deep (behaves close to a fixed end, K ~ 0.65)
    cap = cq.Workplane().box(26, 26, 8, centered=(True, True, False))
    cap = cap.cut(cq.Workplane().rect(hole, hole).extrude(4).translate((0, 0, 4)))
    # cutting guide for 150 mm buckling samples
    guide = cq.Workplane().box(BUCKLE_LEN + 20, 14, 8, centered=(False, True, False))
    guide = guide.cut(cq.Workplane().box(BUCKLE_LEN + 14, hole, 6, centered=(False, True, False)).translate((3, 0, 3)))
    guide = guide.cut(cq.Workplane().box(0.8, 20, 10).translate((3 + BUCKLE_LEN, 0, 6)))
    return {"wood_test_squaring_block": sq.val(), "wood_test_crush_pad": pad.val(),
            "wood_test_buckling_cap": cap.val(), "wood_test_150mm_cut_guide": guide.val()}


def export_kit(outdir: str, spec: Spec) -> list[str]:
    import cadquery as cq
    os.makedirs(outdir, exist_ok=True)
    files = []
    for name, shp in _jigs().items():
        for ext in ("stl", "step"):
            p = os.path.join(outdir, f"{name}.{ext}")
            cq.exporters.export(cq.Workplane().add(shp), p)
            files.append(p)
    page = os.path.join(outdir, "wood_test.html")
    with open(page, "w") as f:
        f.write(procedure_html(spec))
    files.append(page)
    return files


def procedure_html(spec: Spec) -> str:
    book = WOODS["teacher"]
    exp_crush = book.f_c_L * 9 / G
    exp_buckle = math.pi ** 2 * book.E_L * (3 ** 4 / 12) / (K_CAPS * BUCKLE_LEN) ** 2 / G
    box = "<td style='min-width:60px'></td>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Wood test</title><style>
body{{font:15px/1.5 -apple-system,Helvetica,sans-serif;max-width:900px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:6px}}th{{background:#f4efe8}}
li{{margin-bottom:6px}}.warn{{background:#fff3cd;padding:8px;border-radius:6px}}</style></head><body>
<h1>Wood test — how strong are <i>your</i> 3 × 3 sticks?</h1>
<p>The app's strength predictions are only as good as its wood data. The supplied sticks are an unknown species,
so these three quick tests replace the book values with measurements. Do them on sticks from the same batch you'll build with.</p>
<h2>You need</h2>
<ul><li>About 6 sticks, a 0.01 g scale, and a <b>digital scale that reads to at least 25 kg</b> (bathroom scale,
luggage scale, or a heavy-duty kitchen scale). A school materials-testing machine is even better.</li>
<li>Printed helpers: squaring block, two crush pads, two buckling end caps, 150 mm cut guide (files in this folder).</li>
<li>Calipers if you have them, a fine saw or sharp knife, 240-grit sandpaper on a flat surface, a flat offcut of board
to press with, and a friend filming the scale display (to catch the peak reading).</li></ul>

<h2>Test 1 — Density (5 min)</h2>
<ol><li>Measure the stick size with calipers in a few places (nominal 3.0 mm).</li>
<li>Weigh 5 whole sticks together and measure their length. Enter count, length, and total mass in the app.</li></ol>

<h2>Test 2 — Crushing strength (15 min)</h2>
<ol><li>Cut 5 pieces a little over {CRUSH_LEN:.0f} mm long from different sticks.</li>
<li>Push each piece into the squaring block and sand both ends flush on the sandpaper. That makes them exactly
{CRUSH_LEN:.0f} mm with square ends — square ends matter more than anything else here.</li>
<li>Put a crush pad on the scale, stand the sample in its pocket, put the second pad on top (pocket down), then lay the
board on top. Zero the scale first with the pads on.</li>
<li>Press down <b>slowly and steadily</b> (about 10 s to failure) with both hands flat on the board. The reading climbs,
then stops climbing or drops as the wood crushes. Record the <b>highest</b> reading.</li>
<li>Repeat for all 5. Expect very roughly {exp_crush:.0f} kg each if the sticks are medium balsa.</li></ol>

<h2>Test 3 — Stiffness by buckling (10 min)</h2>
<ol><li>Cut 3 straight pieces exactly {BUCKLE_LEN:.0f} mm long using the cut guide. Pick straight ones.</li>
<li>Put one end cap on the scale (socket up), stand the stick in it, put the other cap on top, and press down
gently through the board, keeping it vertical.</li>
<li>The stick bows sideways and the reading stops rising. Record that steady reading (the buckling load) — don't keep
pushing until it snaps. Expect roughly {exp_buckle:.1f} kg for medium balsa.</li></ol>

<h2>Enter the results</h2>
<p>In the app: <b>Wood and glue → Wood test…</b>, type in the numbers, check the warnings, click <b>Apply</b>. The app then
uses your measured density, crushing strength and stiffness for every design (it still applies the
{spec.quality:.0%} build-quality factor on top, because a glued tower is never as good as a clean sample).</p>
<table><tr><th></th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th></tr>
<tr><th>Crush peak (kg)</th>{box * 5}</tr>
<tr><th>Buckling load (kg)</th>{box * 3}<td>—</td><td>—</td></tr></table>
<p>Density: sticks ____ × length ____ mm, total mass ____ g, size ____ mm.</p>
<p class="warn">Safety: wood can splinter when it crushes. Wear glasses and keep fingers off the sides of the sample.</p>
</body></html>"""
