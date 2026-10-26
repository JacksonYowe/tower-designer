"""Build instructions as a single self-contained HTML page (print it to PDF from the browser)."""
from __future__ import annotations

import html
import math
from collections import Counter, defaultdict

import numpy as np

from .analysis import Result, analyse
from .families import FAMILY_NAMES
from .lasercut import part_ids, sheet_svg, stick_plan
from .materials import GLUES, STOCK_BY_ID
from .spec import Spec
from .structure import Structure

ISO = math.radians(30)


def _proj(p):
    x, y, z = p
    u = x * math.cos(ISO) - y * math.sin(ISO)
    v = -z + (x * math.sin(ISO) + y * math.cos(ISO)) * 0.35
    return u, v


def tower_svg(st: Structure, highlight: set[int] = frozenset(), done: set[int] = frozenset(),
              util=None, height=300) -> str:
    """Isometric line drawing. highlight = parts added in this step, done = parts already on."""
    segs = []
    for m in st.members:
        a, b = _proj(st.nodes[m.i]), _proj(st.nodes[m.j])
        segs.append((m, a, b))
    us = [q[0] for s in segs for q in s[1:]]
    vs = [q[1] for s in segs for q in s[1:]]
    pad = 8
    x0, y0 = min(us) - pad, min(vs) - pad
    w, h = max(us) - min(us) + 2 * pad, max(vs) - min(vs) + 2 * pad
    lines = []
    order = sorted(range(len(segs)), key=lambda k: (segs[k][0].part in highlight, segs[k][0].part in done))
    for k in order:
        m, a, b = segs[k]
        if util is not None:
            u = float(min(util[k], 1.0))
            col = f"rgb({int(60 + 195 * u)},{int(170 - 120 * u)},{int(90 - 60 * u)})"
            sw, op = 0.8 + 1.6 * max(m.sec.w, m.sec.t) / 8, 1
        elif m.part in highlight:
            col, sw, op = "#e8590c", 2.2, 1
        elif m.part in done:
            col, sw, op = "#6b5a45", 1.1, 1
        else:
            col, sw, op = "#c9bfb2", 0.6, 0.6
        lines.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{col}" '
                     f'stroke-width="{sw:.2f}" stroke-linecap="round" opacity="{op}"/>')
    width = height * w / h
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.1f} {y0:.1f} {w:.1f} {h:.1f}" '
            f'width="{width:.0f}" height="{height}">' + "".join(lines) + "</svg>")


STEP_TITLES = {
    "stick": lambda s, npn: {1: "Base ring", 2: "Legs into the jig", 3 + npn: "Top cap"}.get(
        s, f"Storey {s - 2}: ring and bracing" if 3 <= s < 3 + npn else f"Step {s}"),
    "panel": lambda s, npn: {1: "Lattice panels", 2: "Glue the corners", 3: "Top cap"}.get(s, f"Step {s}"),
    "diagrid": lambda s, npn: {1: "Base ring", 2: "Inner strips (set A)", 3: "Outer strips (set B)",
                               4: "Rings at the crossings", 5: "Top cap"}.get(s, f"Step {s}"),
}

STEP_HOW = {
    ("stick", 1): "If the base ring is one piece, lay it in its pocket in the base plate now. If it is "
                  "separate strips they are glued between the legs at the start of the next step.",
    ("stick", 2): "Put the mast in the base plate, then stack spacer, collar segments, spacer, collar... "
                  "Drop each leg through the base ring (if one-piece) into its foot pocket and press it "
                  "into the collar notches. Put a rubber band round every collar to hold the legs. "
                  "Glue the base ring to the legs.",
    ("panel", 1): "Laser cut the panels. Check each against its drawing, sand the edges lightly and weigh "
                  "them. The grain runs up the panel.",
    ("panel", 2): "Assemble the jig (mast, spacers, collars). Stand the panels round it with their edges "
                  "in the collar notches and their feet in the base-plate pockets. Rubber-band each collar. "
                  "Run a thin bead of glue down each corner where two panel edges meet.",
    ("diagrid", 2): "Assemble the jig. Lay the A strips (they twist one way) into the collar notches with "
                    "their feet in the pockets. Rubber-band each collar. Glue their feet to the base ring.",
    ("diagrid", 3): "Lay the B strips over the A strips, twisting the other way. Glue every crossing "
                    "(a small dab, pressed for 20 s).",
}


def build_steps(st: Structure, res: Result, spec: Spec):
    ids = part_ids(st)
    steps = sorted({p.step for p in st.parts})
    npn = len(st.levels) - 1
    base = st.family.rstrip("3")
    titles = STEP_TITLES[base]
    out = []
    cum = 0.0
    done: set[int] = set()
    for s in steps:
        parts = [i for i, p in enumerate(st.parts) if p.step == s]
        joints = [j for j, jt in enumerate(st.joints) if jt.step == s]
        wood = sum(res.mass.per_part[i] for i in parts)
        glue = sum(res.mass.per_joint[j] for j in joints)
        glue += sum(res.mass.part_laminate[i] for i in parts)
        cum += wood + glue
        out.append(dict(step=s, title=titles(s, npn), parts=parts, joints=joints, wood=wood, glue=glue,
                        cum=cum, svg=tower_svg(st, set(parts), set(done)), how=STEP_HOW.get((base, s), ""),
                        ids=[ids[i] for i in parts]))
        done |= set(parts)
    return out


def _fmt_g(g):
    return f"{g:.2f} g" if g >= 0.1 else f"{g * 1000:.0f} mg"


def make_html(st: Structure, res: Result, spec: Spec, sheets, laser_problems, jig_parts, jig_notes,
              files: dict[str, str]) -> str:
    ids = part_ids(st)
    E = html.escape
    used = Counter(res.joint_glue)
    if res.mass.laminate_glue > 0:
        used.setdefault(res.laminate_glue_id, 0)
    glue_order = [g for g in GLUES if g in used]
    # what each single glue would give, to show the trade-off
    alts = {g: analyse(st, spec, accurate=True, glue_key=g) for g in GLUES}
    fac_txt = ", ".join(f"{E(GLUES[g].name)} ×{f:.1f} " + ("(measured with the glue test)" if g in spec.glue_factors
                        else "(default, matched to a 5 g per 15 g rule of thumb)") for g, f in spec.factors().items())
    E = html.escape
    p = st.params
    steps = build_steps(st, res, spec)

    # ---- shopping list ----------------------------------------------------------------------
    shop = Counter()
    for sh in sheets:
        shop[sh.stock.id] += 1
    plan = stick_plan(st)
    for stock, bins in plan.items():
        shop[stock] += len(bins)
    rows, total = [], 0.0
    for sid, q in sorted(shop.items()):
        s = STOCK_BY_ID[sid]
        spare = 1 if s.is_sheet else 1
        rows.append(f"<tr><td>{E(s.title)}</td><td>{q} (+{spare} spare)</td><td>${s.price:.2f}</td>"
                    f"<td>${s.price * (q + spare):.2f}</td></tr>")
        total += s.price * (q + spare)
    for g in glue_order:
        gl = GLUES[g]
        rows.append(f"<tr><td>{E(gl.product)} <span class='small'>({E(gl.where)})</span></td><td>1</td>"
                    f"<td>${gl.price:.2f}</td><td>${gl.price:.2f}</td></tr>")
        total += gl.price
    shop_html = "".join(rows)

    # ---- parts table ------------------------------------------------------------------------
    prow = []
    for i, part in enumerate(st.parts):
        if part.kind == "corner":
            continue
        size = part.sec.label() if part.sec else ""
        if part.kind in ("cap", "base") and part.info.get("g_ang") is not None:
            size = f"{part.info['t']:g} mm one-piece frame (laser)"
        if part.kind == "panel":
            size = f"{part.info['t']:g} mm lattice panel (laser)"
        grain = ""
        if part.kind in ("cap", "base", "panel") and part.members:
            gmax = max(st.members[m].grain_deg for m in part.members)
            grain = f"up to {gmax:.0f}°"
        else:
            grain = "along"
        m = res.mass.per_part[i]
        prow.append(f"<tr><td><b>{ids[i]}</b></td><td>{E(part.name)}</td><td>{E(size)}</td>"
                    f"<td>{part.length:.1f}</td><td>{grain}</td><td>{m / spec.margin * 1000:.0f} mg</td>"
                    f"<td><b>{m * 1000:.0f} mg</b></td><td>{part.step}</td></tr>")

    # ---- glue: which glue where ---------------------------------------------------------------
    glue_rows = ""
    for g in glue_order:
        gl = GLUES[g]
        js = [j for j, c in enumerate(res.joint_glue) if c == g]
        kinds = Counter(st.joints[j].kind for j in js)
        what = ", ".join(f"{v} {k}" for k, v in kinds.items())
        if g == res.laminate_glue_id and res.mass.laminate_glue > 0:
            what += (", " if what else "") + "laminating L/box legs"
        dry_tot = sum(res.mass.per_joint[j] for j in js) + (res.mass.laminate_glue if g == res.laminate_glue_id else 0)
        dry_each = np.mean([res.mass.per_joint[j] for j in js]) * 1000 if js else 0
        wet_each = np.mean([res.mass.per_joint_wet[j] for j in js]) * 1000 if js else 0
        glue_rows += (f"<tr><td><b>{E(gl.name)}</b><br><span class='small'>{E(gl.product)}</span></td><td>{len(js)}</td>"
                      f"<td>{E(what)}</td><td>{wet_each:.1f} mg</td><td>{dry_each:.1f} mg</td><td>{dry_tot:.2f} g</td>"
                      f"<td class='small'>{E(gl.note)} {E(gl.cure)}.</td></tr>")
    alt_rows = "".join(
        f"<tr><td>{E(GLUES[g].name)} everywhere</td><td>{a.load_kg:.1f} kg</td><td>{a.mass.glue:.2f} g</td>"
        f"<td>{a.mass.total:.2f} g{' ⚠️ over limit' if a.mass.total > spec.max_mass else ''}</td></tr>"
        for g, a in alts.items())
    alt_rows += (f"<tr><td><b>This design ({'mixed per joint' if res.glue == 'mixed' else E(GLUES[res.glue].name)})"
                 f"</b></td><td><b>{res.load_kg:.1f} kg</b></td><td><b>{res.mass.glue:.2f} g</b></td>"
                 f"<td><b>{res.mass.total:.2f} g</b></td></tr>")

    # ---- laser + print + hand-cut lists -------------------------------------------------------
    laser_html = ""
    for k, sh in enumerate(sheets, start=1):
        labs = ", ".join(pc.label for pc, _, _ in sh.placed)
        laser_html += (f"<h4>Sheet {k}: {E(sh.stock.title)}</h4><p class='small'>File: "
                       f"<code>laser/sheet{k}_{sh.name}.dxf</code> (and .svg). Pieces: {E(labs)}</p>"
                       f"<div class='sheet'>{sheet_svg(sh, labels=True, for_laser=False)}</div>")
    if laser_problems:
        laser_html += "<p class='warn'>" + "<br>".join(E(x) for x in laser_problems) + "</p>"
    if not sheets:
        laser_html = "<p>No laser-cut parts in this design.</p>"
    hand = ""
    for stock, bins in plan.items():
        s = STOCK_BY_ID[stock]
        for b_i, b in enumerate(bins, start=1):
            cuts = ", ".join(f"{lab} {ln:.1f} mm" for lab, ln in b)
            hand += f"<li>{E(s.title)} #{b_i}: {E(cuts)}</li>"
    hand_html = f"<ul>{hand}</ul>" if hand else "<p>None — everything is laser cut.</p>"
    # shaving and scarfing (supplied 3 x 3 sticks)
    shaved = defaultdict(list)
    for i, part in enumerate(st.parts):
        if part.sec is not None and part.sec.stock == "teacher-3x3" and (part.sec.t < 2.95 or part.sec.w < 2.95):
            shaved[(round(part.sec.t, 1), round(part.sec.w, 1))].append(ids[i])
    style = st.params.get("joint", "mitre")
    prep = ""
    if style == "lap":
        prep += ("<p><b>Lap every brace and ring end over the leg.</b> Sand the end to the meeting angle so it lies flat "
                 "across the outside face of the leg, overlapping it by the full leg width. The glue then bonds side "
                 "grain to side grain. Only shape the ends: every piece stays a full 3 × 3 mm along its length.</p>")
    elif any(j.kind == "butt" for j in st.joints):
        prep += ("<p><b>Mitre every brace and ring end</b> to the meeting angle (use the cutting guides) so it sits flush "
                 "against the side of the leg with no gap. A square-on end-grain joint only takes about 60 % of the "
                 "strength of a side-grain one, so a gap makes it much weaker.</p>")
    jig_rows = "".join(f"<tr><td><code>jig/{E(j.name)}.stl</code></td><td>{j.qty}</td><td>{E(j.note)}</td></tr>"
                       for j in jig_parts)
    jig_warn = "".join(f"<p class='warn'>{E(n)}</p>" for n in jig_notes)

    # ---- steps ------------------------------------------------------------------------------
    step_html = ""
    for k, s in enumerate(steps, start=1):
        js = [st.joints[j] for j in s["joints"]]
        by = defaultdict(list)
        for j in s["joints"]:
            by[(res.joint_glue[j], st.joints[j].kind)].append(j)
        items = []
        for (g, kind), jj in sorted(by.items()):
            wet = np.mean([res.mass.per_joint_wet[j] for j in jj]) * 1000
            dry = np.mean([res.mass.per_joint[j] for j in jj]) * 1000
            items.append(f"<li><b>{len(jj)} × {kind}</b> joint{'s' if len(jj) > 1 else ''} with <b>{E(GLUES[g].name)}</b>: "
                         f"apply ≈ {wet:.0f} mg each (dries to ≤ {dry:.0f} mg)</li>")
        glue_line = (f"<p><b>Glue:</b></p><ul>{''.join(items)}</ul>" if items
                     else "<p><b>Glue joints:</b> none in this step.</p>")
        step_html += f"""
<section class="step"><div class="fig">{s['svg']}</div><div class="txt">
<h3>Step {k}. {E(s['title'])}</h3>
<p>{E(s['how'])}</p>
<p><b>Parts:</b> {E(', '.join(s['ids']))}</p>
{glue_line}
<p class="check">⚖️ Weigh-in: after this step the tower must weigh <b>no more than {s['cum']:.2f} g</b>
(this step adds {_fmt_g(s['wood'])} wood + {_fmt_g(s['glue'])} glue). Weigh it once the glue has dried.</p>
</div></section>"""

    # ---- top summary ------------------------------------------------------------------------
    gov = E(res.governing)
    jl = sorted(res.lam_joints.values())
    weakest_joint = f"{jl[0] / 9.81:.1f} kg" if jl else "n/a"
    summary_svg = tower_svg(st, util=res.util, height=420)
    pstr = ", ".join(f"{k} = {v:.1f}" if isinstance(v, float) else f"{k} = {v}" for k, v in p.items())
    css = """
body{font:15px/1.5 -apple-system,Helvetica,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}
h1{margin-bottom:0}h2{border-bottom:2px solid #e8590c;padding-bottom:4px;margin-top:36px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}td,th{border:1px solid #ddd;padding:4px 6px;text-align:left}
th{background:#f6f1ea}.small{font-size:12px;color:#555}.warn{background:#fff3cd;padding:8px;border-radius:6px}
.check{background:#e7f5ff;padding:8px;border-radius:6px}.step{display:flex;gap:16px;border-bottom:1px solid #eee;padding:12px 0;page-break-inside:avoid}
.step .fig{flex:0 0 200px;text-align:center}.sheet svg{width:100%;height:auto;border:1px solid #ccc;background:#fff}
.hero{display:flex;gap:24px;align-items:flex-start}.kpi{font-size:28px;font-weight:700;color:#e8590c}
code{background:#f3f3f3;padding:1px 4px;border-radius:3px}
@media print{.step{page-break-inside:avoid}h2{page-break-before:auto}}"""
    files_html = "".join(f"<li><code>{E(k)}</code> — {E(v)}</li>" for k, v in files.items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Tower build instructions</title>
<style>{css}</style></head><body>
<h1>{E(FAMILY_NAMES[st.family])}</h1>
<p class="small">Generated by Tower Designer. Height {st.height:.0f} mm, fits over a {spec.pole_d:.0f} mm pole
({2 * spec.bore_r:.0f} mm clear bore). Wood: {E(spec.wood_props(False).name)}.</p>
<div class="hero"><div>{summary_svg}<p class="small">Colour = how hard each piece works at failure (green → red).</p></div><div>
<p>Predicted failure load<br><span class="kpi">{res.load_kg:.1f} kg</span></p>
<p>Mass including +3 % on every piece and joint<br><span class="kpi">{res.mass.total:.2f} g</span>
<span class="small">(limit {spec.max_mass:.1f} g; wood {res.mass.wood:.2f} g, glue {res.mass.glue:.2f} g)</span></p>
<p><b>Predicted first failure:</b> {gov}.<br><b>Weakest glue joint:</b> {weakest_joint}.
<b>Whole-tower buckling:</b> {res.lam_global / 9.81:.0f} kg.</p>
<p class="small">The prediction already knocks strength down to {spec.quality:.0%} for hand-built variable wood, assumes the
load plate bears {spec.uneven:.0%} unevenly and the tower is built up to {spec.sway:.1f} mm out of true. Treat it as a target:
real towers usually fail somewhere between 60 % and 100 % of it.</p></div></div>

<h2>1. Glue — chosen joint by joint</h2>
<p>Every joint gets the <b>lightest glue that is still strong enough not to be the weak point</b>. Joints that carry
little load get light glue; the few critical joints get the strong (heavier) one. Glue amounts come from each glue's
viscosity and each joint's glued area, edge length and end grain: a squeezed film, a bead round the edges, and what soaks
into the wood. Hand factors used: {fac_txt}.
This design: <b>{res.mass.glue:.2f} g of glue on {res.mass.wood:.2f} g of wood</b> ({res.mass.glue_ratio * 15:.1f} g per 15 g).</p>
<table><tr><th>Glue</th><th>Joints</th><th>Used for</th><th>Apply each</th><th>Dried each (+3 %)</th><th>Total dried</th><th>Notes</th></tr>{glue_rows}</table>
<p><b>Trade-off</b> — the same tower with one glue everywhere:</p>
<table><tr><th>Glue</th><th>Predicted load</th><th>Glue mass</th><th>Total mass</th></tr>{alt_rows}</table>
<p class="small">Tips: decant each glue into its own small cup and apply with a toothpick; label the cups. Wipe away
squeeze-out before it dries — it is dead weight. Mix only small batches of epoxy. Balsa cement and PVA can be used on the
same joint area; superglue and epoxy should go on clean, dry wood.</p>

<h2>2. Shopping list (Bunnings)</h2>
<table><tr><th>Item</th><th>Qty</th><th>Each</th><th>Total</th></tr>{shop_html}
<tr><td colspan="3"><b>Total</b></td><td><b>${total:.2f}</b></td></tr></table>
<p class="small">Also: rubber bands, packing tape or baking paper (to stop glue sticking to the jig), 240-grit sandpaper on a flat block,
a sharp hobby knife, a fine razor saw if any sticks are hand cut, and a 0.01 g scale. Weigh a sheet and enter its density in the app
(mass ÷ volume) — paulownia varies a lot.</p>

<h2>3. Laser cutting</h2>
<p>Grain always runs along the 915 mm length of the sheet — the files are already laid out that way, so <b>do not rotate parts</b>
in the laser software. Red = cut, blue = labels (do not cut; engrave lightly or ignore). Do a test cut first:
paulownia chars more than balsa, so use low power, high speed and several passes, and air assist.</p>
{laser_html}

<h2>4. Hand-cut sticks</h2>{hand_html}{('<h3>Shaping the sticks</h3>' + prep) if prep else ''}

<h2>5. 3D-printed jig</h2>
<p>PLA, 0.2 mm layers, 3 walls, 15 % infill, no supports. Cover every surface that touches a joint with packing tape
(PVA and balsa cement barely stick to PLA; superglue and epoxy stick hard).</p>
<table><tr><th>File</th><th>Qty</th><th>Notes</th></tr>{jig_rows}</table>{jig_warn}

<h2>6. Parts</h2>
<p class="small">"Max" mass is the predicted mass + 3 %. Weigh every piece; any piece heavier than its max should be re-cut from lighter wood.</p>
<table><tr><th>ID</th><th>Part</th><th>Size</th><th>Length mm</th><th>Grain</th><th>Predicted</th><th>Max</th><th>Step</th></tr>
{''.join(prow)}</table>

<h2>7. Build order</h2>{step_html}
<section class="step"><div class="txt"><h3>Final. Remove from the jig and finish</h3>
<p>Let the glue cure for at least 24 h. Take off the rubber bands or tape, lift the tower and mast off the base plate and pull
the mast up out of the top. Then take the jig out from the bottom up: the lowest spacer, then each segment of the lowest collar
(push it inwards off the legs and lower it out through the base), then the next spacer, and so on. Sand the top flat with short
straight strokes on sandpaper laid on a flat table so the load plate sits evenly on every leg — uneven bearing is the main reason
towers fail early.
Final weigh-in: <b>≤ {res.mass.total:.2f} g</b> (limit {spec.max_mass:.1f} g).</p></div></section>

<h2>8. Import into Onshape</h2>
<ol><li>In an Onshape document click <b>+ → Import…</b> and pick <code>tower.step</code>. Every piece comes in as its own part.</li>
<li>Do the same with <code>jig/jig_assembly.step</code> or the individual jig files in <code>jig/</code>.</li>
<li>Insert both into an Assembly to check fit, make drawings, or edit the jig.</li></ol>

<h2>9. Files</h2><ul>{files_html}</ul>
<p class="small">Design parameters: {E(pstr)}</p>
</body></html>"""
