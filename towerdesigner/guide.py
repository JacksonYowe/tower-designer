"""Real-world assembly guide (print-ready HTML -> PDF) for the 3 x 3 stick towers.

Everything in it is generated from the design: cut list with lengths and end angles, which glue
goes on which joint and how much, grain rules, jig stacking heights, and the build order.
"""
from __future__ import annotations

import base64
import html
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict

import numpy as np

from . import cad
from .analysis import Result, analyse
from .families import FAMILY_NAMES
from .instructions import tower_svg
from .stepmodels import Step, _natural
from .lasercut import part_ids, stick_plan
from .materials import GLUES, STOCK_BY_ID
from .spec import Spec
from .structure import Structure

E = html.escape
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

GLUE_HOW = {
    "cement": ("Double-glue. Brush a thin coat on both faces, wait 5–10 min until it is tacky-dry (it soaks into "
               "the wood), add a second thin coat to one face and press together for 30 s. Leave 12 h before loading."),
    "pva": ("Thin coat on one face only, press together for 30–60 s, then hold with a rubber band or peg for 20 min. "
            "Full strength after 24 h."),
    "epoxy": ("Mix a pea-sized blob of each part on scrap card for 1 min. Apply with a toothpick within 4 min. "
              "It doesn't shrink, so fill the joint completely. Handle after 1 h, full strength after 24 h."),
    "ca": ("Fit the joint dry first. Add one small drop at the edge and let it wick in. Hold 10 s. Wear glasses, "
           "work with a window open, and keep it off the PLA jig (it bonds to it)."),
}
GLUE_COLOUR = {"cement": "#d91a1a", "pva": "#1a66f2", "epoxy": "#991ad9", "ca": "#26993f"}  # = stepmodels.GLUE_RGB


def _img(path: str | None, width="100%") -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:{width};height:auto">'


# ---------------------------------------------------------------------------------------------
# small explanatory drawings
# ---------------------------------------------------------------------------------------------
def svg_grain() -> str:
    lines = "".join(f'<line x1="20" y1="{30 + k * 6}" x2="300" y2="{30 + k * 6 + (2 if k % 3 == 0 else 0)}" '
                    f'stroke="#b07a3a" stroke-width="1"/>' for k in range(6))
    return f"""<svg viewBox="0 0 420 120" width="420" height="120" xmlns="http://www.w3.org/2000/svg">
<rect x="20" y="28" width="280" height="32" fill="#f0d3a3" stroke="#7a4f22"/>{lines}
<polygon points="300,28 320,18 320,50 300,60" fill="#e2b77a" stroke="#7a4f22"/>
<g stroke="#7a4f22">{''.join(f'<circle cx="{306 + (k % 3) * 5}" cy="{30 + (k // 3) * 9}" r="1.6" fill="none"/>' for k in range(9))}</g>
<text x="20" y="85" font-size="12" fill="#333">Side grain: long faces (good glue surface, strong)</text>
<text x="20" y="102" font-size="12" fill="#333">End grain: the cut end (soaks up glue, weak bond)</text>
<text x="325" y="40" font-size="11" fill="#333">end</text>
<line x1="60" y1="16" x2="260" y2="16" stroke="#333" marker-end="url(#a)"/>
<text x="110" y="12" font-size="11">grain runs along the stick</text>
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z"/></marker></defs>
</svg>"""


def svg_runout() -> str:
    good = "".join(f'<line x1="10" y1="{14 + k * 5}" x2="190" y2="{14 + k * 5}" stroke="#b07a3a"/>' for k in range(5))
    bad = "".join(f'<line x1="{10 + k * 40}" y1="12" x2="{50 + k * 40}" y2="36" stroke="#b07a3a"/>' for k in range(5))
    return f"""<svg viewBox="0 0 420 60" width="420" height="60" xmlns="http://www.w3.org/2000/svg">
<rect x="10" y="10" width="180" height="28" fill="#f0d3a3" stroke="#2b8a3e" stroke-width="2"/>{good}
<g transform="translate(220,0)"><rect x="10" y="10" width="180" height="28" fill="#f0d3a3" stroke="#c92a2a" stroke-width="2"/>{bad}</g>
<text x="10" y="54" font-size="11">✔ grain parallel to the edges</text><text x="230" y="54" font-size="11">✘ grain runs off the edge (snaps)</text>
</svg>"""


def svg_level_cut(angle: float) -> str:
    """Stick leaning at (90 - angle) deg standing on the table with a level-cut foot."""
    tilt = 90 - angle
    t = math.radians(tilt)
    L, w = 150, 18
    x0, y0 = 60, 150
    dx, dy = L * math.sin(t), -L * math.cos(t)
    ox, oy = w * math.cos(t), w * math.sin(t)
    pts = [(x0, y0), (x0 + ox / math.cos(t) if t else x0 + w, y0), (x0 + dx + ox / math.cos(t), y0 + dy),
           (x0 + dx, y0 + dy)]
    p = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    return f"""<svg viewBox="0 0 300 175" width="300" height="175" xmlns="http://www.w3.org/2000/svg">
<line x1="10" y1="150" x2="290" y2="150" stroke="#555" stroke-width="2"/>
<polygon points="{p}" fill="#f0d3a3" stroke="#7a4f22"/>
<line x1="{x0}" y1="150" x2="{x0}" y2="40" stroke="#999" stroke-dasharray="4 3"/>
<text x="{x0 + 8}" y="60" font-size="12">{tilt:.0f}° lean</text>
<text x="150" y="168" font-size="12">foot cut level: {angle:.0f}° to the stick ({tilt:.0f}° off square)</text>
</svg>"""


def svg_lap(angle: float) -> str:
    g = math.radians(angle)
    return f"""<svg viewBox="0 0 300 150" width="300" height="150" xmlns="http://www.w3.org/2000/svg">
<rect x="130" y="5" width="22" height="140" fill="#e9c48d" stroke="#7a4f22"/>
<text x="157" y="140" font-size="11">strip</text>
<g transform="rotate({90 - angle:.1f} 141 75)">
<polygon points="20,66 141,66 141,84 20,84" fill="#f5d55a" stroke="#8a6d10" opacity="0.95"/>
<polygon points="141,66 262,66 262,84 141,84" fill="#f5d55a" stroke="#8a6d10" opacity="0.6"/></g>
<text x="5" y="20" font-size="11">ring piece lies ON the strip's</text><text x="5" y="34" font-size="11">outer face, to its centre line</text>
<text x="160" y="118" font-size="11">end trimmed parallel</text><text x="160" y="132" font-size="11">to the strip ({angle:.0f}°)</text>
</svg>"""


def svg_scarf() -> str:
    return """<svg viewBox="0 0 340 150" width="300" height="132" xmlns="http://www.w3.org/2000/svg">
<rect x="150" y="5" width="22" height="140" fill="#9e7347" stroke="#5a3a1a"/><text x="176" y="140" font-size="11">B strip</text>
<polygon points="20,62 196,62 126,86 20,86" fill="#f5d55a" stroke="#8a6d10"/>
<polygon points="196,62 320,62 320,86 126,86" fill="#e8c23a" stroke="#8a6d10" opacity="0.92"/>
<line x1="196" y1="62" x2="126" y2="86" stroke="#d91a1a" stroke-width="3"/>
<text x="5" y="20" font-size="11">two ring pieces overlap on a 1-in-3</text><text x="5" y="34" font-size="11">diagonal (~9 mm) right over the strip</text>
<text x="200" y="110" font-size="11">red line = glued scarf face</text>
</svg>"""


def svg_crossing() -> str:
    return """<svg viewBox="0 0 340 140" width="300" height="124" xmlns="http://www.w3.org/2000/svg">
<g transform="rotate(-20 150 70)"><rect x="139" y="0" width="22" height="140" fill="#e9c48d" stroke="#7a4f22"/></g>
<g transform="rotate(20 150 70)"><rect x="139" y="0" width="22" height="140" fill="#f0d3a3" stroke="#7a4f22" opacity="0.95"/></g>
<circle cx="150" cy="70" r="14" fill="none" stroke="#d9480f" stroke-width="2" stroke-dasharray="3 2"/>
<text x="185" y="72" font-size="11">glue here: the two faces</text><text x="185" y="86" font-size="11">touch where B crosses A</text>
<text x="10" y="20" font-size="11">no cutting needed</text>
</svg>"""


# ---------------------------------------------------------------------------------------------
def _cut_groups(st: Structure):
    """Pieces grouped by (group, length, end angles) — what you actually cut."""
    ids = part_ids(st)
    groups = defaultdict(list)
    for i, p in enumerate(st.parts):
        if p.sec is None or p.kind == "corner":
            continue
        key = (p.group or p.name, round(p.length, 1), round(p.end_angles[0]), round(p.end_angles[1]),
               tuple(p.info.get("end_style", ["?", "?"])))
        groups[key].append(i)
    return groups, ids, cad.channel_of(st)


def _chan_txt(chan_of, pis) -> str:
    ch = chan_of.get(pis[0])
    return f"<b class='tag'>CG{ch['guide']}</b> ch {ch['ch']}" if ch else "—"


def physical_host(st: Structure, j: int) -> int:
    """The strip a ring end is actually glued to: the outer (B) strip at that crossing."""
    jt = st.joints[j]
    if jt.kind == "lap" or jt.kind == "scarf":
        return jt.parts[1]
    at = [p for k in range(len(st.joints)) if st.joints[k].kind == "lap" and st.joints[k].node == jt.node
          for p in st.joints[k].parts[:2]]
    return max(at, key=lambda p: st.parts[p].offset) if at else jt.parts[1]



def epoxy_summary(st: Structure, res: Result, ids, joints) -> str:
    """Epoxy joints in words: whole rings by name when every joint in them is epoxy, else one by one."""
    by = defaultdict(list)
    for j in joints:
        by[st.parts[st.joints[j].parts[0]].group or "?"].append(j)
    out = []
    for grp, js in by.items():
        epo = [j for j in js if res.joint_glue[j] == "epoxy"]
        if not epo:
            continue
        if len(epo) == len(js):
            ps = sorted({ids[st.joints[j].parts[0]] for j in js}, key=_natural)
            out.append(f"every joint of the {grp.lower().replace('top cap', 'top ring')} ({len(js)} ends, {ps[0]}–{ps[-1]})")
        else:
            out.append(", ".join(sorted({f"{ids[st.joints[j].parts[0]]}→{ids[st.joints[j].parts[1]]}" for j in epo},
                                        key=_natural)))
    return "; ".join(out)


def glue_legend(used, scarf: bool = False) -> str:
    dots = "".join(f'<span style="display:inline-block;margin-right:14px"><span style="display:inline-block;width:11px;'
                   f'height:11px;border-radius:6px;background:{GLUE_COLOUR[g]};vertical-align:-1px"></span> '
                   f'<b>{E(GLUES[g].name)}</b></span>' for g in used)
    sq = (" On the ring maps: <b>●</b> dot = ring end glued to a strip, <b>■</b> square = scarf overlap glued to "
          "the next ring piece. Same colour = same glue." if scarf else "")
    return f'<p class="legend">Glue dots: {dots}{sq}</p>'


def svg_layers() -> str:
    """Section through one crossing, looking down: layers from the pole outwards."""
    return """<svg viewBox="0 0 470 170" width="470" height="170" xmlns="http://www.w3.org/2000/svg">
<rect x="10" y="40" width="60" height="90" fill="#dde3ea" stroke="#889"/><text x="14" y="150" font-size="11">pole side</text>
<rect x="95" y="62" width="70" height="46" fill="#a9c3f5" stroke="#4a6fb5"/><text x="98" y="125" font-size="10">jig collar</text>
<text x="98" y="137" font-size="10">(inside, removed)</text>
<rect x="165" y="70" width="30" height="30" fill="#dbb885" stroke="#7a4f22"/><text x="160" y="60" font-size="11">A strip</text>
<rect x="195" y="70" width="30" height="30" fill="#9e7347" stroke="#5a3a1a"/><text x="196" y="118" font-size="11">B strip</text>
<rect x="225" y="70" width="30" height="30" fill="#f5d55a" stroke="#8a6d10"/><text x="230" y="60" font-size="11">ring</text>
<circle cx="195" cy="85" r="4" fill="#d91a1a"/><circle cx="225" cy="85" r="4" fill="#d91a1a"/>
<line x1="165" y1="150" x2="255" y2="150" stroke="#333"/><text x="170" y="163" font-size="10">3 + 3 + 3 mm, glued face to face</text>
<text x="275" y="72" font-size="11">outside of tower →</text>
<text x="275" y="92" font-size="10">dots = glue: crossing (A–B)</text><text x="275" y="106" font-size="10">and ring end (B–ring)</text>
</svg>"""


def svg_base_section() -> str:
    """Side view of a foot in the base plate."""
    return """<svg viewBox="0 0 470 170" width="470" height="170" xmlns="http://www.w3.org/2000/svg">
<rect x="10" y="110" width="450" height="40" fill="#a9c3f5" stroke="#4a6fb5"/>
<text x="18" y="142" font-size="11">base plate (jig)</text>
<rect x="140" y="98" width="60" height="12" fill="#a9c3f5" stroke="none"/>
<line x1="10" y1="110" x2="140" y2="110" stroke="#4a6fb5"/><line x1="200" y1="110" x2="460" y2="110" stroke="#4a6fb5"/>
<polygon points="150,30 172,30 180,122 158,122" fill="#9e7347" stroke="#5a3a1a"/><text x="110" y="28" font-size="11">strip foot</text>
<rect x="160" y="101" width="120" height="16" fill="#f5d55a" stroke="#8a6d10"/>
<text x="290" y="100" font-size="11">base ring: sits in a groove,</text><text x="290" y="114" font-size="11">0.5 mm clear of the jig</text>
<line x1="140" y1="122" x2="290" y2="122" stroke="#333" stroke-dasharray="3 2"/><text x="300" y="136" font-size="10">table level (pocket floor)</text>
<circle cx="182" cy="108" r="4" fill="#d91a1a"/>
</svg>"""


def ring_map(st: Structure, res: Result, parts: list[int], ids: list[str]) -> tuple[str, str]:
    """Top-down map of one ring with a glue-coloured dot at every joint, plus a joint-by-joint table."""
    pset = set(parts)
    # joints onto the strips (two per piece) and scarf joints between neighbouring pieces (one per crossing)
    ring_joints = [(j, jt) for j, jt in enumerate(st.joints) if jt.parts[0] in pset and jt.kind != "scarf"]
    scarf_joints = [(j, jt) for j, jt in enumerate(st.joints) if jt.kind == "scarf" and jt.parts[0] in pset]
    if not ring_joints:
        return "", ""
    xy = {jt.node: st.nodes[jt.node][:2] for _, jt in ring_joints}
    R = max(float(np.hypot(*v)) for v in xy.values())
    S, cx, cy = 150 / R, 200, 190
    P = lambda v: (cx + v[0] * S, cy - v[1] * S)     # looking down from above, x to the right
    lines, dots, labels = [], [], []
    for i in parts:
        js = [(j, jt) for j, jt in ring_joints if jt.parts[0] == i]
        if len(js) != 2:
            continue
        (ja, a), (jb, b) = js
        pa, pb = np.array(P(xy[a.node])), np.array(P(xy[b.node]))
        lines.append(f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="#d9b13b" stroke-width="6"/>')
        mid = (pa + pb) / 2
        out = (mid - [cx, cy]) / (np.linalg.norm(mid - [cx, cy]) or 1)
        lp = mid + out * 13
        labels.append(f'<text x="{lp[0]:.1f}" y="{lp[1] + 3:.1f}" font-size="7.5" text-anchor="middle">{E(ids[i])}</text>')
        for j, jt, pend in ((ja, a, pa), (jb, b, pb)):
            d = pend + (mid - pend) / (np.linalg.norm(mid - pend) or 1) * 5    # dot just inside this piece's end
            dots.append(f'<circle cx="{d[0]:.1f}" cy="{d[1]:.1f}" r="3.6" fill="{GLUE_COLOUR[res.joint_glue[j]]}" '
                        f'stroke="#fff" stroke-width="0.8"/>')
    for j, jt in scarf_joints:     # scarf glue: a square marker just outside the ring at that crossing
        p = np.array(P(xy[jt.node])) if jt.node in xy else None
        if p is None:
            continue
        o = p + (p - [cx, cy]) / (np.linalg.norm(p - [cx, cy]) or 1) * 6
        dots.append(f'<rect x="{o[0] - 3:.1f}" y="{o[1] - 3:.1f}" width="6" height="6" '
                    f'fill="{GLUE_COLOUR[res.joint_glue[j]]}" stroke="#fff" stroke-width="0.8"/>')
    # strip labels at the nodes (first few and S1 highlighted)
    host_at = {}
    for j, jt in ring_joints:
        host_at[jt.node] = ids[physical_host(st, j)]
    first = ids[physical_host(st, ring_joints[0][0])][:1] + "1"
    for node, lab in host_at.items():
        p = np.array(P(xy[node]))
        inn = p + ([cx, cy] - p) / (np.linalg.norm([cx, cy] - p) or 1) * 14
        weight = "bold" if lab == first else "normal"
        col = "#c92a2a" if lab == first else "#666"
        labels.append(f'<text x="{inn[0]:.1f}" y="{inn[1] + 3:.1f}" font-size="7" text-anchor="middle" '
                      f'font-weight="{weight}" fill="{col}">{E(lab)}</text>')
    svg = (f'<svg viewBox="0 0 400 380" width="330" height="314" xmlns="http://www.w3.org/2000/svg">'
           f'<circle cx="{cx}" cy="{cy}" r="{12.5 * S:.1f}" fill="#eef1f5" stroke="#aab"/>'
           f'<text x="{cx}" y="{cy + 3}" font-size="8" text-anchor="middle" fill="#889">pole</text>'
           + "".join(lines) + "".join(dots) + "".join(labels) +
           f'<text x="8" y="372" font-size="8" fill="#555">Seen from above. The strip each end is glued to is named inside '
           f'the ring; {first} (red) is at the base-plate mark. Dots: glue on the strip at each end'
           + ('; squares: glue on the scarf overlap.' if scarf_joints else '.') + '</text></svg>')
    # table: one row per piece, glue at each end
    rows = ""
    for i in sorted(parts, key=lambda i: _natural(ids[i])):
        js = [(j, jt) for j, jt in ring_joints if jt.parts[0] == i][:2]
        cells = "".join(
            f"<td>over <b>{E(ids[physical_host(st, j)])}</b>: <b style='color:{GLUE_COLOUR[res.joint_glue[j]]}'>"
            f"● {E(GLUES[res.joint_glue[j]].name)}</b></td>" for j, jt in js)
        if scarf_joints:
            sc = [(j, jt) for j, jt in scarf_joints if i in jt.parts[:2]]
            nxt = [(j, jt) for j, jt in sc if jt.parts[0] == i or jt.parts[1] == i]
            cells += "<td>" + ", ".join(
                f"with <b>{E(ids[jt.parts[1] if jt.parts[0] == i else jt.parts[0]])}</b>: "
                f"<b style='color:{GLUE_COLOUR[res.joint_glue[j]]}'>■ {E(GLUES[res.joint_glue[j]].name)}</b>"
                for j, jt in nxt) + "</td>"
        rows += f"<tr><td><b>{E(ids[i])}</b></td>{cells}</tr>"
    cnt = Counter(res.joint_glue[j] for j, _ in ring_joints + scarf_joints)
    summary = ", ".join(f"<b style='color:{GLUE_COLOUR[g]}'>{n} × {E(GLUES[g].name)}</b>" for g, n in cnt.most_common())
    table = (f"<p><b>Every joint in this ring</b> ({summary}):</p>"
             f"<table class='tight'><tr><th>Piece</th><th>End 1 (on strip)</th><th>End 2 (on strip)</th>"
             + ("<th>Scarf overlaps</th>" if scarf_joints else "") + f"</tr>{rows}</table>")
    return svg, table


LABELS: dict[str, str] = {}      # jig file name -> engraved ID, filled in by make_guide_html


def _lab(name: str) -> str:
    lab = LABELS.get(name, "")
    return f'<b class="tag">{E(lab)}</b> ' if lab else ""


def _files_list(names: list[str]) -> str:
    """Compact list of jig parts: engraved ID + file; collar segments grouped per collar."""
    collars = defaultdict(list)
    other = []
    for n in sorted(names, key=_natural):
        if n.startswith("collar_"):
            c, seg = n.split("_seg")
            collars[c].append(int(seg))
        else:
            other.append(n)
    out = [f"{_lab(n)}<code>{E(n)}.stl</code>" for n in other]
    for c, segs in sorted(collars.items(), key=lambda kv: _natural(kv[0])):
        k = c.split("_")[1]
        out.append(f'<b class="tag">C{k}-{min(segs)} … C{k}-{max(segs)}</b> <code>{E(c)}_seg{min(segs)}.stl</code> … '
                   f"<code>{E(c)}_seg{max(segs)}.stl</code> ({len(segs)} pieces)")
    return "<br>".join(out)


def _sand_tag(a0, s0, a1, s1) -> str:
    tags = []
    for a, s in ((a0, s0), (a1, s1)):
        off = int(round(90 - a))
        if s == "scarf":
            t = LABELS.get("scarf_sanding_guide", "SG")
        elif s in ("level", "lap", "mitre") and off >= 1:
            t = LABELS.get(f"end_sanding_jig_{off}deg", "")
        else:
            t = ""
        if t and t not in tags:
            tags.append(t)
    return " + ".join(f"<b class='tag'>{E(t)}</b>" for t in tags) or "—"


def _end_txt(a, s):
    off = 90 - a
    return {"level": f"level cut, ~{off:.0f}° off square", "lap": f"lap end, trimmed {off:.0f}° off square",
            "scarf": "scarf: 1-in-3 slope through the thickness, ~9 mm",
            "mitre": f"mitre, {off:.0f}° off square"}.get(s, "square")


GLOSSARY = [
    ("Strip", "One of the long 220 mm sticks that run from the table to the top, leaning."),
    ("Set A / set B", "The two sets of strips. A strips are the inner layer and lean one way; B strips are the outer "
                      "layer and lean the other way. The pieces are identical until you letter them."),
    ("Letter face", "The face you write A or B on. It always faces outwards on the tower."),
    ("Foot (F)", "The end of a strip that stands on the table. You mark it F."),
    ("Crossing", "Where a B strip passes over an A strip; they are glued face to face there."),
    ("Level", "One of the heights where the strips cross (1 = lowest). Rings sit at some levels."),
    ("Tick", "A pencil line on a strip, made with the marking gauge MG1, showing where a crossing goes."),
    ("Ring", "A hoop of short pieces glued round the outside at one level."),
    ("Scarf", "A long sloping end (1 in 3, about 9 mm) so two ring pieces overlap and glue on a big face."),
    ("Level cut", "An angled end cut so a leaning strip stands flat on a level surface."),
    ("Lap joint", "Two pieces glued face to face where one overlaps the other."),
    ("Pocket", "A shallow V-shaped hole in the base plate where two strip feet stand."),
    ("Collar / segment", "A printed ring that holds the strips at the right place; it comes in numbered segments so "
                         "it can be taken out afterwards."),
    ("Notch", "The slot in a collar segment that a strip presses into."),
    ("Mast / spacer", "The rod up the middle of the jig, and the tubes stacked on it that set each collar's height. "
                      "Each spacer has a disc on top that the collar segments rest on."),
    ("Grain", "The direction of the wood fibres. In a stick it runs along its length."),
    ("End grain", "The cut end of a stick. It soaks up glue and bonds poorly."),
    ("Double-glue", "Balsa cement method: coat both faces, let it go tacky, coat again and press."),
    ("Cure", "The time a glue needs to reach full strength."),
    ("Buckling", "A slender piece bowing sideways under compression; the usual way thin sticks fail."),
]


def print_plan(jig_parts, bed: float = 256.0) -> list[tuple[str, list[str], float]]:
    """Rough print plates in the order to print them: (title, IDs, grams of PLA). Simple shelf packing."""
    from .stepmodels import PREP_JIGS
    order = [("Test first", lambda n: n.startswith("fit_test")),
             ("Preparation jigs", lambda n: n.startswith(PREP_JIGS) and not n.startswith("fit_test")),
             ("Base plate", lambda n: n.startswith("base_plate")),
             ("Mast and spacers", lambda n: n == "mast" or n.startswith("spacer"))]
    order += [(f"Collar C{k}", (lambda k: lambda n: n.startswith(f"collar_{k}_"))(k))
              for k in sorted({int(j.name.split("_")[1]) for j in jig_parts if j.name.startswith("collar_")})]
    plates = []
    for title, pick in order:
        items = [j for j in jig_parts if pick(j.name)]
        if not items:
            continue
        cur, x, y, row = [], 0.0, 0.0, 0.0
        grams = 0.0
        for j in sorted(items, key=lambda j: -max(j.shape.BoundingBox().xlen, j.shape.BoundingBox().ylen)):
            bb = j.shape.BoundingBox()
            w, h = (bb.xlen, bb.ylen) if bb.xlen >= bb.ylen else (bb.ylen, bb.xlen)
            if j.name == "mast":
                w, h = bb.zlen, bb.xlen
            if x + w > bed:
                x, y, row = 0.0, y + row + 5, 0.0
            if y + h > bed and cur:
                plates.append((title, cur, grams))
                cur, x, y, row, grams = [], 0.0, 0.0, 0.0, 0.0
            cur.append(j.label or j.name)
            grams += j.shape.Volume() / 1000 * 1.24 * 0.45     # PLA, 3 walls + 15 % infill ~ 45 % solid
            x += w + 5
            row = max(row, h)
        plates.append((title, cur, grams))
    return plates


def make_guide_html(st: Structure, res: Result, spec: Spec, jig_parts, images: dict,
                    steps: list[Step] | None = None) -> str:
    from .stepmodels import plan_steps
    steps = steps or plan_steps(st, jig_parts)
    LABELS.clear()
    LABELS.update({jp.name: jp.label for jp in jig_parts if getattr(jp, "label", "")})
    ids = part_ids(st)
    by_grp = defaultdict(list)
    for i, p in enumerate(st.parts):
        by_grp[p.group].append(i)
    groups, _, guide_keys = _cut_groups(st)
    plan = stick_plan(st)
    n_sticks = sum(len(b) for b in plan.values())
    stick_len = STOCK_BY_ID["teacher-3x3"].length
    fam = FAMILY_NAMES.get(st.family, st.family)
    info = st.info
    measured = spec.measured_fc > 0
    diag = st.family.startswith("diagrid")
    w = spec.wood_props(False)

    # ---- glue plan ------------------------------------------------------------------------------
    loc = defaultdict(Counter)
    wet = defaultdict(list)
    dry = defaultdict(list)
    for j, jt in enumerate(st.joints):
        a = st.parts[jt.parts[0]].group or st.parts[jt.parts[0]].name
        b = st.parts[jt.parts[1]].group or st.parts[jt.parts[1]].name
        if b.startswith("Strip"):
            b = "the strips (lapped across the outside)"
        place = f"{a} → {b}" if jt.kind != "lap" else "Crossings (B strip over A strip)"
        g = res.joint_glue[j]
        loc[(place, jt.kind)][g] += 1
        wet[(place, g)].append(res.mass.per_joint_wet[j])
        dry[(place, g)].append(res.mass.per_joint[j])
    glue_rows = ""
    for (place, kind), cnt in sorted(loc.items()):
        for g, n in cnt.most_common():
            glue_rows += (f"<tr><td>{E(place)}</td><td>{E(kind)}</td><td style='color:{GLUE_COLOUR[g]}'><b>"
                          f"{E(GLUES[g].name)}</b></td><td>{n}</td><td>≈ {np.mean(wet[(place, g)]) * 1000:.0f} mg</td>"
                          f"<td>≤ {np.mean(dry[(place, g)]) * 1000:.0f} mg</td></tr>")
    used = [g for g in GLUES if g in set(res.joint_glue)]
    glue_tot = Counter()
    for j, g in enumerate(res.joint_glue):
        glue_tot[g] += res.mass.per_joint[j]
    glue_how = "".join(
        f"<tr><td><b style='color:{GLUE_COLOUR[g]}'>{E(GLUES[g].name)}</b><br><span class='s'>{E(GLUES[g].product)} — "
        f"{E(GLUES[g].where)}</span></td><td>{E(GLUE_HOW[g])}</td><td>{E(GLUES[g].cure)}</td>"
        f"<td>{res.joint_glue.count(g)} joints, {glue_tot[g]:.2f} g dry</td></tr>" for g in used)
    crit = [j for j, g in enumerate(res.joint_glue) if g == "epoxy"]
    crit_txt = ""
    if crit:
        crit_txt = ("<p class='warn'><b>Epoxy joints:</b> " + E(epoxy_summary(st, res, ids, range(len(st.joints)))) +
                    ". These carry the most load. The same glue is used on every matching joint all the way round, "
                    "because the heavier side in the real test could be anywhere.</p>")
    pva_only = analyse(st, spec, accurate=True, glue_key="pva")

    # ---- cut list ----------------------------------------------------------------------------------
    cut_rows = ""
    for (grp, ln, a0, a1, es), pis in sorted(groups.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
        guide = _chan_txt(guide_keys, pis)
        cut_rows += (f"<tr><td><b>{E(grp)}</b><br><span class='s'>{E(ids[pis[0]])}…{E(ids[pis[-1]])}</span></td>"
                     f"<td>{len(pis)}</td><td><b>{ln:.1f} mm</b></td><td>{E(_end_txt(a0, es[0]))}</td>"
                     f"<td>{E(_end_txt(a1, es[1]))}</td><td>{guide}</td><td>{_sand_tag(a0, es[0], a1, es[1])}</td></tr>")
    stick_rows = "".join(f"<li>Stick {k}: {E(', '.join(f'{lab} ({l:.1f})' for lab, l in b))}</li>"
                         for k, b in enumerate(plan.get("teacher-3x3", []), start=1))
    strip_ids = {ids[i] for i, p in enumerate(st.parts) if p.kind in ("diag", "leg")}
    strip_sticks = [k for k, b in enumerate(plan.get("teacher-3x3", []), start=1) if any(l in strip_ids for l, _ in b)]
    n_strip_sticks = max(strip_sticks) if strip_sticks else 0
    scarf = st.params.get("ring_joint") == "scarf"

    # ---- jig ------------------------------------------------------------------------------------------
    step_of = defaultdict(list)
    for n, stp in enumerate(steps, start=1):
        for f in stp.jig:
            step_of[f].append(n)
    for jp in jig_parts:
        if jp.name.startswith(("fit_test", "cutting_guide", "end_sanding", "scarf_sanding", "marking_gauge")):
            step_of[jp.name] = ["Step 0 (preparing parts)"]
    jig_rows = ""
    for jp in sorted(jig_parts, key=lambda j: _natural(getattr(j, "label", "") or j.name)):
        if jp.name.startswith("collar") and not jp.name.endswith("_seg1"):
            continue
        name = jp.name
        qty = jp.qty
        if name.startswith("collar"):
            c = name.split("_seg")[0]
            qty = sum(1 for j in jig_parts if j.name.startswith(c + "_seg"))
            name = f"{c}_seg1 … seg{qty}"
        orient = ("lying flat on the bed" if name.startswith("mast") else
                  "engraved ID facing up" if name.startswith("collar") else
                  "disc down (upside down)" if name.startswith("spacer") else
                  "wide flat bottom down" if name.startswith(("scarf", "end_sanding")) else "flat face down")
        used_in = step_of.get(jp.name, [])
        used_txt = ", ".join(str(u) for u in used_in) if used_in else "—"
        if used_in and all(isinstance(u, int) for u in used_in) and len(used_in) > 2:
            used_txt = f"steps {used_in[0]}–{used_in[-1]}"
        lab = jp.label if not jp.name.startswith("collar") else f"C{jp.name.split('_')[1]}-1 … -{qty}"
        jig_rows += (f"<tr><td><b class='tag'>{E(lab)}</b></td><td><code>{E(name)}.stl</code></td><td>{qty}</td>"
                     f"<td>{E(used_txt)}</td><td>{E(orient)}</td></tr>")
    c = cad.collar_heights(st)
    boss, stack = cad.jig_stack(st)
    stack_rows = ""
    for item in stack:
        k = item["k"]
        if item["kind"] == "spacer":
            stack_rows += (f"<tr><td>{_lab(f'spacer_{k}')}spacer_{k}.stl ({item['length']:.1f} mm tall, disc on top)</td>"
                           f"<td>{item['z0']:.1f} → {item['z1']:.1f} mm</td></tr>")
        else:
            nseg = sum(1 for j in jig_parts if j.name.startswith(f"collar_{k}_seg"))
            stack_rows += (f"<tr><td><b class='tag'>C{k}-1 … C{k}-{nseg}</b> collar_{k}_seg1 … seg{nseg}.stl, each on "
                           f"its number</td><td>{item['z0']:.1f} → {item['z1']:.1f} mm</td></tr>")
    n_jig_files = len(jig_parts)
    plates = print_plan(jig_parts, min(spec.bed[0], spec.bed[1]))
    plate_rows = "".join(f"<tr><td>{n}</td><td>{E(t)}</td><td>{E(', '.join(sorted(ids_, key=_natural)))}</td>"
                         f"<td>≈ {g:.0f} g, {max(0.25, g / 11):.1f} h</td></tr>"
                         for n, (t, ids_, g) in enumerate(plates, start=1))

    # ---- steps --------------------------------------------------------------------------------------
    done: set[int] = set()
    cum = 0.0
    step_html = ""
    prev_jig = None
    for n, stp in enumerate(steps, start=1):
        pset = set(stp.parts)
        js = [j for j, jt in enumerate(st.joints) if all(p in done | pset for p in jt.parts)
              and not all(p in done for p in jt.parts)] if pset else []
        cum += sum(res.mass.per_part[i] for i in stp.parts) + sum(res.mass.per_joint[j] for j in js)
        by = defaultdict(list)
        for j in js:
            by[(res.joint_glue[j], st.joints[j].kind)].append(j)
        glue_li = "".join(
            f"<li>{len(jj)} × {E(kind)} with <b style='color:{GLUE_COLOUR[g]}'>{E(GLUES[g].name)}</b>: apply ≈ "
            f"{np.mean([res.mass.per_joint_wet[j] for j in jj]) * 1000:.0f} mg each, dries to ≤ "
            f"{np.mean([res.mass.per_joint[j] for j in jj]) * 1000:.0f} mg</li>" for (g, kind), jj in sorted(by.items()))
        crit_here = epoxy_summary(st, res, ids, js) if any(res.joint_glue[j] == "epoxy" for j in js) else ""
        is_ring = bool(stp.parts) and all(st.parts[i].kind in ("ring", "base", "cap") for i in stp.parts)
        pic = (_img(stp.picture, "62%" if is_ring else "100%") if stp.picture
               else tower_svg(st, pset, done, height=260))
        ring_block = ""
        if is_ring:
            rsvg, rtable = ring_map(st, res, stp.parts, ids)
            if rsvg:
                ring_block = (f'<div class="cols"><div style="flex:0 0 48%"><p><b>Glue map of this ring '
                              f'(top view)</b></p>{rsvg}</div><div>{rtable}</div></div>')
        pieces = (f"{len(stp.parts)} pieces: {E(', '.join(sorted((ids[i] for i in stp.parts), key=_natural)))}"
                  if stp.parts else "none")
        jig_txt = ("the same as the step before: nothing to add or remove" if stp.jig and stp.jig == prev_jig else
                   _files_list(stp.jig) if stp.jig else "none (the jig comes off in this step)")
        prev_jig = stp.jig
        cure = ("<p class='warn'>⏸ <b>Wait at least 1 hour</b> before the next step so the glue can grab. Don't load "
                "or remove anything for 12 hours.</p>" if js and stp.key not in ("remove",) else "")
        step_html += f"""<div class="stepbox">
<h3>Step {n}. {E(stp.title)}</h3>
<div class="cols"><div style="flex:0 0 42%">{pic}<p class="s">3D model of this step: <code>{E(stp.model or '—')}</code>
(open in Onshape: blue = jig, tan = already built, orange = this step, coloured dots = glue)</p></div><div>
<p><b>Jig files on the bench:</b><br>{jig_txt}</p>
<p><b>Pieces added:</b> {pieces}</p>
{glue_legend(sorted({res.joint_glue[j] for j in getattr(stp, 'new_joints', [])}, key=list(GLUES).index), scarf and is_ring) if getattr(stp, 'new_joints', []) else ''}
{'<p><b>Glue in this step:</b></p><ul>' + glue_li + '</ul>' if glue_li else ''}
{'<p class="warn"><b>Epoxy joints here:</b> ' + E(crit_here) + '. Mix epoxy in small batches (enough for 3–4 joints at a time); it starts to set after about 5 minutes.</p>' if crit_here else ''}
</div></div>
{ring_block}
<p><b>Do this:</b></p><ol>{''.join(f'<li>{a}</li>' for a in stp.actions)}</ol>
{'<p><b>Check:</b></p><ul>' + ''.join(f'<li>☐ {E(c)}</li>' for c in stp.checks) + '</ul>' if stp.checks else ''}
{'<p><b>Common mistakes:</b></p><ul>' + ''.join(f'<li>{E(m)}</li>' for m in stp.mistakes) + '</ul>' if stp.mistakes else ''}
{cure}
{(f'<p class="check">⚖ Weigh after this step, once the glue is dry: no more than <b>{cum:.2f} g</b>. If it is heavier, you are using too much glue: wipe joints and use less on the next step.</p>' if js else f'<p class="check">⚖ Weigh these pieces before going on: together they should be no more than <b>{cum:.2f} g</b>. If they are heavier, swap in lighter sticks now.</p>') if stp.parts else ''}
</div>"""
        done |= pset

    # ---- step 0: preparing the parts ------------------------------------------------------------------
    stick_of = {}
    for k, b in enumerate(plan.get("teacher-3x3", []), start=1):
        for lab, _ in b:
            stick_of[lab] = k
    prep_rows = ""
    cups = []
    for (grp, ln, a0, a1, es), pis in sorted(groups.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
        sticks_used = sorted({stick_of.get(ids[i]) for i in pis if stick_of.get(ids[i])})
        st_txt = (f"{sticks_used[0]}–{sticks_used[-1]}" if len(sticks_used) > 1 else
                  (str(sticks_used[0]) if sticks_used else "—"))
        gmass = sum(res.mass.per_part[i] for i in pis)
        cups.append(f"{grp} ({ids[pis[0]]}–{ids[pis[-1]]}, {len(pis)})")
        prep_rows += (f"<tr><td>☐ <b>{E(grp)}</b><br><span class='s'>{E(ids[pis[0]])}…{E(ids[pis[-1]])}</span></td>"
                      f"<td><b>{len(pis)}</b></td><td><b>{np.mean([st.parts[i].length for i in pis]):.1f} mm</b></td>"
                      f"<td>{E(_end_txt(a0, es[0]))}</td><td>{E(_end_txt(a1, es[1]))}</td>"
                      f"<td>{_chan_txt(guide_keys, pis)}</td>"
                      f"<td>{_sand_tag(a0, es[0], a1, es[1])}</td><td>{st_txt}</td><td>≤ {gmass:.2f} g</td></tr>")
    has_scarf = any("scarf" in p.info.get("end_style", []) for p in st.parts)
    sj = [jp for jp in jig_parts if jp.name.startswith("end_sanding")]
    level = cad.level_parts(st)
    roll = sorted({st.parts[i].group for i in level
                   if guide_keys.get(i) and cad.needs_roll(st, i, guide_keys[i])})
    keep = sorted({st.parts[i].group for i in level} - set(roll))
    lab = lambda n: _lab(n) if n in LABELS else ""
    sj_li = "".join(f"<li>{_lab(jp.name)}<code>{E(jp.name)}.stl</code>: {E(jp.note)}</li>" for jp in sj)
    step0 = f"""<div class="stepbox"><h3>Step 0. Preparing the parts</h3>
<div class="cols"><div style="flex:0 0 46%">{_img(images.get('prep'))}<p class="s">The preparation jigs, each with its ID engraved.</p></div><div>
<p><b>Jigs for this step:</b></p><ul>
<li>{lab('fit_test_coupon')}<code>fit_test_coupon.stl</code>: finds the slot size your sticks need. Print it first.</li>
<li>{_lab('cutting_guide_1')}<code>cutting_guide_1.stl</code>: one channel per piece, each with a saw slot and a stop one piece-length away. Channel numbers and lengths are engraved beside each channel.</li>
{f"<li>{lab('marking_gauge')}<code>marking_gauge.stl</code>: marks where every crossing and ring goes on a strip, and checks its length.</li>" if level else ''}
{sj_li}
{f"<li>{_lab('scarf_sanding_guide')}<code>scarf_sanding_guide.stl</code>: makes the 1-in-3 scarf on every ring end. Two grooves: O UP and O DOWN.</li>" if has_scarf else ''}
</ul><p><b>Also:</b> razor saw, 240-grit sandpaper glued flat to a board, sharp pencil, 0.01 g scale, {len(cups)} small cups or bags.</p>
</div></div>
<p><b>Everything you need to make</b> ({res.n_parts} pieces from {n_sticks} sticks). Tick each row when its pieces are cut, sanded and labelled:</p>
<table class="tight"><tr><th>Piece</th><th>Qty</th><th>Length</th><th>End 1</th><th>End 2</th><th>Cut with</th><th>Sand with</th><th>From stick</th><th>Group mass</th></tr>{prep_rows}</table>
<p><b>Do this:</b></p><ol>
<li><b>Check the fit.</b> Slide a stick into each slot of {lab('fit_test_coupon')}. Find the smallest slot it slides into with light friction. If that isn't 3.3, set <b>Jig slot size</b> in the app to that number and export the build pack again before printing the other jigs.</li>
<li><b>Sort the sticks.</b> Weigh each stick and number them 1–{n_sticks} in pencil, heaviest first. The stick plan in section 9 puts the strips on sticks 1–{n_strip_sticks}.</li>
<li><b>Label {len(cups)} cups:</b> {E('; '.join(cups))}. Every piece goes straight into its cup.</li>
<li><b>Cut to length</b> in {_lab('cutting_guide_1')}(full method in section 9). Square up the stick's end at the slot first. Then for each piece: push the stick along until its end touches the stop, and saw at the slot. No measuring: the stop sets the length.</li>
{f'''<li><b>Letter the strips as they come off the guide.</b> Each strip lies in the channel with its foot at the left (the slot end).
{"".join(f"<li>{E(g)}: write <b>{E(g.split()[-1])}</b> and its number on the <b>top</b> face near the left end, then <b>F</b> at that end.</li>" for g in keep).join(("<ul>", "</ul>")) if keep else ""}
{"".join(f"<li>{E(g)}: <b>roll it over</b> (keep the left end on the left), then write <b>{E(g.split()[-1])}</b> and its number on the new top face near the left end, and <b>F</b> at that end.</li>" for g in roll).join(("<ul>", "</ul>")) if roll else ""}
The lettered face is the <b>letter face</b>: it faces outwards on the tower.</li>
<li><b>Mark the crossings</b> in {lab('marking_gauge')}. Lay each strip in its channel (A or B), letter face up, foot against the angled stop. Pencil a tick through every notch onto the letter face and carry it round onto both side faces. Copy the number beside the notch (1, 2, 3R …; R means a ring goes there). The strip's top end should line up with the END notch, within 1 mm.</li>
<li><b>Sand the strip ends in {_lab(sj[0].name) if sj else ""}.</b> Put the jig on the sandpaper board, letter face of the strip towards the side marked LETTER FACE THIS SIDE. Push the foot (F) end down the slot labelled with its letter and FOOT (for example A FOOT) until it touches the sandpaper. Pinch the strip to the jig so it can't slide, and rub in small strokes. Stop as soon as the plastic all round the slot shows sanding marks: then the end is flush and exact. Do the top end the same way in the matching TOP slot. Check: the strip stands on the table at its lean without rocking. Never push the strip further down while sanding; that only makes it shorter.</li>''' if level else ''}
{f"<li><b>Scarf the ring ends in {_lab('scarf_sanding_guide')}.</b> Pencil an <b>O</b> on one face of each ring piece (its outer face). End 1: lay the piece in groove <b>O UP</b> (O face upwards) with its end 1 mm past the thin front edge, press it down with a finger and rub the guide on the sandpaper board until the plastic either side starts to scuff. End 2: the same in groove <b>O DOWN</b> (O face downwards). Pencil a small 2 next to end 2. Check: two pieces laid together, O faces up, overlap on their slopes with no light between them.</li>" if has_scarf else ''}
<li><b>Count and check.</b> Count each cup against the table (cut 2–3 spare ring pieces of each size). Weigh each cup: it should be no more than the group mass shown, or swap in pieces from lighter sticks.</li></ol>
<p class="check">⚖ All pieces together should weigh no more than <b>{sum(res.mass.per_part):.2f} g</b> before any glue.</p></div>"""

    # ---- appendix: full parts list ------------------------------------------------------------------
    end_glue = defaultdict(list)
    for j, jt in enumerate(st.joints):
        for p in (jt.parts[0], physical_host(st, j)):
            end_glue[p].append(GLUES[res.joint_glue[j]].name)
    part_rows = ""
    for i in sorted(range(len(st.parts)), key=lambda i: _natural(ids[i])):
        p = st.parts[i]
        if p.sec is None or p.kind == "corner":
            continue
        es = p.info.get("end_style", ["", ""])
        g = Counter(end_glue[i])
        part_rows += (f"<tr><td>{E(ids[i])}</td><td>{E(p.group or p.name)}</td><td>{p.length:.1f}</td>"
                      f"<td>{E(_end_txt(p.end_angles[0], es[0]))}</td><td>{E(_end_txt(p.end_angles[1], es[1]))}</td>"
                      f"<td>{E(', '.join(f'{v}× {k}' for k, v in g.items()))}</td><td>{res.mass.per_part[i] * 1000:.0f}</td></tr>")
    file_rows = "".join(f"<tr><td><code>{E(f)}.stl / .step</code></td><td>{E(', '.join(str(x) for x in v))}</td></tr>"
                        for f, v in sorted(step_of.items(), key=lambda kv: _natural(kv[0])))
    step_models = "".join(f"<tr><td>Step {n}</td><td>{E(s.title)}</td><td><code>{E(s.model)}</code></td></tr>"
                          for n, s in enumerate(steps, start=1) if s.model)

    # ---- page ----------------------------------------------------------------------------------------
    twist_deg = info.get("twist_k", 0) * 180 / max(info.get("n", 1), 1)
    from .structure import bore_clearance
    pole_gap = min(bore_clearance(st.nodes[m.i], st.nodes[m.j], m.sec.half_extent) for m in st.members) - spec.pole_d / 2
    strip_grp = [k for k in groups if k[0] in ("Strip A", "Strip B")]
    lean_angle = strip_grp[0][2] if strip_grp else 80
    ring_grp = [k for k in groups if k[0].startswith("Ring") or k[0] in ("Base ring", "Top cap")]
    lap_angle = ring_grp[0][2] if ring_grp else 80
    geometry = ""
    if diag:
        geometry = (f"<p>{info['n']} strips twist one way (<b>set A</b>, the inner layer) and {info['n']} twist the "
                    f"other way (<b>set B</b>, the outer layer). Each strip turns through {twist_deg:.0f}° on its "
                    f"way up. The base is Ø{2 * info['base_r']:.0f} mm and the top Ø{2 * info['top_r']:.0f} mm, "
                    f"measured to the strip centres, so the tower narrows like a cooling tower. Feet are "
                    f"{360 / info['n']:.1f}° apart round the base. The strips cross at {len(st.levels) - 2} levels: "
                    f"{', '.join(f'{z:.0f}' for z in st.levels[1:-1])} mm above the table. Rings sit at some of "
                    f"those levels. The printed jig sets every one of these positions for you.</p>")
    util_svg = tower_svg(st, util=res.util, height=300)
    if st.params.get("ring_joint") == "scarf":
        ring_how = ("Neighbouring pieces <b>overlap each other on a 9 mm diagonal (scarf)</b> right over each B strip: one "
                    "piece lies flat on the strip and the next lies on top of it. Glue goes on the strip patch and on "
                    "the diagonal faces, so each ring is one continuous hoop.")
    else:
        ring_how = ("Two pieces meet end to end on each B strip, each covering half its width, and each end is glued (dots).")
    toc = ["How to use this guide", "How the tower works (2b: how the rings attach)", "Safety", "What you need", "Before you start: test your wood and glue",
           "Wood and grain direction", "Glue plan", "Printing the jig", "Cutting the pieces", "Build steps",
           "Final checks", "Test day", "Troubleshooting and fixing mistakes", "Appendix A: every piece",
           "Appendix B: jig and model files"]
    css = """
@page{size:A4;margin:14mm 13mm 16mm}
body{font:10.3pt/1.45 -apple-system,Helvetica,Arial,sans-serif;color:#222;margin:0}
h1{font-size:24pt;margin:0 0 4px}h2{font-size:14pt;border-bottom:2px solid #e8590c;padding-bottom:3px;margin:20px 0 8px;page-break-after:avoid}
h3{font-size:12pt;margin:2px 0 6px;color:#9c3d07}table{border-collapse:collapse;width:100%;font-size:8.8pt;margin:6px 0}
td,th{border:1px solid #ccc;padding:3px 5px;text-align:left;vertical-align:top}th{background:#f6f1ea}
tr{page-break-inside:avoid}.s{color:#666;font-size:8pt}.warn{background:#fff3cd;padding:6px 8px;border-radius:4px}
.check{background:#e7f5ff;padding:5px 8px;border-radius:4px}.kpi{font-size:20pt;font-weight:700;color:#e8590c}
table.tight td,table.tight th{padding:0 4px;font-size:7.4pt;line-height:1.25}.tag{background:#1f2937;color:#fff;border-radius:3px;padding:0 4px;font-size:8.5pt}.legend{background:#f8f9fa;border:1px solid #ddd;border-radius:4px;padding:4px 8px}.cols{display:flex;gap:16px}.cols>div{flex:1}.pb{page-break-before:always}.figrow{display:flex;gap:10px;align-items:flex-start;page-break-inside:avoid}
.stepbox{page-break-before:always}code{font-size:8.5pt;background:#f3f3f3;padding:0 3px;border-radius:2px}
ul,ol{margin:4px 0 6px 18px;padding:0}li{margin-bottom:3px}.toc li{margin:1px 0}"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Tower assembly guide</title><style>{css}</style></head><body>
<h1>{E(fam)}</h1><p style="font-size:13pt;margin:0">Assembly guide</p>
<p class="s">Generated by Tower Designer. Brief: ≤ {spec.max_mass:.0f} g, {st.height:.0f} mm tall, fits over a {spec.pole_d:.0f} mm
pole, built only from supplied 3 × 3 mm sticks (full size; only the joints are shaped).</p>
<div class="cols"><div>{_img(images.get('tower'))}</div><div>
<p>Predicted failure load<br><span class="kpi">{res.load_kg:.0f} kg</span></p>
<p>Predicted mass<br><span class="kpi">{res.mass.total:.2f} g</span><br><span class="s">wood {res.mass.wood:.2f} g + glue
{res.mass.glue:.2f} g, including +3 % on every piece and joint</span></p>
<p><b>{res.n_parts} pieces · {res.n_joints} glue joints · {n_sticks} sticks</b><br><span class="s">(assuming {stick_len:.0f} mm sticks)</span></p>
<p><b>Weakest point:</b> {E(res.governing)}.<br>Gap to the pole: {pole_gap:.1f} mm all round.</p>
<p class="warn">{"Strength is based on your wood test." if measured else "Strength is based on <b>book values for medium balsa</b> because the stick species is unknown. Do the wood test first (section 5)."}</p>
<p><b>Contents</b></p><ol class="toc s">{''.join(f'<li>{E(t)}</li>' for t in toc)}</ol>
</div></div>

<h2 class="pb">1. How to use this guide</h2>
<ul>
<li>Every piece has an ID: A1–A{len(by_grp.get('Strip A', [])) or 19} and B1–B{len(by_grp.get('Strip B', [])) or 19} for the strips, R1-1, R3-1 … for the rings (ring number, then piece), BASE…, CAP… . Write it on the piece in pencil when you cut it.</li>
<li>Every build step has a picture <b>and a 3D model</b> in the <code>steps/</code> folder. Open the models in Onshape
(<b>+ → Import</b>) and orbit round them if a picture isn't clear. In every model the <b>jig is blue</b>, pieces you have
<b>already built are tan</b>, and <b>the pieces you add in that step are orange</b>.</li>
<li>Every step lists the <b>jig parts</b> it uses by their engraved ID (dark tag, e.g. <b class="tag">SP2</b>) and file name in <code>jig/</code>.</li>
<li><b>Glue colour code</b>, the same in every picture, model, table and step: each glue point is a coloured mark.
{glue_legend(used, scarf)}</li>
<li>⚖ boxes are weigh-ins. Weigh the tower after each step once its glue is dry. If you are over, you are using too much glue.</li>
<li>☐ boxes are checks to tick before you move on. ⏸ boxes mean wait for the glue.</li></ul>
<p><b>Words used in this guide</b></p>
<table class="tight">{''.join(f'<tr><td style="width:22%"><b>{E(a)}</b></td><td>{E(b)}</td></tr>' for a, b in GLOSSARY)}</table>

<h2>2. How the tower works</h2>
<div class="cols"><div style="flex:0 0 30%">{util_svg}<p class="s">Colour = how hard each piece works at the predicted
failure load (green light → red at its limit).</p></div><div>
{geometry}
<p><b>Load path.</b> The load plate presses on the top ring and the strip tops. Each strip carries its share straight down
to the table. The strips lean in opposite directions, so where an A strip crosses a B strip they brace each other. That
shortens the length that can buckle and turns the whole wall into a stiff lattice shell, like a woven basket.</p>
<p><b>Why it narrows.</b> A wider base resists tipping and buckling of the whole tower. The narrower top keeps the
strips short and steep, so they stay in compression.</p>
<p><b>What the rings do.</b> They hold the crossings at their levels so the lattice keeps its shape, and stop the strips
spreading outwards at the base and squeezing inwards at the top.</p>
<p><b>Why lap joints.</b> Every joint is one face glued flat on another (side grain to side grain). That is far
stronger than gluing onto a cut end.</p>
<p><b>How it is predicted to fail.</b> {E(res.governing)}. The whole tower would buckle at about
{res.lam_global / 9.81:.0f} kg, well above the failure load.</p></div></div>

<h2 class="pb">2b. How the rings attach, and why nothing hits the jig</h2>
<div class="cols"><div>{svg_layers()}</div><div><p>The tower is built in <b>three layers</b>, from the inside out:
the A strips, then the B strips glued over them at every crossing, then the rings glued on the outside of the B strips.
Each layer is one 3 mm stick thick, and every joint is a face glued flat on a face.</p></div></div>
<div class="cols">
<div>{_img(images.get('detail_crossing'))}<p class="s"><b>A crossing.</b> The B strip lies over the A strip; glue goes where they touch (dot).</p></div>
<div>{_img(images.get('detail_ring'))}<p class="s"><b>A ring at a crossing level.</b> Ring pieces (gold) lie on the B strips. {ring_how}</p></div></div>
<div class="cols">
<div>{_img(images.get('detail_base'))}<p class="s"><b>The base ring in the jig.</b> It lies on the outside of the feet, in a shallow groove in the
base plate (blue) that keeps it 0.5 mm clear of the plastic.</p></div>
<div>{_img(images.get('detail_cap'))}<p class="s"><b>The top ring.</b> Its top is flush with the strip tops so the load plate presses on both.</p></div></div>
<div class="cols"><div>{svg_base_section()}</div><div>
<p><b>Why the base ring doesn't hit the jig:</b></p><ul>
<li>The strip feet stand in 2 mm-deep pockets in the base plate. The base ring sits just above the feet on the outside, in its own
groove, with 0.5 mm of air underneath, so it is never glued to or trapped by the plate.</li>
<li>The collars hold the strips from the <b>inside</b>; every ring is on the <b>outside</b>. The lowest collar starts
{(cad.collar_heights(st) or [0])[0] - cad.COLLAR_T / 2:.0f} mm up, well above the base ring.</li>
<li>Collars sit between the ring levels, never at the same height.</li>
<li>Checked in the 3D model: no jig part overlaps any piece of the tower anywhere.</li>
<li>When you lift the finished tower off, the feet come straight up out of their pockets and the base ring out of its groove.</li></ul>
</div></div>

<h2>3. Safety</h2><ul>
<li>Wear safety glasses when cutting, sanding, and crush-testing (wood splinters).</li>
<li>Glue: work with a window open.{" Balsa cement gives off solvent fumes and is flammable: keep it away from flames and put the lid back on between joints." if "cement" in used else ""}{" Superglue gives off fumes and bonds skin in seconds." if "ca" in used else ""}{" Wear disposable gloves for epoxy; wash skin with soap if you get any on you." if "epoxy" in used else ""}{" Acetone (nail-polish remover, for fixing mistakes) is flammable too." if "cement" in used else ""}</li>
<li>Razor saws and hobby knives: cut away from your fingers, on a cutting mat, with the stick held in the guide.</li>
<li>Let the 3D printer cool before removing prints.</li></ul>

<h2>4. What you need</h2>
<div class="cols"><div><p><b>Materials</b></p><ul>
<li><b>{n_sticks + 2} × 3 × 3 mm sticks</b>: {n_sticks} plus 2 spare (stick plan in section 9).</li>
{''.join(f"<li>{E(GLUES[g].product)} — {E(GLUES[g].where)}</li>" for g in used)}
<li>PLA filament for the jig, about 150 g.</li>
<li>Packing tape, masking tape, baking paper.</li>
<li>Acetone (nail-polish remover) and cotton buds, for undoing a wrong joint.</li></ul></div><div><p><b>Tools</b></p><ul>
<li>Fine razor saw (or a new hobby-knife blade) and a cutting mat.</li>
<li>240-grit sandpaper glued flat to a board, plus a small sanding stick.</li>
<li>Scale reading to 0.01 g. Ruler. Calipers if you have them.</li>
<li>A small brush or toothpicks{", small cups (one per glue)" if len(used) > 1 else ""}, scrap card, small spring pegs, sharp pencil, fine marker.</li>
<li>{len(groups)} small cups or bags for sorting the pieces.</li>
<li>Bambu printer (jig) and the app on the Mac.</li></ul></div></div>

<h2>5. Before you start: test your wood and glue</h2>
<p>The prediction above uses {"your measured wood" if measured else "book values"} and a standard glue allowance.
Two quick tests make it match <b>your</b> sticks and <b>your</b> gluing. Both kits are in the <code>test kits/</code> folder of this build pack.</p>
<div class="cols"><div><p><b>Wood test</b> (<code>Wood test kit/</code>, about 30 min)</p><ol>
<li>Weigh 5 whole sticks: this gives the density.</li>
<li>Crush 5 pieces 12 mm long, ends sanded square in <code>wood_test_squaring_block.stl</code>, between
<code>wood_test_crush_pad.stl</code> pads on a 25 kg scale. Record each peak.</li>
<li>Press 3 straight 150 mm pieces in <code>wood_test_buckling_cap.stl</code> caps until they bow, and record the load.</li>
<li>Enter the results in the app (Wood and glue → Wood test…) and click Apply.</li></ol></div><div>
<p><b>Glue test</b> (<code>Glue test kit/</code>, a day for curing)</p><ol>
<li>For each glue, weigh a 100 mm rod and 5 short strips together.</li>
<li>Glue the strips onto the rod in <code>glue_test_jig_base.stl</code> + <code>glue_test_jig_comb.stl</code>, exactly as
you'll glue the tower.</li>
<li>When dry, weigh again. Enter the masses in the app (Glue test…) and click Apply.</li></ol></div></div>
<p class="warn">After either test, <b>re-run the search and export the build pack again</b>. The design, cut list, and
this guide may change.</p>

<h2 class="pb">6. Wood and grain direction</h2>
<div class="figrow"><div>{svg_grain()}</div><div>
<p>The grain (the wood fibres) runs along the length of every 3 × 3 stick. Wood is strong and stiff along the grain:
about <b>20–30× stiffer</b> than across it, and that is the direction every piece of this tower carries its load.
<b>Never split or rip a stick lengthwise.</b> Only cut across it, so the grain always runs along each piece.</p></div></div>
<div class="figrow"><div>{svg_runout()}</div><div><p><b>Pick straight-grained sticks.</b> Hold each stick to the light and
look at all four faces. Grain lines should run parallel to the edges. If they run off an edge ("run-out"), the stick
will snap there. Use those sticks for short ring pieces only, cutting round the run-out.</p></div></div>
<p><b>Sort the sticks before cutting:</b></p><ol>
<li>Weigh every stick and write its mass on it. Mass ÷ volume (length × 9 mm²) gives the density; book balsa is about
{w.density:.0f} kg/m³.</li>
<li>Rank them heaviest to lightest. Heavier (denser) sticks are stiffer and stronger.</li>
<li>Use the <b>heaviest {n_strip_sticks} sticks</b> (sticks 1–{n_strip_sticks} in the stick plan) for the
{'strips' if diag else 'legs'}. They carry the load.</li>
<li>Use the lightest sticks for ring pieces, which carry little load.</li>
<li>Reject any stick that is bent, cracked, or much lighter than the others.</li></ol>
<p><b>At the joints:</b> glue side grain to side grain. A cut end (end grain) soaks up glue and bonds at only about 60 % of
the strength, which is why this design laps every joint face-on-face. Keep sticks dry and flat; don't leave them in
the sun or a damp room before building.</p>

<h2>7. Glue plan</h2>
{glue_legend(used)}
<div class="cols"><div style="flex:0 0 36%">{_img(images.get('glue_map'))}<p class="s">Glue map: every glue point on the tower,
coloured by the glue to use. Each step's picture and 3D model shows the same dots.</p></div><div>
<p>{"Each joint gets the lightest glue that is still strong enough for the load it carries." if len(used) > 1 else f"Every joint uses {E(GLUES[used[0]].name)}."} "Apply" is roughly what to put
on; "dries to" is the most that should stay in the joint. A pinhead dab is about 2–4 mg.</p>
<table><tr><th>Where</th><th>Joint</th><th>Glue</th><th>Count</th><th>Apply (wet)</th><th>Dries to</th></tr>{glue_rows}</table>
{crit_txt}</div></div>
<p><b>How to use each glue</b></p>
<table><tr><th>Glue</th><th>Method</th><th>Cure</th><th>In this tower</th></tr>{glue_how}</table>
<p><b>Preparing a joint:</b> make sure the two faces touch flat with no light showing between them. If they don't, touch
the contact area (only) once on the sandpaper. Blow off dust before gluing. Don't fill gaps with extra glue; fix the fit instead.</p>
<p class="s">If you'd rather not use balsa cement: With PVA on every joint, this tower would weigh
{pva_only.mass.total:.2f} g and hold {pva_only.load_kg:.0f} kg{' — over the limit. To switch, set Glue to PVA only in the app, re-run the search, and export again' if pva_only.mass.total > spec.max_mass else ''}.</p>

<h2 class="pb">8. Printing the jig</h2>
<div class="cols"><div style="flex:0 0 40%">{_img(images.get('jig'))}</div><div>
<p>All {n_jig_files} files are in <code>jig/</code> as <b>STL</b> (for Bambu Studio) and <b>STEP</b> (for Onshape).
<code>jig/jig_assembly.step</code> shows the whole jig assembled.</p>
<p><b>Print settings (PLA):</b> 0.2 mm layers, 3 walls, 15 % infill, no supports, brim off. Print each part in the
orientation in the table below: collar segments with the engraved ID facing up (the notches lean, so upside down is wrong),
spacers disc-down, the mast lying flat.</p>
<p><b>Print in this order</b> (rough times on a Bambu A1/P1):</p>
<table class="tight"><tr><th>Plate</th><th>What</th><th>Parts</th><th>About</th></tr>{plate_rows}</table>
<p><b>After printing:</b></p><ol>
<li>Print <b class="tag">FT</b> first and check your sticks fit (Step 0, item 1) before printing anything else.</li>
<li>Remove any stringing from the notches and slots with a knife.</li>
<li>Check every part's engraved ID is readable (C1-1, C1-2 …, SP1 …); go over faint ones with a fine marker.</li>
<li>Stack the jig dry on the mast (Step 1) once before you start, to check every segment sits on its number.</li></ol></div></div>
<p>Every part has its <b>ID engraved</b> on it (the bold tag below). The guide uses these IDs everywhere.</p>
<table><tr><th>ID</th><th>File</th><th>Qty</th><th>Used in</th><th>Print orientation</th></tr>{jig_rows}</table>
<p><b>Stack on the mast, bottom to top</b> (heights above the table):</p>
<table><tr><th>Part</th><th>Sits at</th></tr><tr><td>{_lab('base_plate')}base_plate.stl boss</td><td>up to {boss:.1f} mm</td></tr>{stack_rows}</table>

<h2 class="pb">9. Cutting the pieces</h2>
<p>Every piece is a full 3 × 3 stick cut to length; only the ends are shaped. The cutting guide {_lab('cutting_guide_1')}has one
channel per piece, numbered and labelled with its length. Each channel is open at the left, has a saw slot near the left end and a
<b>stop</b> exactly one piece-length to the right of the slot. You never measure.</p><ol>
<li>Tape the guide to the bench. Pick the channel for the piece (table below).</li>
<li><b>Square up the start.</b> Lay the stick in the channel from the left so its end pokes about 5 mm past the slot, and saw
through the slot. Throw the offcut away.</li>
<li><b>Cut a piece.</b> Slide the stick to the right until its fresh end touches the stop. Hold it there with a finger well away
from the slot, and saw gently through the slot. The piece between the slot and the stop is exactly the right length, and the
end left on the stick is already cut for the next piece.</li>
<li>Repeat step 3 along the stick. <b>Don't turn the stick over between cuts</b>, so the angled strip ends stay parallel.</li>
<li>Write the piece ID on it in pencil and drop it in its cup. Strips also get their letter and F (Step 0).</li></ol>
<table><tr><th>Piece</th><th>Qty</th><th>Length</th><th>End 1</th><th>End 2</th><th>Cut with</th><th>Sand ends with</th></tr>{cut_rows}</table>
<div class="figrow"><div>{svg_level_cut(lean_angle)}</div><div>{svg_scarf() if st.params.get("ring_joint") == "scarf" else svg_lap(lap_angle)}</div><div>{svg_crossing()}</div></div>
<ul>
<li><b>Level cuts (strip ends):</b> the strips lean, so both ends are cut at the same angle and parallel to each other.
The cutting guide gets the angle close. The sanding jig {"".join(_lab(j.name) for j in jig_parts if j.name.startswith("end_sanding"))}then makes it exact, including the slight inward tilt of the tower.</li>
<li><b>Ring ends:</b> {ring_how} {"Cut the ring pieces square, then sand each scarf in " + _lab('scarf_sanding_guide') + "(Step 0): end 1 in groove O UP, end 2 in groove O DOWN, so the overlaps stack." if st.params.get("ring_joint") == "scarf" else "Trim the end parallel to the strip so the two pieces butt neatly."}</li>
<li><b>Crossings:</b> no cutting at all. The faces touch where B crosses A; you only glue there.</li>
<li>Cut 2 or 3 spares of each ring length. Ring pieces are tiny and easy to lose or break.</li></ul>
<p><b>Stick plan</b> (which pieces come from which stick, in mm, allowing 5 mm to square up the start and 2 mm per saw cut). Cut the strips from the heaviest sticks:</p>
<ul class="s">{stick_rows}</ul>

<h2 class="pb">10. Build steps</h2>
<p>Before Step 1: all pieces cut and labelled, jig printed and test-fitted, glues and tools laid out, a clear flat bench.</p>
<p><b>A sensible plan</b> (each session 1–2 hours):</p>
<table class="tight"><tr><th>Session</th><th>Do</th><th>Then wait</th></tr>
<tr><td>1</td><td>Step 0: test fit, cut, letter, mark and sand every piece</td><td>—</td></tr>
<tr><td>2</td><td>Steps 1–2 (jig and A strips, no glue), then the first half of Step 3</td><td>1 h between batches</td></tr>
<tr><td>3</td><td>The rest of Step 3 (crossings)</td><td>12 h</td></tr>
<tr><td>4</td><td>Base ring and the first ring</td><td>1 h</td></tr>
<tr><td>5</td><td>The other rings and the top ring</td><td>24 h</td></tr>
<tr><td>6</td><td>Step 9: remove the jig, finish, weigh</td><td>—</td></tr></table>
{step0}
{step_html}

<h2 class="pb">11. Final checks</h2><ul>
<li>☐ Mass is no more than <b>{res.mass.total:.2f} g</b> (limit {spec.max_mass:.0f} g).</li>
<li>☐ It stands on a flat table without rocking; every foot touches.</li>
<li>☐ The top is flat: lay a ruler across it in three directions and check for gaps.</li>
<li>☐ No joint has a visible gap or loose piece. Press gently on each ring; nothing moves.</li>
<li>☐ The pole slides through without touching (about {pole_gap:.0f} mm gap).</li>
{"<li>☐ Every epoxy joint is solid.</li>" if crit else ""}
<li>☐ No masking tape or jig parts left inside.</li></ul>

<h2>12. Test day</h2><ol>
<li>Weigh the tower in front of the tester and record it.</li>
<li>Slide it over the pole, centred. Check the gap all round.</li>
<li>Seat the load plate gently and check it touches the top ring and strip tops all the way round. Don't shim; sand if needed.</li>
<li>Add load smoothly and evenly. Shock loads break brittle joints first.</li>
<li>Record the load at failure and where it failed (which piece or joint). Compare it with the prediction ({res.load_kg:.0f} kg) and the weakest point listed above.</li></ol>
<table><tr><th>Tower mass (g)</th><th>Failure load (kg)</th><th>Where it failed</th><th>Efficiency (load ÷ mass)</th></tr>
<tr><td style="height:24px"></td><td></td><td></td><td></td></tr></table>

<h2>13. Troubleshooting</h2>
<table><tr><th>Problem</th><th>Likely cause</th><th>Fix</th></tr>
<tr><td>Strips won't sit in the collar notches</td><td>A segment on the wrong number, upside down, or spacers swapped</td><td>Check each segment sits on its own number with its ID on top, and each spacer's disc says "under" the collar above it.</td></tr>
<tr><td>A crossing doesn't touch (gap)</td><td>Strip not fully seated, or bowed</td><td>Re-seat the strip in its notches. Swap bowed strips for straight ones. Never fill a gap with glue.</td></tr>
<tr><td>Over mass at a weigh-in</td><td>Too much glue</td><td>Use less on the remaining joints; wipe squeeze-out immediately. Do the glue test.</td></tr>
<tr><td>Glue stuck the tower to the jig</td><td>Glue ran onto the plastic</td><td>Free it with a thin blade; sand the joint smooth. Keep packing tape on the base plate.</td></tr>
<tr><td>Ring not level</td><td>Pieces glued at slightly different heights</td><td>Measure each piece from the table before the glue sets.</td></tr>
<tr><td>Collar segment won't come out</td><td>Glue on the notch</td><td>Push it inwards firmly; if stuck, cut the segment with a knife. It is only a jig part.</td></tr>
<tr><td>Tower rocks on the table</td><td>Feet not level</td><td>Short straight strokes on the sandpaper board with a book on top (Step 9). Never circles.</td></tr>
<tr><td>Balsa cement joint stays soft</td><td>Single coat, or not cured</td><td>Use the double-glue method and wait 12 h.</td></tr></table>
<p><b>Fixing mistakes</b></p>
<table><tr><th>What went wrong</th><th>What to do</th></tr>
<tr><td>A strip breaks before it's glued</td><td>Cut a new one from a spare stick in channel 1, letter it, mark it in the marking gauge and sand its ends, exactly like the others.</td></tr>
<tr><td>A glued piece breaks, or a joint is glued in the wrong place</td><td>{"Balsa cement softens with acetone: dab a little on the joint with a cotton bud, wait a minute, and ease the pieces apart. Scrape off the old glue, let it dry, and glue again." if "cement" in used else "Cut through the joint with a fresh blade, sand both faces clean and glue again."} Replace a broken piece with a spare.</td></tr>
<tr><td>A scarf sanded too far (the piece is now too short to overlap)</td><td>Use one of the spare ring pieces.</td></tr>
<tr><td>A strip end sanded too far (strip short of the END notch)</td><td>Over 1 mm short: cut a new strip. The tops must be level for the load plate.</td></tr>
<tr><td>You run out of sticks</td><td>You have 2 spare; the offcuts at the end of the strip sticks are long enough for ring pieces.</td></tr>
<tr><td>A jig part is lost or broken</td><td>Reprint just that file; its name is in the table in section 8 and Appendix B.</td></tr>
<tr><td>Over mass at the end</td><td>Don't sand pieces thinner. Look for big glue blobs and scrape them off with a blade.</td></tr></table>

<h2 class="pb">Appendix A: every piece</h2>
<table><tr><th>ID</th><th>Group</th><th>Length mm</th><th>End 1</th><th>End 2</th><th>Glue at its joints</th><th>Max mg</th></tr>{part_rows}</table>

<h2 class="pb">Appendix B: jig and model files</h2>
<p><b>Jig files</b> (<code>jig/</code>; each has a .stl to print and a .step for Onshape) and the steps that use them:</p>
<table><tr><th>File</th><th>Used in step</th></tr>{file_rows}</table>
<p><b>Step models</b> (<code>steps/</code>, open in Onshape):</p>
<table><tr><th>Step</th><th>What</th><th>File</th></tr>{step_models}</table>
<p><b>Whole tower:</b> <code>tower.step</code>. <b>Assembled jig:</b> <code>jig/jig_assembly.step</code>.</p>
<p><b>Importing into Onshape:</b> in a document click <b>+ → Import…</b> and choose the file. STEP files come in as proper
editable parts; STL files come in as meshes (fine for viewing, awkward to edit).</p>

</body></html>"""


def write_pdf(html_path: str, pdf_path: str) -> bool:
    """Print the HTML to PDF with headless Chrome (installed on this Mac)."""
    chrome = CHROME if os.path.exists(CHROME) else shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        return False
    try:
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf_path}", "file://" + os.path.abspath(html_path)],
                       check=True, capture_output=True, timeout=180)
        return os.path.exists(pdf_path)
    except (subprocess.SubprocessError, OSError):
        return False
