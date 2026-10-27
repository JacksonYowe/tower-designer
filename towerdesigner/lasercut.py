"""Laser-cut files: every laser part nested on real Bunnings sheets, grain along the sheet.

Pieces are never rotated (except 180°) so the grain direction chosen by the analysis is kept.
Output per sheet: DXF (layers CUT / LABELS / SHEET) and SVG (red = cut, blue = labels only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .materials import SHEETS, STOCK_BY_ID, Stock
from .structure import Structure

EDGE = 3.0    # mm margin from sheet edges
GAP = 2.0     # mm between pieces


@dataclass
class Piece:
    part: int
    label: str
    rings: list[np.ndarray]   # first = outside, rest = holes; grain along +x; bbox min at 0,0
    t: float

    @property
    def size(self) -> tuple[float, float]:
        o = self.rings[0]
        return float(o[:, 0].max()), float(o[:, 1].max())


@dataclass
class Sheet:
    stock: Stock
    placed: list[tuple[Piece, float, float]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.stock.t:g}mm_x{self.stock.w:g}"


def part_ids(st: Structure) -> list[str]:
    """Short labels like L1, A7 (strip set A), R3-2 (ring 3), CAP4, BASE1, P3, G12."""
    prefix = {"leg": "L", "brace": "B", "ring": "R", "cap": "CAP", "base": "BASE", "panel": "P",
              "diag": "S", "gusset": "G", "corner": "K"}
    count: dict[str, int] = {}
    out = []
    for p in st.parts:
        k = prefix.get(p.kind, "X")
        g = p.group or ""
        if p.kind == "diag" and g.startswith("Strip "):     # diagrid strips: A1…, B1… (set + number)
            k = g.split()[1]
        elif p.kind == "ring" and g.startswith("Ring ") and g.split()[1].isdigit():
            k = f"R{g.split()[1]}-"                          # ring pieces: R3-1… (ring + number)
        count[k] = count.get(k, 0) + 1
        out.append(f"{k}{count[k]}")
    return out


def _wire_points(wire) -> np.ndarray:
    from OCP.BRep import BRep_Tool
    from OCP.BRepTools import BRepTools_WireExplorer
    exp = BRepTools_WireExplorer(wire.wrapped)
    pts = []
    while exp.More():
        p = BRep_Tool.Pnt_s(exp.CurrentVertex())
        pts.append((p.X(), p.Y()))
        exp.Next()
    return np.array(pts)


def _panel_rings(part) -> list[list[np.ndarray]]:
    import cadquery as cq
    faces = []
    for a, b, w in part.info["rects"]:
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        u = d / L
        nrm = np.array([-u[1], u[0]]) * (w / 2)
        ext = u * (w / 2)  # extend past the node so strips overlap cleanly
        pts = [a - ext - nrm, b + ext - nrm, b + ext + nrm, a - ext + nrm]
        faces.append(cq.Face.makeFromWires(cq.Wire.makePolygon([cq.Vector(x, y, 0) for x, y in pts], close=True)))
    fused = faces[0].fuse(*faces[1:]).clean()
    out = []
    for f in fused.Faces():
        out.append([_wire_points(f.outerWire())] + [_wire_points(w) for w in f.innerWires()])
    return out


def collect_pieces(st: Structure) -> list[Piece]:
    ids = part_ids(st)
    pieces = []
    for pi, p in enumerate(st.parts):
        if not p.laser or p.kind == "corner":
            continue
        groups = _panel_rings(p) if p.kind == "panel" else (p.outline or [])
        t = p.info.get("t") or (p.sec.t if p.sec else 1.5)
        for k, rings in enumerate(groups):
            lo = rings[0].min(axis=0)
            rr = [r - lo for r in rings]
            lab = ids[pi] if len(groups) == 1 else f"{ids[pi]}{'abcdefgh'[k]}"
            pieces.append(Piece(pi, lab, rr, float(t)))
    return pieces


def nest(pieces: list[Piece]) -> tuple[list[Sheet], list[str]]:
    sheets: list[Sheet] = []
    problems = []
    by_t: dict[float, list[Piece]] = {}
    for pc in pieces:
        by_t.setdefault(pc.t, []).append(pc)
    for t, pcs in sorted(by_t.items()):
        stocks = sorted([s for s in SHEETS if abs(s.t - t) < 1e-6], key=lambda s: s.w)
        if not stocks:
            problems.append(f"No {t} mm sheet sold")
            continue
        pcs = sorted(pcs, key=lambda p: (-p.size[1], -p.size[0]))
        open_sheets: list[tuple[Sheet, list]] = []   # (sheet, shelves [y0, height, x_used])
        for pc in pcs:
            L, W = pc.size
            fits = [s for s in stocks if W <= s.w - 2 * EDGE and L <= s.length - 2 * EDGE]
            if not fits:
                problems.append(f"{pc.label} ({L:.0f} x {W:.0f} mm) does not fit any {t} mm sheet")
                continue
            placed = False
            for sh, shelves in open_sheets:
                if sh.stock not in fits:
                    continue
                for shelf in shelves:
                    if W <= shelf[1] + 1e-9 and shelf[2] + L <= sh.stock.length - EDGE:
                        sh.placed.append((pc, shelf[2], shelf[0]))
                        shelf[2] += L + GAP
                        placed = True
                        break
                if placed:
                    break
                y_top = shelves[-1][0] + shelves[-1][1] + GAP if shelves else EDGE
                if y_top + W <= sh.stock.w - EDGE:
                    shelves.append([y_top, W, EDGE + L + GAP])
                    sh.placed.append((pc, EDGE, y_top))
                    placed = True
                    break
            if not placed:
                stock = fits[0]
                sh = Sheet(stock)
                shelves = [[EDGE, W, EDGE + L + GAP]]
                sh.placed.append((pc, EDGE, EDGE))
                open_sheets.append((sh, shelves))
                sheets.append(sh)
    return sheets, problems


def write_dxf(sheet: Sheet, path: str):
    import ezdxf
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    doc.layers.add("CUT", color=1)
    doc.layers.add("LABELS", color=5)
    doc.layers.add("SHEET", color=8)
    msp = doc.modelspace()
    s = sheet.stock
    msp.add_lwpolyline([(0, 0), (s.length, 0), (s.length, s.w), (0, s.w)], close=True,
                       dxfattribs={"layer": "SHEET"})
    for pc, x, y in sheet.placed:
        for r in pc.rings:
            msp.add_lwpolyline([(float(px + x), float(py + y)) for px, py in r], close=True,
                               dxfattribs={"layer": "CUT"})
        L, W = pc.size
        msp.add_text(pc.label, height=min(2.5, max(W * 0.6, 1.2)),
                     dxfattribs={"layer": "LABELS"}).set_placement((x + 1, y + W / 2 - 0.6))
    msp.add_text(f"GRAIN -> along length. {s.title}", height=3,
                 dxfattribs={"layer": "LABELS"}).set_placement((5, s.w + 3))
    doc.saveas(path)


def sheet_svg(sheet: Sheet, scale: float = 1.0, labels: bool = True, for_laser: bool = True) -> str:
    s = sheet.stock
    pad = 8
    W, H = s.length, s.w + pad
    parts = []
    parts.append(f'<rect x="0" y="0" width="{s.length}" height="{s.w}" fill="{"none" if for_laser else "#f3e6cf"}" '
                 f'stroke="#999" stroke-width="0.3"/>')
    for pc, x, y in sheet.placed:
        d = ""
        for r in pc.rings:
            pts = " L ".join(f"{px + x:.3f},{s.w - (py + y):.3f}" for px, py in r)
            d += f"M {pts} Z "
        fill = "none" if for_laser else "#d9b98a"
        parts.append(f'<path d="{d}" fill="{fill}" fill-rule="evenodd" stroke="#ff0000" stroke-width="0.1"/>')
        if labels:
            L, Wd = pc.size
            fs = min(2.5, max(Wd * 0.6, 1.2))
            parts.append(f'<text x="{x + 1:.2f}" y="{s.w - (y + Wd / 2) + fs * 0.35:.2f}" font-size="{fs:.2f}" '
                         f'font-family="Helvetica" fill="#0000ff">{pc.label}</text>')
    parts.append(f'<text x="5" y="{s.w + 6}" font-size="4" font-family="Helvetica" fill="#0000ff">'
                 f'GRAIN ⟶ along the length · {s.title}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W * scale}mm" height="{H * scale}mm" '
            f'viewBox="0 0 {W} {H}">' + "".join(parts) + "</svg>")


def write_svg(sheet: Sheet, path: str):
    with open(path, "w") as f:
        f.write(sheet_svg(sheet))


def stick_plan(st: Structure) -> dict[str, list[list[tuple[str, float]]]]:
    """Hand-cut sticks packed onto stock lengths (first-fit decreasing, 5 mm start trim, 2 mm per saw cut)."""
    ids = part_ids(st)
    by: dict[str, list[tuple[str, float]]] = {}
    for pi, p in enumerate(st.parts):
        if p.laser or p.sec is None:
            continue
        # a piece cut between two angled slots wastes a little more than one saw cut
        two_slot = any(e in ("lap", "mitre") for e in p.info.get("end_style", []))
        by.setdefault(p.stock, []).append((ids[pi], p.length, 3.5 if two_slot else 2.0))
    out = {}
    for stock, items in by.items():
        L = STOCK_BY_ID[stock].length
        bins: list[list] = []
        packed: list[list] = []
        for lab, ln, waste in sorted(items, key=lambda x: -x[1]):
            for b in packed:
                if 5 + sum(x[1] + x[2] for x in b) + ln + waste <= L:     # 5 mm to square the start
                    b.append((lab, ln, waste))
                    break
            else:
                packed.append([(lab, ln, waste)])
        out[stock] = [[(lab, ln) for lab, ln, _ in b] for b in packed]
    return out
