"""Parametric tower families: turn a parameter dict into a Structure.

Three families:
  stick   - legs + rings + bracing, every piece a separate stick or laser-cut strip
  panel   - each face is ONE laser-cut lattice panel; panels glued together at the corners
  diagrid - two sets of straight strips twisted in opposite directions (hyperboloid-like)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .materials import SHEET_THICKNESSES, STOCK_BY_ID, max_sheet_width, sheet_for
from .sections import STICK3, Section, shaved_section, stick_section, strip_section
from .spec import Spec
from .structure import Infeasible, Joint, Member, Part, Structure, bore_clearance

LEG_STICKS = ("sq-5", "sq-6.5", "sq-9.5", "rect-3x9.5", "rect-3x12.5", "dowel-5", "dowel-9.5")
BRACE_STICKS = ("sq-5", "rect-3x9.5", "dowel-5")
HOLE_CLEAR = 0.2      # mm added to laser-cut holes so sticks slide in
END_GRAIN_BOND = 0.6  # a square-cut (end-grain) glue joint is only ~60 % as strong as a side-grain one
LAP_BOND = 0.85       # side-grain lap, knocked down for the small offset it puts in the load path
MIN_STRIP = 2.0       # mm, narrowest strip that survives laser cutting and handling


# --------------------------------------------------------------------------------------------
# parameter spaces
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Param:
    name: str
    kind: str                 # 'int' | 'float' | 'choice'
    lo: float = 0.0
    hi: float = 1.0
    choices: tuple = ()
    width: bool = False       # scaled up/down when fitting the design to the mass budget
    label: str = ""

    def decode(self, x: float):
        x = min(max(float(x), 0.0), 1.0)
        if self.kind == "choice":
            return self.choices[min(int(x * len(self.choices)), len(self.choices) - 1)]
        v = self.lo + x * (self.hi - self.lo)
        return int(round(v)) if self.kind == "int" else v

    def encode(self, v) -> float:
        if self.kind == "choice":
            i = self.choices.index(v) if v in self.choices else 0
            return (i + 0.5) / len(self.choices)
        if self.hi == self.lo:
            return 0.5
        return min(max((float(v) - self.lo) / (self.hi - self.lo), 0.0), 1.0)

    def clamp(self, v):
        if self.kind == "choice":
            return v
        v = min(max(v, self.lo), self.hi)
        return int(round(v)) if self.kind == "int" else float(v)


T = tuple(SHEET_THICKNESSES)
COMMON = [
    Param("n_sides", "int", 3, 6, label="Number of legs / sides"),
    Param("base_r", "float", 22, 110, label="Base radius to leg centre (mm)"),
    Param("top_ratio", "float", 0.2, 1.0, label="Top radius ÷ base radius"),
    Param("n_panels", "int", 2, 12, label="Number of panels (storeys)"),
    Param("panel_ratio", "float", 0.75, 1.2, label="Panel height growth upwards"),
    Param("pattern", "choice", choices=("Z", "X", "W"), label="Bracing pattern"),
]
FRAME = [
    Param("cap_t", "choice", choices=T, label="Top cap thickness (mm)"),
    Param("cap_w", "float", 4, 16, width=True, label="Top cap band width (mm)"),
    Param("cap_mode", "choice", choices=("frame", "strips"), label="Top cap: one-piece or strips"),
    Param("base_t", "choice", choices=T, label="Base ring thickness (mm)"),
    Param("base_w", "float", 2, 16, width=True, label="Base ring width (mm)"),
    Param("base_mode", "choice", choices=("frame", "strips"), label="Base ring: one-piece or strips"),
]
SPACES: dict[str, list[Param]] = {
    "stick": COMMON + [
        Param("leg_type", "choice", choices=LEG_STICKS + ("strip", "angle", "box"), label="Leg type"),
        Param("leg_t", "choice", choices=T, label="Leg strip thickness (mm)"),
        Param("leg_w", "float", 3, 16, width=True, label="Leg width (mm)"),
        Param("brace_type", "choice", choices=BRACE_STICKS + ("strip",), label="Brace type"),
        Param("brace_t", "choice", choices=T, label="Brace thickness (mm)"),
        Param("brace_w", "float", 2, 8, width=True, label="Brace width (mm)"),
        Param("ring_t", "choice", choices=T, label="Ring thickness (mm)"),
        Param("ring_w", "float", 2, 8, width=True, label="Ring width (mm)"),
        Param("gusset", "choice", choices=(0.0, 6.0, 9.0, 12.0), label="Gusset size (mm, 0 = none)"),
    ] + FRAME,
    "panel": COMMON + [
        Param("panel_t", "choice", choices=T, label="Panel sheet thickness (mm)"),
        Param("edge_w", "float", 2, 10, width=True, label="Panel edge (half-leg) width (mm)"),
        Param("diag_w", "float", 2, 8, width=True, label="Diagonal width (mm)"),
        Param("ring_w", "float", 2, 8, width=True, label="Ring width (mm)"),
    ] + FRAME[:3],
    "diagrid": [
        Param("n_sides", "int", 3, 10, label="Strips in each direction"),
        Param("twist", "float", 0.15, 0.95, label="Twist (fraction of half-turn per strip)"),
        Param("base_r", "float", 22, 110, label="Base radius (mm)"),
        Param("top_ratio", "float", 0.2, 1.0, label="Top radius ÷ base radius"),
        Param("member_type", "choice", choices=BRACE_STICKS + ("strip",), label="Strip type"),
        Param("member_t", "choice", choices=T, label="Strip thickness (mm)"),
        Param("member_w", "float", 2, 12, width=True, label="Strip width (mm)"),
        Param("ring_t", "choice", choices=T, label="Ring thickness (mm)"),
        Param("ring_w", "float", 2, 8, width=True, label="Ring width (mm)"),
        Param("ring_every", "int", 0, 2, label="Rings at crossings (0 none, 1 all, 2 every other)"),
    ] + FRAME,
}

SPACES["stick3"] = [
    Param("n_sides", "int", 3, 20, label="Number of legs / sides"),
    Param("base_r", "float", 22, 110, label="Base radius to leg centre (mm)"),
    Param("top_ratio", "float", 0.2, 1.0, label="Top radius ÷ base radius"),
    Param("n_panels", "int", 2, 16, label="Number of panels (storeys)"),
    Param("panel_ratio", "float", 0.75, 1.2, label="Panel height growth upwards"),
    Param("pattern", "choice", choices=("Z", "X", "W"), label="Bracing pattern"),
    Param("joint", "choice", choices=("mitre", "lap"), label="Brace/ring ends: mitred flush or lapped over the leg"),
]
SPACES["diagrid3"] = [
    Param("n_sides", "int", 3, 28, label="Sticks in each direction"),
    Param("twist", "float", 0.15, 0.95, label="Twist (fraction of half-turn per stick)"),
    Param("base_r", "float", 22, 110, label="Base radius (mm)"),
    Param("top_ratio", "float", 0.2, 1.0, label="Top radius ÷ base radius"),
    Param("ring_every", "int", 0, 2, label="Rings at crossings (0 none, 1 all, 2 every other)"),
    Param("joint", "choice", choices=("mitre", "lap"), label="Ring ends: mitred flush or lapped over the strip"),
    Param("ring_joint", "choice", choices=("half lap", "scarf"),
          label="Lapped ring pieces: meet end to end (half lap) or overlap on a diagonal (scarf)"),
]

FAMILY_NAMES = {
    "stick3": "3×3 stick lattice",
    "diagrid3": "3×3 twisted diagrid",
    "stick": "Stick lattice",
    "panel": "Laser-cut panel tower",
    "diagrid": "Twisted diagrid",
}


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def angle_between(a, b) -> float:
    """Acute angle between two lines, degrees (0..90)."""
    c = abs(float(unit(a) @ unit(b)))
    return math.degrees(math.acos(min(1.0, c)))


def level_heights(H: float, n_panels: int, ratio: float) -> list[float]:
    h = ratio ** np.arange(n_panels)
    h = h / h.sum() * H
    return [0.0] + list(np.cumsum(h))


def strip_outline(length: float, width: float, cut0=(0.0, 1.0), cut1=(0.0, 1.0)) -> np.ndarray:
    """Rectangle along +x (the grain) with angled ends. cut = end-line direction (dx, dy)."""
    c0 = cut0[0] / cut0[1] if abs(cut0[1]) > 1e-6 else 0.0
    c1 = cut1[0] / cut1[1] if abs(cut1[1]) > 1e-6 else 0.0
    h = width / 2
    # keep the centre-line length, shear the ends
    return np.array([
        [-c0 * h, -h], [length - c1 * h, -h], [length + c1 * h, h], [c0 * h, h],
    ])


def regular_polygon(R: float, n: int, rot: float) -> np.ndarray:
    a = rot + 2 * np.pi * np.arange(n) / n
    return np.c_[R * np.cos(a), R * np.sin(a)]


def rect_poly(cx, cy, sx, sy, ang) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    pts = np.array([[-sx / 2, -sy / 2], [sx / 2, -sy / 2], [sx / 2, sy / 2], [-sx / 2, sy / 2]])
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + [cx, cy]


def circle_poly(cx, cy, r, n=24) -> np.ndarray:
    a = 2 * np.pi * np.arange(n) / n
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


class Builder:
    def __init__(self, family: str, params: dict, spec: Spec):
        self.family, self.p, self.spec = family, params, spec
        self.nodes: list[np.ndarray] = []
        self.members: list[Member] = []
        self.parts: list[Part] = []
        self.joints: list[Joint] = []
        self.gusset_cfg = None
        self.strip_maker = strip_section   # makes ring/cap strips; 3x3 mode swaps in shaved sticks
        self.joint_style = "mitre"         # how brace/ring ends meet their host: 'mitre' or 'lap'
        self.ring_scarf = 0.0              # mm: ring pieces overlap each other on a long diagonal (0 = butt)
        self._scarf_waiting: dict[int, int] = {}   # node -> ring part whose scarf end is waiting for its partner

    # -- primitives --------------------------------------------------------------------------
    def node(self, xyz) -> int:
        self.nodes.append(np.asarray(xyz, float))
        return len(self.nodes) - 1

    def part(self, **kw) -> int:
        self.parts.append(Part(**kw))
        return len(self.parts) - 1

    def member(self, i, j, sec, kind, part, **kw) -> int:
        self.members.append(Member(i, j, sec, kind, part, **kw))
        self.parts[part].members.append(len(self.members) - 1)
        return len(self.members) - 1

    def axis(self, m: int) -> np.ndarray:
        mm = self.members[m]
        return unit(self.nodes[mm.j] - self.nodes[mm.i])

    def length(self, i, j) -> float:
        return float(np.linalg.norm(self.nodes[j] - self.nodes[i]))

    def glue_end(self, m: int, end: int, host: int, host_axis, kind="butt", area=None, host_w=3.0,
                 piece_kind: str = ""):
        """Glue one end of member m onto the side of a host piece.

        Two ways to shape the end (the contact can never be longer than the stick allows):
          mitre - end cut at the meeting angle so it sits flush on the host's side.
                  Glued face = section / sin(angle); it is end grain at 90 deg, more side grain
                  the shallower the angle.
          lap   - end shaped to lie on the host's outer face. Neighbouring pieces meet on the same
                  host (two ring pieces at every crossing), so each end only covers HALF the host:
                  glued face = width x host width / (2 sin(angle)), side grain to side grain, and the
                  overlap puts the piece slightly off-line (taken as 15 % weaker).
          scarf - (ring pieces only) the two pieces meeting on a strip are cut on a long diagonal
                  through their thickness: one lies flat on the strip, the next overlaps it. The ring
                  becomes one continuous hoop, so the ring force passes piece to piece through the
                  long side-grain scarf face instead of through the small patch on the strip. The
                  strip patch is still shared (half each) and only has to hold the ring in place.
        """
        mm = self.members[m]
        x = self.axis(m)
        gamma = max(angle_between(x, host_axis), 15.0)
        s = math.sin(math.radians(gamma))
        c2 = math.cos(math.radians(gamma)) ** 2
        scarfed = (kind == "butt" and self.joint_style == "lap" and self.ring_scarf > 0
                   and piece_kind in ("ring", "base", "cap"))
        strength_area = None
        if scarfed:
            kind = "scarf end"
            area = mm.sec.size_y * host_w / (2 * s)          # glue on the strip patch (shared)
            per = 2 * (mm.sec.size_y + host_w / 2) / s
            end_grain, bond = 0.0, 1.0
            strength_area = mm.sec.size_y * self.ring_scarf  # the ring force goes through the scarf face
            gamma = 15.0                                     # ...as shear along it
        elif kind == "butt" and self.joint_style == "lap":
            kind = "lap end"
            area = mm.sec.size_y * host_w / (2 * s)
            per = 2 * (mm.sec.size_y + host_w / 2) / s
            end_grain, bond = 0.0, LAP_BOND
        else:
            if area is None:
                area = mm.sec.A / s
            per = 2 * (mm.sec.size_y / s + mm.sec.size_z)
            end_grain = mm.sec.A * (1 - c2)
            bond = END_GRAIN_BOND + (1 - END_GRAIN_BOND) * c2
        mm.ends[end] = dict(host=host, area=strength_area or area, kind=kind, gamma=gamma, gusset=None, bond=bond)
        node = mm.i if end == 0 else mm.j
        self.joints.append(Joint(node, (mm.part, host), area, kind, perimeter=per, end_grain=end_grain))
        mm.ends[end]["joint"] = len(self.joints) - 1
        if scarfed:
            # the diagonal faces of the two ring pieces meeting here are glued to each other
            other = self._scarf_waiting.pop(node, None)
            if other is None:
                self._scarf_waiting[node] = mm.part
            else:
                sc = self.ring_scarf
                self.joints.append(Joint(node, (other, mm.part), mm.sec.size_y * sc, "scarf",
                                         perimeter=2 * (sc + mm.sec.size_y), end_grain=0.0))

    # -- strips between two nodes, butt-glued at both ends ------------------------------------
    def strip_between(self, a, b, sec, kind, name, normal, hosts, step, mid=None, group=""):
        """A straight piece from node a to b (optionally through node `mid`).

        hosts = ((host_part, host_axis, host_half_width) or None, same for end b).
        normal = face normal (the strip's thickness direction).
        """
        L = self.length(a, b)
        x = unit(self.nodes[b] - self.nodes[a])
        y = unit(np.cross(normal, x))
        trims, cuts = [], []
        for h in hosts:
            if h is None:
                trims.append(0.0)
                cuts.append((0.0, 1.0))
                continue
            hax = unit(h[1])
            s = max(math.sin(math.radians(angle_between(x, hax))), 0.26)
            trims.append(h[2] / s)
            cuts.append((float(hax @ x), float(hax @ y)))
        ring_scarf = self.ring_scarf if (self.joint_style == "lap" and kind in ("ring", "base", "cap")) else 0.0
        if ring_scarf:
            # scarfed ring ends run on past the strip's centre by half the scarf, overlapping the next piece
            Lc = L + ring_scarf
        elif self.joint_style == "lap":
            # lapped ends run on to the middle of the host's face, where the next piece starts
            Lc = L
        else:
            Lc = max(L - trims[0] - trims[1], 0.3 * L)
        p = self.part(
            name=name, kind=kind, sec=sec, volume=sec.A * Lc, length=Lc,
            laminate_area=sec.lamination_bond * Lc, laser=sec.laser, stock=sec.stock, step=step,
            end_angles=tuple(90 - angle_between((c[0], c[1], 0), (0, 1, 0)) for c in cuts),
            group=group, info=dict(end_style=[("scarf" if ring_scarf else "lap" if self.joint_style == "lap"
                                               else "mitre") if h else "square"
                                              for h in hosts]),
        )
        if sec.laser:
            self.parts[p].outline = [[strip_outline(Lc, sec.w, cuts[0], cuts[1])]]
        pts = [a, b] if mid is None else [a, mid, b]
        ms = []
        for u, v in zip(pts[:-1], pts[1:]):
            ms.append(self.member(u, v, sec, kind, p, grain_deg=0.0, frame=False, ref=y))
        if mid is not None:
            self.members[ms[0]].frame = self.members[ms[1]].frame = True
        for end, h in enumerate(hosts):
            if h is not None:
                m = ms[0] if end == 0 else ms[-1]
                self.glue_end(m, end, h[0], h[1], host_w=2 * h[2], piece_kind=kind)
        return p, ms

    def _frame_fits(self, corner_nodes, t, w, leg_secs) -> bool:
        n = len(corner_nodes)
        pts = np.array([self.nodes[c][:2] for c in corner_nodes])
        R = float(np.mean(np.linalg.norm(pts, axis=1)))
        w = max(w, max(max(s.size_y, s.size_z) for s in leg_secs) + 3.0)
        cos_n = math.cos(math.pi / n)
        if (R - (w / 2) / cos_n) * cos_n < self.spec.bore_r:
            return False
        side0 = pts[1 % n] - pts[0]
        g = math.atan2(side0[1], side0[0])
        outer = regular_polygon(R + (w / 2) / cos_n, n, math.atan2(pts[0, 1], pts[0, 0]) - g)
        return float(outer[:, 1].max() - outer[:, 1].min()) <= max_sheet_width(t) - 2

    # -- horizontal ring of n nodes (one-piece laser-cut frame, or separate strips) -----------
    def ring(self, corner_nodes, t, w, mode, kind, step, leg_parts, leg_secs, mids=None,
             name="Ring", face_normals=None):
        n = len(corner_nodes)
        spec = self.spec
        z = float(self.nodes[corner_nodes[0]][2])
        if mode == "frame" and not self._frame_fits(corner_nodes, t, w, leg_secs):
            mode = "strips"   # too big for one sheet, or the hole would foul the pole
        if mode == "frame":
            # polygon centred on the leg centres; legs pass through laser-cut holes at corners
            pts = np.array([self.nodes[c][:2] for c in corner_nodes])
            R = float(np.mean(np.linalg.norm(pts, axis=1)))
            rot = math.atan2(pts[0, 1], pts[0, 0])
            leg_size = max(max(s.size_y, s.size_z) for s in leg_secs)
            w = max(w, leg_size + 3.0)
            cos_n = math.cos(math.pi / n)
            R_out = R + (w / 2) / cos_n
            R_in = R - (w / 2) / cos_n
            if R_in * cos_n < spec.bore_r:
                raise Infeasible(f"{name}: inner hole smaller than the pole clearance")
            # grain along the first side
            side0 = pts[1 % n] - pts[0]
            g_ang = math.atan2(side0[1], side0[0])
            c, s = math.cos(-g_ang), math.sin(-g_ang)
            Rm = np.array([[c, -s], [s, c]])
            outer = regular_polygon(R_out, n, rot) @ Rm.T
            inner = regular_polygon(R_in, n, rot) @ Rm.T
            span = float(outer[:, 1].max() - outer[:, 1].min())
            if span > max_sheet_width(t) - 2:
                raise Infeasible(f"{name}: one-piece frame is wider than the sheet")
            holes = []
            for k, cn in enumerate(corner_nodes):
                sec = leg_secs[k]
                q = (pts[k]) @ Rm.T
                th = math.atan2(pts[k][1], pts[k][0]) - g_ang
                if sec.shape == "round":
                    holes.append(circle_poly(q[0], q[1], sec.w / 2 + HOLE_CLEAR))
                else:
                    holes.append(rect_poly(q[0], q[1], sec.size_z + HOLE_CLEAR * 2,
                                           sec.size_y + HOLE_CLEAR * 2, th))
            area = 0.5 * n * math.sin(2 * math.pi / n) * (R_out ** 2 - R_in ** 2)
            hole_area = sum(math.pi * s.w ** 2 / 4 if s.shape == "round" else s.size_y * s.size_z
                            for s in leg_secs)
            sec = Section("rect", t, w, sheet_for(t, span + 2).id, True)
            p = self.part(name=name, kind=kind, sec=sec, volume=(area - hole_area) * t, laser=True,
                          stock=sec.stock, step=step, outline=[[outer, inner] + holes],
                          length=2 * math.pi * R, info=dict(g_ang=g_ang, z=z, t=t))
            gvec = np.array([math.cos(g_ang), math.sin(g_ang), 0.0])
            for k in range(n):
                a, b = corner_nodes[k], corner_nodes[(k + 1) % n]
                pts3 = [a, b] if mids is None else [a, mids[k], b]
                for u, v in zip(pts3[:-1], pts3[1:]):
                    xax = unit(self.nodes[v] - self.nodes[u])
                    yax = unit(np.cross([0, 0, 1], xax))
                    self.member(u, v, sec, kind, p, grain_deg=angle_between(xax, gvec),
                                frame=True, K=0.8, ref=yax)
            for k, cn in enumerate(corner_nodes):
                ls = leg_secs[k]
                hole = 2 * (ls.size_y + ls.size_z)
                # bead on the top and bottom of the hole; the laser-cut hole wall is half end grain
                self.joints.append(Joint(cn, (p, leg_parts[k]), hole * t, "notch", perimeter=2 * hole,
                                         end_grain=0.5 * hole * t))
            return [p]
        # separate strips, each butt glued between two legs, in the face plane
        parts = []
        for k in range(n):
            a, b = corner_nodes[k], corner_nodes[(k + 1) % n]
            if face_normals is not None:
                nrm = face_normals[k]
            else:
                mid = (self.nodes[a] + self.nodes[b]) / 2
                nrm = unit([mid[0], mid[1], 0])
            sec = self.strip_maker("rect", t, w)
            hosts = []
            for kk in (k, (k + 1) % n):
                lm = self.parts[leg_parts[kk]].members
                hosts.append((leg_parts[kk], self.axis(lm[0]), max(leg_secs[kk].size_y, leg_secs[kk].size_z) / 2))
            p, _ = self.strip_between(a, b, sec, kind, f"{name} side {k + 1}", nrm, hosts, step,
                                      mid=None if mids is None else mids[k], group=f"{name}")
            parts.append(p)
        return parts

    # -- gussets: small laser-cut plates on the outside of a node ----------------------------
    def add_gussets(self, node, leg_part, leg_axis, leg_sec, member_ids, normal, step):
        g = self.gusset_cfg
        if not g or not member_ids:
            return
        size, t = g
        sec = strip_section("rect", t, size)
        area_leg = size * max(leg_sec.size_y, 2.0)
        vol = 0.6 * size * size * t
        outline = np.array([[0, -size / 2], [size, -size / 2], [size * 0.25, size / 2], [0, size / 2]])
        p = self.part(name="Gusset", kind="gusset", sec=sec, volume=vol, laser=True, stock=sec.stock,
                      step=step, outline=[[outline]], length=size, group="Gusset")
        total = area_leg
        per = 2 * size + size * 1.4   # outline of the gusset plate
        for m in member_ids:
            mm = self.members[m]
            end = 0 if mm.i == node else 1
            if mm.ends[end] is None:
                continue
            ov = min(size * 0.8, self.length(mm.i, mm.j) / 3) * mm.sec.size_y
            th = angle_between(self.axis(m), leg_axis)  # gusset grain runs along the leg
            mm.ends[end]["gusset"] = dict(area=ov, t=t, size=size, theta=th, joint=len(self.joints))
            total += ov
        self.joints.append(Joint(node, (p, leg_part), total, "gusset", perimeter=per))

    # -- finish ------------------------------------------------------------------------------
    def finish(self, supports, tops, levels, leg_parts, height, info=None) -> Structure:
        nodes = np.array(self.nodes)
        bore = self.spec.bore_r
        for m in self.members:
            part = self.parts[m.part]
            if part.kind in ("cap", "base") and part.info.get("g_ang") is not None:
                continue  # one-piece frames check their own hole size
            c = bore_clearance(nodes[m.i], nodes[m.j], m.sec.half_extent)
            if c < bore - 1e-6:
                raise Infeasible(f"{m.kind} gets within {c:.1f} mm of the pole axis (needs {bore:.1f})")
        for j in self.joints:
            j.step = max(self.parts[p].step for p in j.parts)
        for p in self.parts:
            if p.sec is not None and p.laser and p.kind not in ("cap", "base", "panel"):
                if p.sec.w > max_sheet_width(p.sec.t) or p.sec.w < MIN_STRIP - 1e-9:
                    raise Infeasible(f"{p.name}: strip width {p.sec.w:.1f} mm not cuttable")
            if p.length > 900:
                raise Infeasible(f"{p.name} longer than the 915 mm stock")
        return Structure(self.family, dict(self.p), nodes, self.members, self.parts, self.joints,
                         supports, tops, levels, leg_parts, height, info or {})


def _leg_section(p) -> Section:
    lt = p["leg_type"]
    if lt in STOCK_BY_ID:
        return stick_section(lt)
    t, w = float(p["leg_t"]), float(p["leg_w"])
    if lt == "strip":
        return strip_section("rect", t, max(w, t))
    if (lt == "box" and w < 2 * t + MIN_STRIP + 1) or (lt == "angle" and w < t + MIN_STRIP + 1):
        raise Infeasible("L-angle/box leg too narrow to laminate")
    return strip_section(lt, t, w)


def _brace_section(kind, t, w) -> Section:
    if kind == "shaved":
        return shaved_section("rect", t, w)
    if kind in STOCK_BY_ID:
        return stick_section(kind)
    return strip_section("rect", float(t), float(w))


def _face_normal(a, b, c) -> np.ndarray:
    nrm = unit(np.cross(b - a, c - a))
    mid = (a + b) / 2
    if nrm[:2] @ mid[:2] < 0:
        nrm = -nrm
    return nrm


def _line_intersection(p1, p2, p3, p4) -> np.ndarray:
    """Closest point between two (coplanar) lines p1-p2 and p3-p4."""
    d1, d2, r = p2 - p1, p4 - p3, p1 - p3
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ r, d2 @ r
    den = a * c - b * b
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return (p1 + s * d1 + p3 + t * d2) / 2


# --------------------------------------------------------------------------------------------
# family 1: stick lattice
# --------------------------------------------------------------------------------------------
def build_stick(p: dict, spec: Spec, family: str = "stick", sticks3: bool = False) -> Structure:
    b = Builder(family, p, spec)
    if sticks3:
        b.strip_maker = shaved_section
        b.joint_style = p.get("joint", "mitre")
    n, H = int(p["n_sides"]), spec.height
    leg = _leg_section(p)
    if not spec.allow_sticks and not leg.laser:
        raise Infeasible("sticks disabled")
    rb = float(p["base_r"])
    halves = [leg.half_extent, b.strip_maker("rect", float(p["ring_t"]), float(p["ring_w"])).half_extent]
    if p["cap_mode"] == "strips":
        halves.append(b.strip_maker("rect", float(p["cap_t"]), float(p["cap_w"])).half_extent)
    rt_min = (spec.bore_r + max(halves) + 0.5) / math.cos(math.pi / n)
    rt = max(rt_min, rb * float(p["top_ratio"]))
    rb = max(rb, rt)
    z = level_heights(H, int(p["n_panels"]), float(p["panel_ratio"]))
    npn = len(z) - 1
    th = [2 * math.pi * i / n for i in range(n)]
    R = lambda zz: rb + (rt - rb) * zz / H
    P = [[b.node((R(zk) * math.cos(t), R(zk) * math.sin(t), zk)) for zk in z] for t in th]

    g = float(p.get("gusset", 0))
    b.gusset_cfg = (g, 1.5) if g >= 4 else None

    # legs
    leg_parts = []
    for i in range(n):
        L = b.length(P[i][0], P[i][-1])
        pi = b.part(name=f"Leg {i + 1}", kind="leg", sec=leg, volume=leg.A * L, length=L,
                    laminate_area=leg.lamination_bond * L, laser=leg.laser, stock=leg.stock, step=2,
                    group="Leg")
        if leg.laser:
            b.parts[pi].outline = [[strip_outline(L, sw)] for (_, sw) in leg.strips]
        tang = np.array([-math.sin(th[i]), math.cos(th[i]), 0.0])
        for k in range(npn):
            b.member(P[i][k], P[i][k + 1], leg, "leg", pi, frame=True, K=1.0, ref=tang)
        leg_parts.append(pi)
    leg_axis = [unit(b.nodes[P[i][-1]] - b.nodes[P[i][0]]) for i in range(n)]
    leg_half = max(leg.size_y, leg.size_z) / 2
    normals = [_face_normal(b.nodes[P[i][0]], b.nodes[P[(i + 1) % n][0]], b.nodes[P[i][-1]]) for i in range(n)]

    pattern = p["pattern"]
    mids_at = {}
    if pattern == "W":
        for k in range(1, npn + 1, 2):
            mids_at[k] = [b.node((b.nodes[P[i][k]] + b.nodes[P[(i + 1) % n][k]]) / 2) for i in range(n)]

    # base ring (level 0), intermediate rings, cap (top level)
    rings = {}
    rings[0] = b.ring([P[i][0] for i in range(n)], float(p["base_t"]), float(p["base_w"]),
                      p["base_mode"], "base", 1, leg_parts, [leg] * n, mids=mids_at.get(0),
                      name="Base ring", face_normals=normals)
    for k in range(1, npn):
        rings[k] = b.ring([P[i][k] for i in range(n)], float(p["ring_t"]), float(p["ring_w"]),
                          "strips", "ring", 2 + k, leg_parts, [leg] * n, mids=mids_at.get(k),
                          name=f"Ring {k}", face_normals=normals)
    rings[npn] = b.ring([P[i][npn] for i in range(n)], float(p["cap_t"]), float(p["cap_w"]),
                        p["cap_mode"], "cap", 3 + npn, leg_parts, [leg] * n, mids=mids_at.get(npn),
                        name="Top cap", face_normals=normals)

    def ring_host(k, face):
        parts = rings[k]
        pr = parts[face] if len(parts) > 1 else parts[0]
        a, c = b.nodes[P[face][k]], b.nodes[P[(face + 1) % n][k]]
        return (pr, c - a, b.parts[pr].sec.w / 2)

    # bracing
    bsec = _brace_section(p["brace_type"], p["brace_t"], p["brace_w"])
    if not spec.allow_sticks and not bsec.laser:
        raise Infeasible("sticks disabled")
    node_members: dict[tuple[int, int], list[int]] = {}

    def note(m_ids, *keys):
        for key, m in zip(keys, m_ids):
            if key is not None:
                node_members.setdefault(key, []).append(m)

    for f in range(n):
        i, i1 = f, (f + 1) % n
        nrm = normals[f]
        for k in range(npn):
            step = 3 + k
            hl = lambda leg_i: (leg_parts[leg_i], leg_axis[leg_i], leg_half)
            grp = f"Brace panel {k + 1}"
            if pattern == "Z":
                if k % 2 == 0:
                    a, c, la, lc = P[i][k], P[i1][k + 1], i, i1
                else:
                    a, c, la, lc = P[i1][k], P[i][k + 1], i1, i
                _, ms = b.strip_between(a, c, bsec, "brace", f"Brace F{f + 1}P{k + 1}", nrm,
                                        (hl(la), hl(lc)), step, group=grp)
                note(ms, (la, k), None)
                note(ms[-1:], (lc, k + 1))
            elif pattern == "X":
                xm = _line_intersection(b.nodes[P[i][k]], b.nodes[P[i1][k + 1]],
                                        b.nodes[P[i1][k]], b.nodes[P[i][k + 1]])
                mid = b.node(xm)
                pa, ms = b.strip_between(P[i][k], P[i1][k + 1], bsec, "brace", f"Brace F{f + 1}P{k + 1}a",
                                         nrm, (hl(i), hl(i1)), step, mid=mid, group=grp + " (long)")
                note(ms, (i, k), None)
                note(ms[-1:], (i1, k + 1))
                ax = unit(b.nodes[P[i1][k + 1]] - b.nodes[P[i][k]])
                for half, (s0, l0) in enumerate(((P[i1][k], i1), (P[i][k + 1], i))):
                    _, ms2 = b.strip_between(s0, mid, bsec, "brace", f"Brace F{f + 1}P{k + 1}b{half + 1}",
                                             nrm, (hl(l0), (pa, ax, bsec.w / 2)), step, group=grp + " (half)")
                    note(ms2, (l0, k + (1 if half else 0)))
            else:  # W: V-shapes meeting at ring mid-points
                if k % 2 == 0:
                    mnode, kk = mids_at[k + 1][f], k + 1
                    ends = ((P[i][k], i, k), (P[i1][k], i1, k))
                else:
                    mnode, kk = mids_at[k][f], k
                    ends = ((P[i][k + 1], i, k + 1), (P[i1][k + 1], i1, k + 1))
                for half, (s0, l0, lev) in enumerate(ends):
                    _, ms = b.strip_between(s0, mnode, bsec, "brace", f"Brace F{f + 1}P{k + 1}{'ab'[half]}",
                                            nrm, (hl(l0), ring_host(kk, f)), step, group=grp)
                    note(ms, (l0, lev))

    # rings that butt into legs also count as members at the leg node (for gussets)
    for k, parts in rings.items():
        for pr in parts:
            for m in b.parts[pr].members:
                mm = b.members[m]
                for end, nd in enumerate((mm.i, mm.j)):
                    if mm.ends[end] is not None:
                        for li in range(n):
                            if P[li][k] == nd:
                                node_members.setdefault((li, k), []).append(m)
    if b.gusset_cfg:
        for (li, k), ms in node_members.items():
            b.add_gussets(P[li][k], leg_parts[li], leg_axis[li], leg, ms, normals[li], 3 + max(k - 1, 0))

    return b.finish([P[i][0] for i in range(n)], [P[i][-1] for i in range(n)], z, leg_parts, H,
                    info=dict(base_r=rb, top_r=rt, n=n))


# --------------------------------------------------------------------------------------------
# family 2: laser-cut panel tower
# --------------------------------------------------------------------------------------------
def build_panel(p: dict, spec: Spec) -> Structure:
    b = Builder("panel", p, spec)
    n, H = int(p["n_sides"]), spec.height
    t = float(p["panel_t"])
    corner = strip_section("angle", t, max(float(p["edge_w"]), t + MIN_STRIP))
    ring_sec = strip_section("rect", t, float(p["ring_w"]))
    diag_sec = strip_section("rect", t, float(p["diag_w"]))
    rb = float(p["base_r"])
    rt = max((spec.bore_r + corner.half_extent + 0.5) / math.cos(math.pi / n), rb * float(p["top_ratio"]))
    rb = max(rb, rt)
    z = level_heights(H, int(p["n_panels"]), float(p["panel_ratio"]))
    npn = len(z) - 1
    th = [2 * math.pi * i / n for i in range(n)]
    R = lambda zz: rb + (rt - rb) * zz / H
    P = [[b.node((R(zk) * math.cos(a), R(zk) * math.sin(a), zk)) for zk in z] for a in th]
    sheet_w = max_sheet_width(t)

    # corner legs (made of the two panel edges glued together)
    leg_parts = []
    for i in range(n):
        L = b.length(P[i][0], P[i][-1])
        pi = b.part(name=f"Corner {i + 1}", kind="corner", sec=corner, volume=0.0, length=L,
                    laser=True, stock=corner.stock, step=2, group="Corner")
        tang = np.array([-math.sin(th[i]), math.cos(th[i]), 0.0])
        for k in range(npn):
            b.member(P[i][k], P[i][k + 1], corner, "leg", pi, frame=True, K=1.0, ref=tang)
        leg_parts.append(pi)

    pattern = p["pattern"]
    edge_w = corner.w
    for f in range(n):
        i, i1 = f, (f + 1) % n
        A0, B0, A1, B1 = (b.nodes[P[i][0]], b.nodes[P[i1][0]], b.nodes[P[i][-1]], b.nodes[P[i1][-1]])
        nrm = _face_normal(A0, B0, A1)
        gx = unit((A1 + B1) / 2 - (A0 + B0) / 2)             # grain: up the panel centre-line
        gy = unit(np.cross(nrm, gx))
        origin = (A0 + B0) / 2
        to2d = lambda q: np.array([(q - origin) @ gx, (q - origin) @ gy])
        pp = b.part(name=f"Panel {f + 1}", kind="panel", sec=ring_sec, volume=0.0, laser=True,
                    stock=sheet_for(t).id, step=1, group="Panel", length=float(np.linalg.norm(A1 - A0)))
        rects = []   # (p2d_a, p2d_b, width) for the outline
        vol = 0.0

        def add(u, v, sec, kind, K=0.8):
            nonlocal vol
            x = unit(b.nodes[v] - b.nodes[u])
            m = b.member(u, v, sec, kind, pp, grain_deg=angle_between(x, gx), frame=True, K=K,
                         ref=unit(np.cross(nrm, x)))
            L = b.length(u, v)
            vol += sec.A * L
            rects.append((to2d(b.nodes[u]), to2d(b.nodes[v]), sec.w))
            return m

        for side in (i, i1):  # panel's own edge strips (half of each corner leg)
            for k in range(npn):
                rects.append((to2d(b.nodes[P[side][k]]), to2d(b.nodes[P[side][k + 1]]), edge_w))
            vol += edge_w * t * b.length(P[side][0], P[side][-1])
        mids = {}
        if pattern == "W":
            for k in range(1, npn + 1, 2):
                mids[k] = b.node((b.nodes[P[i][k]] + b.nodes[P[i1][k]]) / 2)
        for k in range(npn + 1):
            if k in mids:
                add(P[i][k], mids[k], ring_sec, "ring")
                add(mids[k], P[i1][k], ring_sec, "ring")
            else:
                add(P[i][k], P[i1][k], ring_sec, "ring")
        for k in range(npn):
            if pattern == "Z":
                a, c = (P[i][k], P[i1][k + 1]) if k % 2 == 0 else (P[i1][k], P[i][k + 1])
                add(a, c, diag_sec, "brace")
            elif pattern == "X":
                mid = b.node(_line_intersection(b.nodes[P[i][k]], b.nodes[P[i1][k + 1]],
                                                b.nodes[P[i1][k]], b.nodes[P[i][k + 1]]))
                for a, c in ((P[i][k], mid), (mid, P[i1][k + 1]), (P[i1][k], mid), (mid, P[i][k + 1])):
                    add(a, c, diag_sec, "brace")
            else:
                if k % 2 == 0:
                    for s0 in (P[i][k], P[i1][k]):
                        add(s0, mids[k + 1], diag_sec, "brace")
                else:
                    for s0 in (P[i][k + 1], P[i1][k + 1]):
                        add(mids[k], s0, diag_sec, "brace")
        # outline bounding box must fit the sheet width
        allp = np.array([q for r in rects for q in r[:2]])
        span = float(allp[:, 1].max() - allp[:, 1].min()) + edge_w
        if span > sheet_w - 2:
            raise Infeasible(f"panel {span:.0f} mm wide does not fit a {sheet_w:.0f} mm sheet")
        b.parts[pp].volume = vol * 0.92   # overlaps where strips meet are counted twice
        b.parts[pp].info = dict(rects=rects, t=t)
        b.parts[pp].outline = None        # built from rects by the laser module (needs booleans)
    # corner glue joints (edges of neighbouring panels)
    panels = [k for k, q in enumerate(b.parts) if q.kind == "panel"]
    for i in range(n):
        L = b.length(P[i][0], P[i][-1])
        b.joints.append(Joint(P[i][-1], (panels[(i - 1) % n], panels[i], leg_parts[i]), t * L, "edge",
                              perimeter=2 * L))

    if p.get("cap_mode", "frame") and float(p.get("cap_w", 0)) > 0:
        b.ring([P[i][npn] for i in range(n)], float(p["cap_t"]), float(p["cap_w"]), p["cap_mode"],
               "cap", 3, leg_parts, [corner] * n, name="Top cap")
    return b.finish([P[i][0] for i in range(n)], [P[i][-1] for i in range(n)], z, leg_parts, H,
                    info=dict(base_r=rb, top_r=rt, n=n))


# --------------------------------------------------------------------------------------------
# family 3: twisted diagrid
# --------------------------------------------------------------------------------------------
def build_diagrid(p: dict, spec: Spec, family: str = "diagrid", sticks3: bool = False) -> Structure:
    b = Builder(family, p, spec)
    if sticks3:
        b.strip_maker = shaved_section
        b.joint_style = p.get("joint", "mitre")
        if p.get("ring_joint") == "scarf":
            b.ring_scarf = 9.0      # 3x the stick size: a 1-in-3 diagonal
    n, H = int(p["n_sides"]), spec.height
    k = max(1, min(n - 1, int(round(float(p["twist"]) * n))))
    phi = k * math.pi / n
    sec = _brace_section(p["member_type"], p["member_t"], p["member_w"])
    if not spec.allow_sticks and not sec.laser:
        raise Infeasible("sticks disabled")
    rb = float(p["base_r"])
    rt = max((spec.bore_r + sec.half_extent + 0.5) / math.cos(math.pi / n), rb * float(p["top_ratio"]))
    rb = max(rb, rt)

    def chord(a0, s):
        q = (1 - s) * rb * np.array([math.cos(a0), math.sin(a0)]) + s * rt * np.array(
            [math.cos(a0 + phi), math.sin(a0 + phi)])
        return np.array([q[0], q[1], s * H])

    def alpha(s):
        q = chord(0.0, s)
        return math.atan2(q[1], q[0])

    # height (as chord fraction) of every crossing level q = 0..k
    s_lev = [0.0]
    for q in range(1, k):
        target, lo, hi = q * math.pi / n, 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if alpha(mid) < target else (lo, mid)
        s_lev.append((lo + hi) / 2)
    s_lev.append(1.0)
    N = [[b.node(chord(2 * math.pi * i / n, s)) for i in range(n)] for s in s_lev]
    z = [s * H for s in s_lev]

    a_parts, b_parts = [], []
    for fam in (0, 1):
        for j in range(n):
            if fam == 0:
                seq = [N[q][j] for q in range(k + 1)]
            else:
                seq = [N[q][(j - q) % n] for q in range(k + 1)]
            L = b.length(seq[0], seq[-1])
            # ends are cut level so the strip stands flat on the table and under the top ring
            tilt0 = angle_between(b.nodes[seq[1]] - b.nodes[seq[0]], (0, 0, 1))
            tilt1 = angle_between(b.nodes[seq[-1]] - b.nodes[seq[-2]], (0, 0, 1))
            pr = b.part(name=f"Strip {'AB'[fam]}{j + 1}", kind="diag", sec=sec, volume=sec.A * L, length=L,
                        laser=sec.laser, stock=sec.stock, step=2 + fam, group=f"Strip {'AB'[fam]}",
                        end_angles=(90 - tilt0, 90 - tilt1), info=dict(end_style=["level", "level"]))
            if sec.laser:
                b.parts[pr].outline = [[strip_outline(L, sec.w)]]
            b.parts[pr].offset = 0.0 if fam == 0 else (sec.size_z + 0.0)
            for u, v in zip(seq[:-1], seq[1:]):
                mid = (b.nodes[u] + b.nodes[v]) / 2
                nrm = unit([mid[0], mid[1], 0])
                x = unit(b.nodes[v] - b.nodes[u])
                b.member(u, v, sec, "diag", pr, frame=True, K=1.0, ref=unit(np.cross(nrm, x)))
            (a_parts if fam == 0 else b_parts).append(pr)
    # lap joints where the two families cross
    for q in range(1, k):
        for j in range(n):
            pa, pb = a_parts[j], b_parts[(j + q) % n]
            ma = b.parts[pa].members[q - 1]
            mb = b.parts[pb].members[q - 1]
            ang = max(angle_between(b.axis(ma), b.axis(mb)), 10.0)
            area = sec.size_y ** 2 / math.sin(math.radians(ang)) if sec.shape != "round" else 0.4 * sec.w ** 2
            per = 4 * sec.size_y / math.sin(math.radians(ang)) if sec.shape != "round" else 2 * sec.w
            b.joints.append(Joint(N[q][j], (pa, pb), area, "lap", perimeter=per))

    every = int(p["ring_every"])
    if every > 0:
        for q in range(1, k):
            if (q - 1) % every:
                continue
            hosts = [a_parts[i] for i in range(n)]
            b.ring([N[q][i] for i in range(n)], float(p["ring_t"]), float(p["ring_w"]), "strips", "ring",
                   4, hosts, [sec] * n, name=f"Ring {q}")
    b.ring([N[0][i] for i in range(n)], float(p["base_t"]), float(p["base_w"]), p["base_mode"], "base", 1,
           a_parts, [sec] * n, name="Base ring")
    b.ring([N[k][i] for i in range(n)], float(p["cap_t"]), float(p["cap_w"]), p["cap_mode"], "cap", 5,
           a_parts, [sec] * n, name="Top cap")
    return b.finish(list(N[0]), list(N[k]), z, a_parts, H,
                    info=dict(base_r=rb, top_r=rt, n=n, twist_k=k))


FULL3 = dict(brace_t=3.0, brace_w=3.0, ring_t=3.0, ring_w=3.0, cap_t=3.0, cap_w=3.0, base_t=3.0, base_w=3.0)


def _place_rings(st: Structure, layer: float) -> Structure:
    """Where the ring pieces really sit (3D models and jig only; the analysis keeps the node positions).

    layer: radial offset of rings/braces that lap over the outside of the strips or legs.
    The base ring is lifted so it rests just above the table (in a shallow groove in the base plate),
    and the top ring is dropped so its top is flush with the strip tops.
    """
    for p in st.parts:
        if p.kind in ("ring", "base", "cap") or (p.kind == "brace" and layer):
            if layer:
                p.offset = layer
            if p.sec is not None and p.kind == "base":
                p.info["dz"] = p.sec.size_y / 2 + 0.5
            elif p.sec is not None and p.kind == "cap":
                p.info["dz"] = -p.sec.size_y / 2
    return st


def _stick3(p: dict, spec: Spec) -> Structure:
    """Stick lattice where every piece is one full-size supplied 3 x 3 stick (only the joints are shaped)."""
    q = dict(p, leg_type="teacher-3x3", leg_t=3.0, leg_w=3.0, brace_type="shaved", gusset=0.0,
             cap_mode="strips", base_mode="strips", **FULL3)
    st = build_stick(q, spec, "stick3", sticks3=True)
    # lapped braces/rings sit on the outer face of the legs: leg half + own half
    return _place_rings(st, STICK3 if q.get("joint") == "lap" else 0.0)


def _diagrid3(p: dict, spec: Spec) -> Structure:
    q = dict(p, member_type="shaved", cap_mode="strips", base_mode="strips", member_t=3.0, member_w=3.0, **FULL3)
    st = build_diagrid(q, spec, "diagrid3", sticks3=True)
    # rings lap over the outer (B) strips: A half + B + ring half from the A centre-line
    return _place_rings(st, 2 * STICK3)


BUILDERS = {"stick": build_stick, "panel": build_panel, "diagrid": build_diagrid,
            "stick3": _stick3, "diagrid3": _diagrid3}


def build(family: str, params: dict, spec: Spec) -> Structure:
    full = {pp.name: pp.clamp(params[pp.name]) if pp.name in params else pp.decode(0.5)
            for pp in SPACES[family]}
    return BUILDERS[family](full, spec)
