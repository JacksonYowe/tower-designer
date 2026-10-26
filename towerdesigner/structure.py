"""Data model for a tower: nodes, members (analysis elements) and parts (physical pieces)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sections import Section


class Infeasible(Exception):
    """The parameters do not describe a buildable tower."""


@dataclass
class Member:
    i: int
    j: int
    sec: Section
    kind: str               # 'leg' | 'brace' | 'ring' | 'cap' | 'base' | 'diag'
    part: int               # index into Structure.parts
    grain_deg: float = 0.0  # angle between member axis and wood grain
    frame: bool = True      # True: carries moments (continuous); False: pinned truss member
    K: float = 1.0          # effective length factor for member buckling
    ref: np.ndarray | None = None   # direction of local y (the strip's width)
    # glued ends: per end, None (continuous / support) or (host_part, glued_area_mm2, kind)
    ends: list = field(default_factory=lambda: [None, None])


@dataclass
class Part:
    """One physical piece of wood (it may span several members)."""
    name: str
    kind: str               # 'leg' | 'brace' | 'ring' | 'cap' | 'base' | 'panel' | 'diag' | 'gusset'
    sec: Section | None
    volume: float           # mm^3 of wood
    length: float = 0.0
    laminate_area: float = 0.0   # mm^2 of glue inside the part (L-angles, boxes, panel corners)
    members: list[int] = field(default_factory=list)
    laser: bool = True
    stock: str = ""
    # laser cutting: list of pieces; each piece is a list of closed polylines (first = outside,
    # the rest are holes). Grain runs along +x.
    outline: list[list[np.ndarray]] | None = None
    end_angles: tuple[float, float] = (90.0, 90.0)   # cut angle at each end (deg, 90 = square)
    step: int = 0                # assembly step this part is added in
    group: str = ""              # label shared by identical parts
    offset: float = 0.0          # radial offset in the 3D model (outer layer of a diagrid)
    info: dict = field(default_factory=dict)


@dataclass
class Joint:
    """A glued connection, used for glue mass and for the instructions."""
    node: int
    parts: tuple[int, ...]
    area: float             # mm^2 glued
    kind: str               # 'butt' | 'lap' | 'gusset' | 'notch' | 'edge'
    step: int = 0
    perimeter: float = 0.0  # mm of joint edge where a fillet of glue forms
    end_grain: float = 0.0  # mm^2 of the glued area that is end grain (soaks up glue)


@dataclass
class Structure:
    family: str
    params: dict
    nodes: np.ndarray                     # (N, 3)
    members: list[Member]
    parts: list[Part]
    joints: list[Joint]
    support_nodes: list[int]
    top_nodes: list[int]
    levels: list[float]                   # heights of node levels (for jigs and drawings)
    leg_parts: list[int]                  # parts that the jig collars must hold
    height: float
    info: dict = field(default_factory=dict)

    def member_vec(self, m: Member) -> np.ndarray:
        return self.nodes[m.j] - self.nodes[m.i]

    def member_len(self, m: Member) -> float:
        return float(np.linalg.norm(self.member_vec(m)))


def bore_clearance(p0: np.ndarray, p1: np.ndarray, half: float) -> float:
    """Smallest horizontal distance from the z axis to a segment, minus half the section size."""
    d = p1[:2] - p0[:2]
    dd = float(d @ d)
    s = 0.0 if dd < 1e-12 else min(1.0, max(0.0, -float(p0[:2] @ d) / dd))
    q = p0[:2] + s * d
    return float(np.hypot(*q)) - half
