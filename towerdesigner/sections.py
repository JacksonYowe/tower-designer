"""Cross-sections a member can have.

Local axes: x along the member, y = the 'width' direction, z = the 'thickness' direction.
For strips laid in a tower face, z is the face normal (out of plane) and y lies in the face.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

from .materials import STOCK_BY_ID, Wood, sheet_for


@dataclass(frozen=True)
class Section:
    shape: str      # 'rect' | 'round' | 'angle' | 'box'
    t: float        # thickness (rect: z size; angle/box: wall; round: diameter)
    w: float        # width (rect: y size; angle: leg length; box: outer size; round: diameter)
    stock: str      # stock id the piece is cut from
    laser: bool     # True = laser-cut strip(s) from a sheet, False = stick cut by hand

    # ---- geometry -------------------------------------------------------------------------
    @cached_property
    def A(self) -> float:
        t, w = self.t, self.w
        if self.shape == "rect":
            return t * w
        if self.shape == "round":
            return math.pi * w * w / 4
        if self.shape == "angle":
            return t * (2 * w - t)
        if self.shape == "box":
            return w * w - (w - 2 * t) ** 2
        raise ValueError(self.shape)

    @cached_property
    def _I(self) -> tuple[float, float, float]:
        """(I_y, I_z, J). I_y resists bending in the z direction, I_z in the y direction."""
        t, w = self.t, self.w
        if self.shape == "rect":
            a, b = max(t, w), min(t, w)
            J = a * b ** 3 * (1 / 3 - 0.21 * b / a * (1 - b ** 4 / (12 * a ** 4)))
            return w * t ** 3 / 12, t * w ** 3 / 12, J
        if self.shape == "round":
            I = math.pi * w ** 4 / 64
            return I, I, 2 * I
        if self.shape == "angle":
            # two rectangles: (w x t) along y, (t x (w - t)) along z; principal axes
            A1, A2 = w * t, t * (w - t)
            c1 = (w / 2, t / 2)
            c2 = (t / 2, t + (w - t) / 2)
            A = A1 + A2
            cy = (A1 * c1[0] + A2 * c2[0]) / A
            cz = (A1 * c1[1] + A2 * c2[1]) / A
            Iyy = (w * t ** 3 / 12 + A1 * (c1[1] - cz) ** 2) + (t * (w - t) ** 3 / 12 + A2 * (c2[1] - cz) ** 2)
            Izz = (t * w ** 3 / 12 + A1 * (c1[0] - cy) ** 2) + ((w - t) * t ** 3 / 12 + A2 * (c2[0] - cy) ** 2)
            Iyz = A1 * (c1[0] - cy) * (c1[1] - cz) + A2 * (c2[0] - cy) * (c2[1] - cz)
            m, d = (Iyy + Izz) / 2, math.hypot((Iyy - Izz) / 2, Iyz)
            J = (w * t ** 3 + (w - t) * t ** 3) / 3
            return m - d, m + d, J  # minor, major (orientation of angle is not tracked)
        if self.shape == "box":
            I = (w ** 4 - (w - 2 * t) ** 4) / 12
            J = (w - t) ** 3 * t
            return I, I, J
        raise ValueError(self.shape)

    @property
    def Iy(self) -> float:
        return self._I[0]

    @property
    def Iz(self) -> float:
        return self._I[1]

    @property
    def J(self) -> float:
        return self._I[2]

    @property
    def I_min(self) -> float:
        return min(self.Iy, self.Iz)

    @property
    def r_min(self) -> float:
        return math.sqrt(self.I_min / self.A)

    @cached_property
    def Z_min(self) -> float:
        """Smallest elastic section modulus (for bending stress)."""
        if self.shape == "rect":
            return min(self.w * self.t ** 2, self.t * self.w ** 2) / 6
        if self.shape == "round":
            return math.pi * self.w ** 3 / 32
        if self.shape == "angle":
            return self.I_min / (0.71 * self.w)
        return self.I_min / (self.w / 2)

    @property
    def size_y(self) -> float:
        return self.w

    @property
    def size_z(self) -> float:
        return self.t if self.shape == "rect" else self.w

    @property
    def half_extent(self) -> float:
        """Radius of a circle that contains the section (for clearance checks)."""
        if self.shape == "rect":
            return math.hypot(self.t, self.w) / 2
        if self.shape == "round":
            return self.w / 2
        if self.shape == "angle":
            return 0.8 * self.w
        return self.w / math.sqrt(2)

    # ---- manufacturing ----------------------------------------------------------------------
    @property
    def strips(self) -> list[tuple[float, float]]:
        """(thickness, width) of each strip laminated together to make this section."""
        t, w = self.t, self.w
        if self.shape == "angle":
            return [(t, w), (t, w - t)]
        if self.shape == "box":
            return [(t, w), (t, w), (t, w - 2 * t), (t, w - 2 * t)]
        return [(t, w)]

    @property
    def lamination_bond(self) -> float:
        """Glued bond width per mm of length (mm^2 of glue per mm of member)."""
        return {"angle": 1, "box": 4}.get(self.shape, 0) * self.t

    def local_buckling_stress(self, wood: Wood) -> float:
        """Plate buckling of the thin walls of angle/box sections (orthotropic, long plates)."""
        t, w = self.t, self.w
        if self.shape == "angle":
            b = w - t / 2
            return wood.G_LT * (t / b) ** 2 + 0.25 * wood.E_T * (t / b) ** 2
        if self.shape == "box":
            b = w - t
            return (math.pi ** 2 * t ** 2 / (6 * b ** 2)) * (
                math.sqrt(wood.E_L * wood.E_T) + 0.02 * wood.E_T + 2 * wood.G_LT)
        return math.inf

    def label(self) -> str:
        if self.shape == "round":
            return f"Ø{self.w:g} dowel"
        if self.stock == "teacher-3x3":
            if self.t < 2.95 or self.w < 2.95:
                return f"3×3 stick shaved to {self.t:.1f} × {self.w:.1f}"
            return "3 × 3 mm stick"
        if not self.laser:
            return STOCK_BY_ID[self.stock].title.replace("Boyle 915 x ", "").replace(" Hobby Wood", "")
        name = {"rect": "strip", "angle": "L-angle", "box": "box"}[self.shape]
        return f"{self.t:g} × {self.w:.1f} mm {name} (laser-cut)"


def stick_section(stock_id: str) -> Section:
    s = STOCK_BY_ID[stock_id]
    if s.kind == "dowel":
        return Section("round", s.w, s.w, stock_id, False)
    return Section("rect", s.t, s.w, stock_id, False)


STICK3 = 3.0
MIN_SHAVE = 1.5   # mm, thinnest you can sensibly sand a 3 x 3 stick down to


def shaved_section(shape: str, t: float, w: float) -> Section:
    """A supplied 3 x 3 stick, optionally sanded thinner (never thicker than 3 mm)."""
    t = min(max(float(t), MIN_SHAVE), STICK3)
    w = min(max(float(w), MIN_SHAVE), STICK3)
    return Section("rect", t, w, "teacher-3x3", False)


def strip_section(shape: str, t: float, w: float) -> Section:
    """A section laser cut from a sheet of thickness t."""
    widest = max(st[1] for st in Section(shape, t, w, "", True).strips)
    stock = sheet_for(t, min(widest, 75)).id
    return Section(shape, t, w, stock, True)
