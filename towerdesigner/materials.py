"""Wood, stock and glue data.

Units used throughout the app: mm, N, MPa (= N/mm^2), grams.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

KG_M3_TO_G_MM3 = 1e-6  # 1 kg/m^3 = 1e-6 g/mm^3


@dataclass(frozen=True)
class Wood:
    """Orthotropic wood. L = along the grain, T = across the grain."""
    name: str
    density: float  # kg/m^3
    E_L: float      # MPa
    E_T: float
    G_LT: float
    f_c_L: float    # compression parallel to grain
    f_t_L: float    # tension parallel to grain
    f_m: float      # bending (modulus of rupture)
    f_c_T: float    # compression perpendicular to grain (crushing)
    f_t_T: float    # tension perpendicular to grain (splitting)
    f_v: float      # shear parallel to grain
    note: str = ""

    @property
    def rho(self) -> float:
        """Density in g/mm^3."""
        return self.density * KG_M3_TO_G_MM3

    def scaled_to_density(self, density: float) -> "Wood":
        """Strength and stiffness of wood scale roughly linearly with density."""
        k = density / self.density
        return replace(
            self, density=density, E_L=self.E_L * k, E_T=self.E_T * k, G_LT=self.G_LT * k,
            f_c_L=self.f_c_L * k, f_t_L=self.f_t_L * k, f_m=self.f_m * k,
            f_c_T=self.f_c_T * k, f_t_T=self.f_t_T * k, f_v=self.f_v * k,
        )

    def factored(self, factor: float) -> "Wood":
        """Apply a build-quality/variability factor to strengths (not stiffness)."""
        return replace(
            self, f_c_L=self.f_c_L * factor, f_t_L=self.f_t_L * factor, f_m=self.f_m * factor,
            f_c_T=self.f_c_T * factor, f_t_T=self.f_t_T * factor, f_v=self.f_v * factor,
        )


def hankinson(f_par: float, f_perp: float, angle_deg, n: float = 2.0):
    """Strength (or stiffness) at an angle to the grain (Hankinson's formula)."""
    th = np.radians(angle_deg)
    s, c = np.abs(np.sin(th)), np.abs(np.cos(th))
    return f_par * f_perp / (f_par * s ** n + f_perp * c ** n)


# Values from published averages (Wood Database / USDA Wood Handbook), dried timber.
# Paulownia tomentosa: 280 kg/m^3, MOR 37.8 MPa, E 4.38 GPa, crushing 20.7 MPa.
# Balsa (Ochroma pyramidale), medium grade: 160 kg/m^3, MOR 19.6, E 3.71 GPa, crushing 11.6.
WOODS = {
    "paulownia": Wood(
        "Paulownia (Bunnings Boyle Hobby Wood)", 280, 4380, 260, 300,
        20.7, 30.0, 37.8, 2.6, 1.6, 4.5,
        note="What Bunnings actually sells as Boyle 'Hobby Wood'.",
    ),
    "teacher": Wood(
        "Supplied 3 × 3 sticks (run the wood test)", 160, 3710, 120, 160,
        11.6, 18.0, 19.6, 0.9, 0.7, 2.1,
        note="Starts from medium balsa values; the wood test replaces them with your measurements.",
    ),
    "balsa": Wood(
        "Balsa (medium grade)", 160, 3710, 120, 160,
        11.6, 18.0, 19.6, 0.9, 0.7, 2.1,
        note="Not in the current Bunnings range; included for comparison.",
    ),
}


@dataclass(frozen=True)
class Stock:
    """A product you can buy. kind: 'sheet', 'rect', 'square', 'dowel'."""
    id: str
    kind: str
    t: float        # thickness (sheet) or small side (stick)
    w: float        # width (sheet) or large side; dowel diameter in t and w
    length: float
    price: float
    title: str

    @property
    def is_sheet(self) -> bool:
        return self.kind == "sheet"


# Exactly the Boyle range you sent (Bunnings AU). All 915 mm long.
STOCK: list[Stock] = [
    Stock("sheet-1.5x75", "sheet", 1.5, 75, 915, 6.31, "Boyle 915 x 75 x 1.5mm Hobby Wood Sheet"),
    Stock("sheet-2.5x75", "sheet", 2.5, 75, 915, 6.63, "Boyle 915 x 75 x 2.5mm Hobby Wood Sheet"),
    Stock("sheet-3x75", "sheet", 3.0, 75, 915, 7.63, "Boyle 915 x 75 x 3mm Hobby Wood Sheet"),
    Stock("sheet-5x75", "sheet", 5.0, 75, 915, 9.01, "Boyle 915 x 75 x 5mm Hobby Wood Sheet"),
    Stock("sheet-6.5x75", "sheet", 6.5, 75, 915, 9.49, "Boyle 915 x 75 x 6.5mm Hobby Wood Sheet"),
    Stock("sheet-2.5x100", "sheet", 2.5, 100, 915, 8.37, "Boyle 915 x 100 x 2.5mm Hobby Wood Sheet"),
    Stock("sheet-5x100", "sheet", 5.0, 100, 915, 11.08, "Boyle 915 x 100 x 5mm Hobby Wood Sheet"),
    Stock("rect-3x9.5", "rect", 3.0, 9.5, 915, 2.49, "Boyle 915 x 3 x 9.5mm Hobby Wood Rectangle"),
    Stock("rect-3x12.5", "rect", 3.0, 12.5, 915, 3.18, "Boyle 915 x 3 x 12.5mm Hobby Wood Rectangle"),
    Stock("sq-5", "square", 5.0, 5.0, 915, 5.23, "Boyle 915 x 5 x 5mm Hobby Wood Square Rod"),
    Stock("sq-6.5", "square", 6.5, 6.5, 915, 4.88, "Boyle 915 x 6.5 x 6.5mm Square Hobby Wood Rod"),
    Stock("sq-9.5", "square", 9.5, 9.5, 915, 4.98, "Boyle 915 x 9.5 x 9.5mm Hobby Wood Square Rod"),
    Stock("sq-25", "square", 25.0, 25.0, 915, 12.51, "Boyle 915 x 25 x 25mm Hobby Wood Square Rod"),
    Stock("teacher-3x3", "square", 3.0, 3.0, 900, 0.0, "Teacher-supplied 3 × 3 mm stick"),
    Stock("dowel-5", "dowel", 5.0, 5.0, 915, 5.19, "Boyle 915 x 5 x 5mm Hobby Wood Dowel"),
    Stock("dowel-9.5", "dowel", 9.5, 9.5, 915, 6.32, "Boyle 915 x 9.5 x 9.5mm Hobby Wood Dowel"),
    Stock("dowel-12.5", "dowel", 12.5, 12.5, 915, 10.58, "Boyle 915 x 12.5 x 12.5mm Hobby Wood Dowel"),
]
STOCK_BY_ID = {s.id: s for s in STOCK}
SHEETS = [s for s in STOCK if s.is_sheet]
STICKS = [s for s in STOCK if not s.is_sheet and s.id not in ("sq-25", "teacher-3x3")]  # 25x25 alone weighs ~40 g
SHEET_THICKNESSES = sorted({s.t for s in SHEETS})


def sheet_for(t: float, min_width: float = 0.0) -> Stock:
    """Cheapest sheet of thickness t that is at least min_width wide."""
    options = [s for s in SHEETS if abs(s.t - t) < 1e-6 and s.w >= min_width]
    if not options:
        raise ValueError(f"No {t} mm sheet wide enough for {min_width:.0f} mm")
    return min(options, key=lambda s: s.price)


def max_sheet_width(t: float) -> float:
    return max((s.w for s in SHEETS if abs(s.t - t) < 1e-6), default=0.0)


@dataclass(frozen=True)
class Glue:
    id: str
    name: str
    product: str
    price: float
    viscosity: float     # mPa·s (thick PVA ~ 8,000; thin superglue ~ 100)
    wet_density: float   # g/cm^3 as it comes out of the bottle (mixed, for epoxy)
    solids: float        # fraction left after drying (PVA loses water, cement loses solvent)
    open_time: float     # s the glue stays liquid and keeps soaking into the wood
    shear: float         # MPa, bond shear strength on light timber
    joint_factor: float  # knock-down for brittleness
    gap_fill: float      # share of the nominal area that actually bonds on a hand-made joint
    sticks_to_pla: bool
    cure: str
    note: str
    where: str           # where to buy

    @property
    def strength(self) -> float:
        return self.shear * self.joint_factor


GLUES = {
    "pva": Glue(
        "pva", "PVA wood glue", "Selleys Aquadhere Quick Set PVA (interior)", 13.0,
        8000.0, 1.10, 0.50, 120.0, 7.0, 1.0, 0.9, False, "Hold 10–20 min, full strength after 24 h",
        "Stronger than the wood, forgiving, half of it is water that dries off. Does not stick to PLA jigs.",
        "Bunnings",
    ),
    "ca": Glue(
        "ca", "Superglue (thin CA)", "Loctite Super Glue liquid", 10.0,
        100.0, 1.06, 1.00, 8.0, 5.0, 0.85, 0.7, True, "Grabs in seconds, full strength in 1 h",
        "Very thin: tiny film and bead but soaks deep into end grain. Brittle, poor at filling gaps, sticks to PLA.",
        "Bunnings",
    ),
    "epoxy": Glue(
        "epoxy", "5-minute epoxy", "Selleys Araldite 5 Minute (mix 1:1)", 15.0,
        20000.0, 1.15, 1.00, 300.0, 12.0, 1.0, 1.0, True, "Work within 5 min, handle after 1 h, full strength 24 h",
        "Strongest and fills gaps perfectly, so it rescues a sloppy or highly loaded joint. Heavy: nothing evaporates.",
        "Bunnings",
    ),
    "cement": Glue(
        "cement", "Balsa cement", "UHU Hart balsa cement", 8.0,
        3000.0, 0.90, 0.30, 60.0, 3.0, 0.9, 0.8, False, "Tacky in 1–2 min, full strength after a few hours",
        "Lightest: about 70 % is solvent that evaporates. Weaker than the wood in shear, so only for lightly loaded joints.",
        "hobby / model shops (not Bunnings)",
    ),
}

# Hand application multiplier on the physical glue model. Calibrated so a typical student tower
# (4 legs of 5 x 5 stick, X-bracing of 2.5 x 3 mm strips, 6 storeys: 14.5 g of wood, 200 joints)
# carries a rule-of-thumb figure of 5 g of glue per 15 g of wood. The bare physics gives ~7 %;
# real hand gluing leaves ~4.8x that (bigger blobs, smearing, second coats).
GLUE_FACTOR = 4.8
TEACHER_RATIO = 5 / 15


@dataclass
class GlueAmount:
    film: float      # g wet, the layer squeezed between the two faces
    fillet: float    # g wet, the bead that forms round the edge of the joint
    soak: float      # g wet, what the wood absorbs (mostly end grain)
    wet: float       # g wet in total -> what you put on
    dry: float       # g left after drying -> what the tower carries


def glue_geometry(glue: Glue) -> tuple[float, float, float]:
    """(film thickness mm, fillet radius mm, soak depth mm) from the glue's viscosity.

    Thicker glue squeezes out less (thicker film) and stands up in a bigger bead; thinner glue
    wicks further into the wood's pores (Washburn: depth = sqrt(surface tension x pore radius
    x time / (2 x viscosity))).
    """
    eta = glue.viscosity
    film = float(np.clip(0.05 * (eta / 1000) ** (1 / 3), 0.015, 0.3))
    fillet = float(np.clip(0.8 * (eta / 1000) ** 0.25, 0.3, 2.5))
    gamma, pore = 0.035, 20e-6                     # N/m, m (paulownia/balsa vessel size)
    soak_m = np.sqrt(gamma * pore * glue.open_time / (2 * eta * 1e-3))
    soak = float(min(soak_m * 1000, 3.0))
    return film, fillet, soak


def glue_amount(area: float, perimeter: float, end_grain: float, glue: Glue, wood: Wood,
                factor: float = GLUE_FACTOR) -> GlueAmount:
    """Glue in one joint from its glued area, the length of its edges and how much end grain it has."""
    film, fillet, soak = glue_geometry(glue)
    porosity = max(0.0, 1 - wood.density / 1500)    # cell-wall material is ~1500 kg/m^3
    fill = 0.25                                       # share of the pores the glue actually reaches
    v_film = area * film
    v_fillet = perimeter * (1 - np.pi / 4) * fillet ** 2
    side = max(area - end_grain, 0.0)
    v_soak = (end_grain * soak + side * min(soak, 0.3)) * porosity * fill
    rho = glue.wet_density * 1e-3                     # g/mm^3
    f, b, s = v_film * rho * factor, v_fillet * rho * factor, v_soak * rho * factor
    wet = f + b + s
    return GlueAmount(f, b, s, wet, wet * glue.solids)

MASS_MARGIN = 1.03  # +3 % on every piece and on the glue of every joint (your rule)
