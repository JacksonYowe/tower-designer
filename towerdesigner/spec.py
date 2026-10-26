"""The design brief and analysis settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from .materials import GLUE_FACTOR, GLUES, WOODS, Glue, Wood


@dataclass
class Spec:
    # requirements
    height: float = 211.0          # mm, tower height to build (must be > min_height)
    min_height: float = 210.0
    pole_d: float = 25.0           # mm, test-jig pole
    clearance: float = 1.5         # mm, gap each side between pole and tower
    max_mass: float = 20.0         # g, wood + glue, includes the 3 % margins
    margin: float = 1.03           # +3 % on every piece and every joint's glue
    max_joints: int = 400          # build effort: most glue joints you're willing to make (0 = no limit)
    jig_slot: float = 3.3          # mm, slot a real stick slides into (from the printed fit-test coupon)

    # material
    wood: str = "teacher"
    measured_density: float = 0.0  # kg/m^3 from weighing your own wood (0 = use book value)
    measured_fc: float = 0.0       # MPa crushing strength from the wood test (0 = book value)
    measured_E: float = 0.0        # MPa stiffness along the grain from the wood test (0 = book value)
    quality: float = 0.85          # strength knock-down for hand-built, variable wood
    glue: str = "mixed"            # 'mixed' (best glue per joint) | 'auto' (best single glue) | a glue id
    glue_factor: float = GLUE_FACTOR  # hand-application multiplier on the glue model (all glues)
    glue_factors: dict = field(default_factory=dict)  # per-glue factors measured with the glue test
    symmetric_glue: bool = True    # same glue on every matching joint round the tower (load side unknown)

    # analysis
    imperfection: float = 0.004    # Perry-Robertson imperfection factor (higher = sloppier build)
    lateral: float = 0.01          # sideways load as a fraction of the vertical load
    uneven: float = 0.2            # most-loaded leg carries this much more than its share
    sway: float = 1.0              # mm, how far out of true the tower is built
    global_knockdown: float = 0.85 # applied to the whole-tower buckling load
    char: float = 0.15             # mm of charred (useless) wood on each laser-cut edge

    # optimiser
    objective: str = "load"        # 'load' (max load under mass limit) | 'efficiency' (load / mass)
    families: list[str] = field(default_factory=lambda: ["stick3", "diagrid3"])
    effort: str = "normal"         # 'quick' | 'normal' | 'thorough'
    allow_sticks: bool = True      # allow hand-cut Bunnings sticks as well as laser-cut strips
    bed: tuple[float, float, float] = (256.0, 256.0, 256.0)

    @property
    def bore_r(self) -> float:
        return self.pole_d / 2 + self.clearance

    def wood_props(self, factored: bool = True) -> Wood:
        w = WOODS[self.wood]
        if self.measured_density > 0:
            w = w.scaled_to_density(self.measured_density)
        if self.measured_fc > 0:   # measured crushing strength: scale every strength by the same ratio
            k = self.measured_fc / w.f_c_L
            w = replace(w, f_c_L=w.f_c_L * k, f_t_L=w.f_t_L * k, f_m=w.f_m * k, f_c_T=w.f_c_T * k,
                        f_t_T=w.f_t_T * k, f_v=w.f_v * k)
        if self.measured_E > 0:    # measured stiffness: scale every stiffness by the same ratio
            k = self.measured_E / w.E_L
            w = replace(w, E_L=w.E_L * k, E_T=w.E_T * k, G_LT=w.G_LT * k)
        return w.factored(self.quality) if factored else w

    def glue_props(self, key: str | None = None) -> Glue:
        k = key or self.glue
        return GLUES[k if k in GLUES else "pva"]

    def factors(self) -> dict[str, float]:
        """Hand factor for every glue: measured value if you ran the glue test, else the default."""
        return {g: float(self.glue_factors.get(g, self.glue_factor)) for g in GLUES}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bed"] = list(self.bed)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Spec":
        d = dict(d)
        if "bed" in d:
            d["bed"] = tuple(d["bed"])
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})
