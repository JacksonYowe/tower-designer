"""Strength checks, mass and the single number the optimiser maximises."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from .fea import P_REF, solve
from .materials import GLUE_FACTOR, GLUES, Glue, Wood, glue_amount, hankinson
from .spec import Spec
from .structure import Structure

G_ACC = 9.81


@dataclass
class MassReport:
    wood: float          # g, including +3 %
    glue: float          # g, including +3 %
    total: float
    per_part: list[float]
    per_joint: list[float]       # g dried, including +3 %
    laminate_glue: float
    per_joint_wet: list[float] = field(default_factory=list)   # g to apply (before drying), +3 %
    part_laminate: list[float] = field(default_factory=list)   # g dried inside each part (L/box legs)

    @property
    def glue_ratio(self) -> float:
        """Glue mass per gram of wood (rule of thumb: 5 g per 15 g = 0.33)."""
        return self.glue / self.wood if self.wood else 0.0


@dataclass
class Result:
    load_N: float
    load_kg: float
    mass: MassReport
    governing: str
    gov_kind: str
    lam_members: np.ndarray       # per member load factor (x P_REF)
    lam_joints: dict              # (member, end) or ('lap', joint) -> load factor
    lam_global: float
    util: np.ndarray              # per member utilisation at the failure load (1 = fails)
    glue: str                     # 'mixed' or the single glue used
    member_N: np.ndarray          # axial force per member at the failure load (N, tension +)
    joint_glue: list = field(default_factory=list)   # glue id at every joint
    laminate_glue_id: str = "pva"
    score: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def efficiency(self) -> float:
        return self.load_kg * 1000 / self.mass.total  # load held / own mass


def _factor(factor, glue: Glue) -> float:
    return factor.get(glue.id, GLUE_FACTOR) if isinstance(factor, dict) else factor


def mass_report(st: Structure, wood: Wood, glue: Glue, margin: float, factor=GLUE_FACTOR,
                joint_glues: list[Glue] | None = None) -> MassReport:
    """glue = glue for laminations (and every joint unless joint_glues gives one per joint).
    factor = one hand factor, or a dict of them per glue id."""
    rho = wood.rho
    per_part = [p.volume * rho * margin for p in st.parts]
    jg = joint_glues or [glue] * len(st.joints)
    amounts = [glue_amount(j.area, j.perimeter, j.end_grain, g, wood, _factor(factor, g))
               for j, g in zip(st.joints, jg)]
    per_joint = [a.dry * margin for a in amounts]
    per_wet = [a.wet * margin for a in amounts]
    # laminations (L-angle / box legs): a long glue line with a bead down both sides
    part_lam = []
    for p in st.parts:
        if p.laminate_area > 0 and p.sec is not None:
            per = 2 * p.laminate_area / max(p.sec.t, 0.5)
            part_lam.append(glue_amount(p.laminate_area, per, 0.0, glue, wood, _factor(factor, glue)).dry * margin)
        else:
            part_lam.append(0.0)
    lam_glue = sum(part_lam)
    w, g = sum(per_part), sum(per_joint) + lam_glue
    return MassReport(w, g, w + g, per_part, per_joint, lam_glue, per_wet, part_lam)


def calibrate_glue_factor(st: Structure, spec: Spec, glue_g: float, wood_g: float, glue_key: str = "pva") -> float:
    """Factor that makes this design carry glue_g of glue per wood_g of wood (e.g. from weighing a test)."""
    m = mass_report(st, spec.wood_props(False), GLUES[glue_key], 1.0, 1.0)
    return (glue_g / wood_g) / m.glue_ratio if m.glue_ratio > 0 else spec.glue_factor


def _charred(sec, char: float):
    """Section left after losing `char` mm of burnt wood on each laser-cut edge (the width edges)."""
    if not sec.laser or char <= 0 or sec.shape == "round":
        return sec
    return dataclasses.replace(sec, w=max(sec.w - 2 * char, 0.5))


def _perry(sigma_y, E, slender, a):
    se = np.pi ** 2 * E / np.maximum(slender, 1e-6) ** 2
    eta = a * slender
    phi = (sigma_y + (1 + eta) * se) / 2
    return phi - np.sqrt(np.maximum(phi ** 2 - sigma_y * se, 0.0))


def joint_classes(st: Structure) -> list[list[int]]:
    """Group joints that are the same joint repeated round the tower (rotational symmetry).

    Two joints match if they join the same kinds of piece, at the same height and radius, on the same
    side of the piece (left/right end). The real load can land on any side, so they must match."""
    groups: dict[tuple, list[int]] = {}
    for j, jt in enumerate(st.joints):
        p = st.nodes[jt.node]
        a, b = (st.parts[q] for q in jt.parts[:2])
        side = 0
        if jt.kind != "lap" and a.members:
            # which end of its piece this joint is: sign of the turn from the piece's middle to the node
            m = st.members[a.members[0]] if st.members[a.members[0]].i == jt.node or \
                st.members[a.members[0]].j == jt.node else st.members[a.members[-1]]
            mid = (st.nodes[m.i] + st.nodes[m.j]) / 2
            side = int(np.sign(mid[0] * p[1] - mid[1] * p[0]))
        key = (jt.kind, a.kind, b.kind, a.offset, b.offset, round(float(p[2]) * 2) / 2,
               round(float(np.hypot(p[0], p[1])) * 2) / 2, side)
        groups.setdefault(key, []).append(j)
    return list(groups.values())


def _solve_lambda(a, b, lam_g):
    """Load factor where lam * (a + b / (1 - lam/lam_g)) = 1 (b = part that grows near buckling)."""
    a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
    tot = a + b
    hi = np.where(tot > 0, 1.0 / np.maximum(tot, 1e-15), 1e9)
    if not np.isfinite(lam_g):
        return hi
    hi = np.minimum(hi, lam_g)
    lo = np.zeros_like(hi)
    for _ in range(40):
        mid = (lo + hi) / 2
        g = mid * (a + b / np.maximum(1 - mid / lam_g, 1e-9)) - 1
        lo, hi = np.where(g < 0, mid, lo), np.where(g < 0, hi, mid)
    return lo


def _worst(md, f, nm):
    """Per member: signed axial force with largest magnitude, and largest end moments."""
    N = np.zeros(nm)
    absN = np.zeros(nm)
    My = np.zeros(nm)
    Mz = np.zeros(nm)
    for e, mi in enumerate(md.member_of):
        if abs(f[e, 6]) > absN[mi]:
            absN[mi] = abs(f[e, 6])
            N[mi] = f[e, 6]
        My[mi] = max(My[mi], abs(f[e, 4]), abs(f[e, 10]))
        Mz[mi] = max(Mz[mi], abs(f[e, 5]), abs(f[e, 11]))
    return N, My, Mz


def analyse(st: Structure, spec: Spec, accurate: bool = False, glue_key: str | None = None) -> Result:
    wood = spec.wood_props()
    fe = solve(st, wood, spec.lateral, spec.uneven, spec.sway, subdivide=3 if accurate else 2)
    md = fe.model
    ms = st.members
    nm = len(ms)
    lam_g = fe.lam_global * spec.global_knockdown

    # ---- member forces (per unit of P_REF): perfect tower + extra from building it crooked ------
    N, My, Mz = _worst(md, fe.f_local, nm)
    Ni, Myi, Mzi = _worst(md, fe.f_imp, nm)
    Ni = np.abs(Ni)

    # ---- member capacity ---------------------------------------------------------------------
    grain = np.array([m.grain_deg for m in ms])
    # laser cutting chars each cut edge; that layer carries no load (mass is unchanged)
    secs = [_charred(m.sec, spec.char) for m in ms]
    A = np.array([sc.A for sc in secs])
    r = np.array([sc.r_min for sc in secs])
    L = np.array([st.member_len(m) for m in ms])
    K = np.array([m.K for m in ms])
    loc = np.array([sc.local_buckling_stress(wood) for sc in secs])
    E = hankinson(wood.E_L, wood.E_T, grain)
    fc = np.minimum(hankinson(wood.f_c_L, wood.f_c_T, grain), loc)
    ft = hankinson(wood.f_t_L, wood.f_t_T, grain)
    fm = hankinson(wood.f_m, 2 * wood.f_t_T, grain)
    sb = _perry(fc, E, K * L / r, spec.imperfection)
    Zy = np.array([(sc.w * sc.t ** 2 / 6) if sc.shape == "rect" else sc.Z_min for sc in secs])
    Zz = np.array([(sc.t * sc.w ** 2 / 6) if sc.shape == "rect" else sc.Z_min for sc in secs])
    Nc, Nt = A * sb, A * ft
    # if the crooked-tower force could flip the sign, check the weaker way
    cap_ax = np.where(N < 0, Nc, np.where(Ni > np.abs(N), np.minimum(Nc, Nt), Nt))
    frame = np.array([m.frame for m in ms])
    mom = np.where(frame, (My + Myi) / (fm * Zy) + (Mz + Mzi) / (fm * Zz), 0.0)
    lam_m = _solve_lambda(np.abs(N) / cap_ax, Ni / cap_ax + mom, lam_g)

    # ---- glue joints -------------------------------------------------------------------------
    # forces passed through each lap joint (perfect + imperfection), independent of glue
    laps = []
    for ji, j in enumerate(st.joints):
        if j.kind != "lap":
            continue
        pb = j.parts[1]
        F0, Fi = np.zeros(3), np.zeros(3)
        for e, mi in enumerate(md.member_of):
            if ms[mi].part != pb:
                continue
            Rt = md.R[e].T
            sl = slice(0, 3) if md.I[e] == j.node else slice(6, 9) if md.J[e] == j.node else None
            if sl is not None:
                F0 += Rt @ fe.f_local[e, sl]
                Fi += Rt @ fe.f_imp[e, sl]
        laps.append((ji, float(np.linalg.norm(F0)), float(np.linalg.norm(Fi))))

    # load factor of every checked joint with every glue
    joint_lam: dict = {}     # key -> {glue id: load factor}
    key_joint: dict = {}     # key -> index of the Joint whose glue decides it
    gusset_of: dict = {}     # gusset joint -> joints of the member ends it reinforces
    for gk, glue in GLUES.items():
        gs, fill = glue.strength, glue.gap_fill
        keys, a_, b_ = [], [], []
        for mi, m in enumerate(ms):
            for end, info in enumerate(m.ends):
                if not info:
                    continue
                F0, Fi = abs(N[mi]), Ni[mi]
                if F0 + Fi < 1e-12:
                    continue
                Ag = info["area"] * fill * info.get("bond", 1.0)
                gam = math.radians(info["gamma"])
                caps = []
                for tension in (True, False):
                    perp = wood.f_t_T if tension else wood.f_c_T * 1.5   # pulling off vs pushing in
                    cap_n = Ag * (min(perp, gs) if tension else perp)
                    cap_s = Ag * min(wood.f_v, gs)
                    cap = 1.0 / math.hypot(math.sin(gam) / cap_n, math.cos(gam) / cap_s)
                    gu = info.get("gusset")
                    if gu:
                        th = gu["theta"]
                        shear = gu["area"] * fill * min(wood.f_v, gs) * (1 - 0.4 * math.sin(math.radians(th)) ** 2)
                        net = gu["t"] * gu["size"] * 0.7 * float(hankinson(
                            wood.f_t_L if tension else wood.f_c_L, wood.f_t_T if tension else wood.f_c_T, th))
                        cap += min(shear, net)
                    caps.append(cap)
                cap = caps[0] if N[mi] > 0 else caps[1]
                if Fi > F0:
                    cap = min(caps)
                keys.append((mi, end))
                key_joint[(mi, end)] = info["joint"]
                if info.get("gusset"):
                    gusset_of.setdefault(info["gusset"]["joint"], set()).add(info["joint"])
                a_.append(F0 / cap)
                b_.append(Fi / cap)
        for ji, F0, Fi in laps:
            cap = st.joints[ji].area * fill * min(0.6 * wood.f_v, gs)
            keys.append(("lap", ji))
            key_joint[("lap", ji)] = ji
            a_.append(F0 / cap)
            b_.append(Fi / cap)
        lj = _solve_lambda(a_, b_, lam_g) if keys else np.array([])
        for k, v in zip(keys, lj.tolist()):
            joint_lam.setdefault(k, {})[gk] = v
    keys_of: dict = {}
    for k, ji in key_joint.items():
        keys_of.setdefault(ji, []).append(k)

    wood_raw = spec.wood_props(False)
    nj = len(st.joints)
    fac = spec.factors()
    dry = {g: [glue_amount(j.area, j.perimeter, j.end_grain, GLUES[g], wood_raw, fac[g]).dry
               for j in st.joints] for g in GLUES}
    # joints that are not strength-checked must at least not be weaker than the wood in shear
    structural = [g for g, gl in GLUES.items() if gl.strength * gl.gap_fill >= wood.f_v]
    # the tower can't be stronger than its weakest joint with the best possible glue, so no joint needs
    # to be stronger than that either (otherwise heavy glue is wasted on joints that can't fail first)
    best_joint = min((max(v.values()) for v in joint_lam.values()), default=np.inf)
    lam_target = min(float(lam_m.min()), lam_g, best_joint)

    classes = joint_classes(st) if spec.symmetric_glue else [[j] for j in range(nj)]

    def assign(mode: str) -> list[str]:
        """Glue for every joint. Joints in the same class (the matching joints all the way round the
        tower) get the same glue: the lightest one that is strong enough for the worst of them."""
        if mode in GLUES:
            return [mode] * nj
        choice = [None] * nj
        for cls in classes:
            ks = [k for ji in cls for k in keys_of.get(ji, [])]
            if ks:
                ok = [g for g in GLUES if all(joint_lam[k][g] >= lam_target * 0.999 for k in ks)]
                if ok:
                    g = min(ok, key=lambda g: sum(dry[g][ji] for ji in cls))
                else:  # no glue is strong enough: use the strongest
                    g = max(GLUES, key=lambda g: min(joint_lam[k][g] for k in ks))
            elif all(ji not in gusset_of for ji in cls):
                g = min(structural, key=lambda g: sum(dry[g][ji] for ji in cls))
            else:
                continue
            for ji in cls:
                choice[ji] = g
        for gj, ends in gusset_of.items():   # a gusset uses the strongest glue of the joints it backs up
            choice[gj] = max((choice[e] for e in ends), key=lambda g: GLUES[g].strength)
        return [c or "pva" for c in choice]

    if glue_key:
        modes = [glue_key]
    elif spec.glue == "auto":
        modes = list(GLUES)
    elif spec.glue == "mixed":
        modes = ["mixed"]
    else:
        modes = [spec.glue]
    best = None
    for mode in modes:
        choice = assign(mode)
        lam_j = {k: joint_lam[k][choice[key_joint[k]]] for k in key_joint}
        lam_glue = mode if mode in GLUES else min(structural, key=lambda g: GLUES[g].solids * GLUES[g].wet_density)
        mass = mass_report(st, wood_raw, GLUES[lam_glue], spec.margin, fac,
                           joint_glues=[GLUES[c] for c in choice])
        lam_jmin = min(lam_j.values()) if lam_j else np.inf
        lam = min(float(lam_m.min()), lam_jmin, lam_g)
        cand = (lam, mass, mode, lam_j, choice, lam_glue)
        if best is None or _better(cand, best, spec):
            best = cand
    lam, mass, gk, lam_j, joint_glue, lam_glue = best

    # ---- governing mode -------------------------------------------------------------------------
    i_m = int(np.argmin(lam_m))
    lam_jmin = min(lam_j.values()) if lam_j else np.inf
    if lam == lam_g:
        gov, kind = "Whole-tower buckling", "global"
    elif lam == lam_jmin:
        key = min(lam_j, key=lam_j.get)
        if key[0] == "lap":
            j = st.joints[key[1]]
            gov = f"Glued crossing ({st.parts[j.parts[0]].name} / {st.parts[j.parts[1]].name}) shears"
        else:
            m = ms[key[0]]
            how = "pulls off" if N[key[0]] > 0 else "crushes into"
            gov = f"Glue joint: {st.parts[m.part].name} {how} {st.parts[m.ends[key[1]]['host']].name}"
        kind = "joint"
    else:
        m = ms[i_m]
        if N[i_m] < 0:
            mode = "buckles" if sb[i_m] < 0.8 * fc[i_m] else "crushes"
        else:
            mode = "snaps in tension"
        gov = f"{st.parts[m.part].name} {mode}"
        if grain[i_m] > 10:
            gov += f" (grain at {grain[i_m]:.0f}°)"
        kind = "member"
    util = lam / np.maximum(lam_m, 1e-12)
    load_N = lam * P_REF
    res = Result(load_N, load_N / G_ACC, mass, gov, kind, lam_m * P_REF, {k: v * P_REF for k, v in lam_j.items()},
                 lam_g * P_REF, util, gk, N * lam, joint_glue, lam_glue)
    res.fe = fe
    res.n_joints = len(st.joints)
    res.n_parts = sum(1 for p in st.parts if p.kind != "corner")
    res.score = score(res, spec)
    return res


def _better(a, b, spec: Spec) -> bool:
    """Compare candidate tuples (lam, mass, glue, ...); prefer PVA unless another is clearly (>3 %) better."""
    sa = a[0] / (a[1].total if spec.objective == "efficiency" else 1.0)
    sb = b[0] / (b[1].total if spec.objective == "efficiency" else 1.0)
    over_a, over_b = a[1].total > spec.max_mass, b[1].total > spec.max_mass
    if over_a != over_b:
        return not over_a
    if over_a and over_b:
        return a[1].total < b[1].total
    if b[2] == "pva" and a[2] != "pva":
        return sa > sb * 1.03
    if sa == sb:
        return a[1].total < b[1].total
    return sa > sb


def score(res: Result, spec: Spec) -> float:
    base = res.load_kg if spec.objective == "load" else res.load_kg / res.mass.total
    if res.mass.total > spec.max_mass:
        base *= (spec.max_mass / res.mass.total) ** 12
    nj = getattr(res, "n_joints", 0)
    if spec.max_joints and nj > spec.max_joints:      # over the build-effort limit
        base *= (spec.max_joints / nj) ** 6
    return base * (1 - 1e-4 * nj)                     # tie-break: fewer joints is easier to build
