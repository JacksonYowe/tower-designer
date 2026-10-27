"""Design search: educated guesses -> evolution -> brute force -> polish.

Every candidate is first *fitted to the mass budget*: all the width-type parameters are scaled
together (by bisection) until the tower weighs just under the limit, so designs are always
compared at the same mass.
"""
from __future__ import annotations

import itertools
import math
import os
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .analysis import analyse, mass_report
from .families import FAMILY_NAMES, SPACES, Param, build
from .materials import GLUES, max_sheet_width
from .sections import shaved_section, strip_section, stick_section
from .spec import Spec
from .structure import Infeasible

EFFORT = {
    "quick": dict(pop=20, gens=10, brute=120, sweeps=1, keep=1),
    "normal": dict(pop=36, gens=28, brute=500, sweeps=2, keep=2),
    "thorough": dict(pop=60, gens=60, brute=1600, sweeps=3, keep=3),
}


@dataclass
class Candidate:
    family: str
    params: dict
    score: float
    load_kg: float = 0.0
    mass: float = 0.0
    governing: str = ""
    glue: str = ""
    stage: str = ""
    ok: bool = True
    reason: str = ""

    def key(self) -> tuple:
        out = []
        for p in SPACES[self.family]:
            v = self.params.get(p.name)
            out.append(round(v, 1) if isinstance(v, float) else v)
        return (self.family, *out)


# ---------------------------------------------------------------------------------------------
# evaluation (runs in worker processes)
# ---------------------------------------------------------------------------------------------
def _scaled(family: str, params: dict, s: float) -> dict:
    out = dict(params)
    for p in SPACES[family]:
        if p.width:
            out[p.name] = p.clamp(params[p.name] * s)
    return out


def fit_to_budget(family: str, params: dict, spec: Spec, extra: float = 0.0) -> dict:
    """Scale all widths together so the tower just fits the mass limit."""
    wood = spec.wood_props(factored=False)
    glue = GLUES[spec.glue if spec.glue in GLUES else "pva"]
    target = spec.max_mass * 0.997 - extra

    def mass(s):
        try:
            st = build(family, _scaled(family, params, s), spec)
        except Infeasible:
            return None
        return mass_report(st, wood, glue, spec.margin, spec.factors()).total

    m1 = mass(1.0)
    lo, hi = 0.2, 4.0
    m_lo, m_hi = mass(lo), mass(hi)
    if m_hi is not None and m_hi <= target:
        return _scaled(family, params, hi)
    if m_lo is not None and m_lo > target:
        return _scaled(family, params, lo)
    if m_lo is None and m_hi is None and m1 is None:
        return params
    best = 1.0 if (m1 is not None and m1 <= target) else lo
    a, b = lo, hi
    for _ in range(18):
        mid = math.sqrt(a * b)
        m = mass(mid)
        if m is None:
            # infeasible part-way (e.g. a frame too small): shrink the bracket towards 1
            if mid < 1.0:
                a = mid
            else:
                b = mid
            continue
        if m <= target:
            best, a = mid, mid
        else:
            b = mid
        if b / a < 1.003:
            break
    return _scaled(family, params, best)


def evaluate(family: str, params: dict, spec: Spec, fit: bool = True, accurate: bool = False,
             stage: str = "") -> Candidate:
    try:
        if fit and spec.objective == "load":
            params = fit_to_budget(family, params, spec)
        st = build(family, params, spec)
        r = analyse(st, spec, accurate=accurate)
        if fit and spec.objective == "load" and r.mass.total > spec.max_mass:
            # the per-joint glue choice came out heavier than the PVA estimate: refit once
            params = fit_to_budget(family, params, spec, extra=r.mass.total - spec.max_mass * 0.997 + 0.02)
            st = build(family, params, spec)
            r = analyse(st, spec, accurate=accurate)
    except Infeasible as e:
        return Candidate(family, params, -1.0, ok=False, reason=str(e), stage=stage)
    except (np.linalg.LinAlgError, RuntimeError, ValueError, ZeroDivisionError, FloatingPointError) as e:
        return Candidate(family, params, -1.0, ok=False, reason=f"analysis failed: {e}", stage=stage)
    full = {p.name: st.params[p.name] for p in SPACES[family]}
    return Candidate(family, full, r.score, r.load_kg, r.mass.total, r.governing, r.glue, stage,
                     ok=r.mass.total <= spec.max_mass + 1e-6 and
                     (not spec.max_joints or r.n_joints <= spec.max_joints))


def _task(args):
    family, params, spec_d, fit, accurate, stage = args
    return evaluate(family, params, Spec.from_dict(spec_d), fit, accurate, stage)


# ---------------------------------------------------------------------------------------------
# educated guesses
# ---------------------------------------------------------------------------------------------
def _panels_for(sec, spec: Spec, ratio: float = 2.5) -> int:
    """Panel count so a leg segment's Euler stress is ~ratio x its crushing strength."""
    w = spec.wood_props()
    seg = math.pi * sec.r_min * math.sqrt(w.E_L / (ratio * w.f_c_L))
    return int(min(12, max(2, math.ceil(spec.height / seg))))


def seeds(family: str, spec: Spec) -> list[dict]:
    H = spec.height
    out = []
    frame = dict(cap_t=1.5, cap_w=8.0, cap_mode="frame", base_t=1.5, base_w=3.0, base_mode="strips")
    if family == "stick":
        legs = [("strip", 5.0, 5.0), ("strip", 6.5, 6.5), ("angle", 1.5, 8.0), ("box", 1.5, 8.0),
                ("angle", 2.5, 9.0), ("box", 1.5, 10.0)]
        if spec.allow_sticks:
            legs += [("sq-5", 5.0, 5.0), ("sq-6.5", 6.5, 6.5)]
        for n in (3, 4, 5, 6):
            for lt, t, w in legs:
                sec = stick_section(lt) if lt.startswith(("sq", "rect", "dowel")) else strip_section(
                    "rect" if lt == "strip" else lt, t, w)
                npn = _panels_for(sec, spec)
                for pattern in ("Z", "X"):
                    out.append(dict(n_sides=n, base_r=H / 4.4, top_ratio=0.55, n_panels=npn, panel_ratio=1.0,
                                    pattern=pattern, leg_type=lt, leg_t=t, leg_w=w, brace_type="strip",
                                    brace_t=1.5, brace_w=3.0, ring_t=1.5, ring_w=3.0, gusset=0.0, **frame))
    elif family == "panel":
        for n in (3, 4, 5, 6):
            for t in (1.5, 2.5, 3.0):
                edge = 6.0
                sw = max_sheet_width(t)
                rb_max = (sw - 6 - edge) / (2 * math.sin(math.pi / n))
                rb = min(H / 4.4, rb_max)
                sec = strip_section("angle", t, edge)
                for pattern in ("X", "Z", "W"):
                    out.append(dict(n_sides=n, base_r=rb, top_ratio=0.6, n_panels=_panels_for(sec, spec),
                                    panel_ratio=1.0, pattern=pattern, panel_t=t, edge_w=edge, diag_w=3.0,
                                    ring_w=3.0, cap_t=1.5, cap_w=8.0, cap_mode="frame"))
    elif family == "diagrid":
        for n in (4, 5, 6, 8, 10):
            for tw in (0.34, 0.5, 0.67):
                for t in (2.5, 3.0, 5.0):
                    out.append(dict(n_sides=n, twist=tw, base_r=H / 4.4, top_ratio=0.6, member_type="strip",
                                    member_t=t, member_w=max(4.0, t), ring_t=1.5, ring_w=2.5, ring_every=1,
                                    **frame))
    elif family == "stick3":
        leg = shaved_section("rect", 3.0, 3.0)
        npn = _panels_for(leg, spec, ratio=2.0)
        for n in (4, 6, 8, 10, 12, 16):
            for pattern in ("Z", "X", "W"):
                for joint in ("mitre", "lap"):
                    for np_ in sorted({max(2, npn - 2), npn, min(16, npn + 2)}):
                        out.append(dict(n_sides=n, base_r=H / 4.4, top_ratio=0.6, n_panels=np_, panel_ratio=1.0,
                                        pattern=pattern, joint=joint))
    elif family == "diagrid3":
        for n in (8, 12, 16, 20, 24):
            for tw in (0.25, 0.34, 0.5):
                for joint, rj in (("mitre", "half lap"), ("lap", "half lap"), ("lap", "shingled lap")):
                    out.append(dict(n_sides=n, twist=tw, base_r=H / 4.4, top_ratio=0.6, ring_every=1, joint=joint,
                                    ring_joint=rj))
    return out


# ---------------------------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------------------------
@dataclass
class SearchState:
    best: list[Candidate] = field(default_factory=list)
    evaluated: int = 0
    log: list[str] = field(default_factory=list)


class Optimiser:
    def __init__(self, spec: Spec, progress=None, stop: threading.Event | None = None, workers: int | None = None):
        self.spec = spec
        self.cfg = EFFORT[spec.effort]
        self.progress = progress or (lambda frac, msg, best: None)
        self.stop = stop or threading.Event()
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self.state = SearchState()
        self.seen: dict[tuple, Candidate] = {}
        self.rng = random.Random(1)

    # -- helpers ---------------------------------------------------------------------------------
    def _log(self, msg: str):
        self.state.log.append(msg)

    def _run(self, pool, jobs: list[tuple[str, dict]], stage: str, fit=True, accurate=False) -> list[Candidate]:
        sd = self.spec.to_dict()
        args = [(f, p, sd, fit, accurate, stage) for f, p in jobs]
        if pool is None:
            res = [_task(a) for a in args]
        else:
            res = list(pool.map(_task, args, chunksize=max(1, len(args) // (self.workers * 4))))
        self.state.evaluated += len(res)
        for c in res:
            if c.ok and c.score > 0:
                k = c.key()
                if k not in self.seen or self.seen[k].score < c.score:
                    self.seen[k] = c
        return res

    def _best(self, family=None, n=1) -> list[Candidate]:
        pool = [c for c in self.seen.values() if c.ok and (family is None or c.family == family)]
        return sorted(pool, key=lambda c: -c.score)[:n]

    def _report(self, frac, msg):
        b = self._best()
        self.progress(frac, msg, b[0] if b else None)

    # -- stages ----------------------------------------------------------------------------------
    def run(self) -> list[Candidate]:
        fams = [f for f in self.spec.families if f in SPACES]
        t0 = time.time()
        pool = ProcessPoolExecutor(self.workers) if self.workers > 1 else None
        try:
            # 1. educated guesses
            self._report(0.01, "Stage 1/4: educated guesses")
            jobs = [(f, s) for f in fams for s in seeds(f, self.spec)]
            self._run(pool, jobs, "guess")
            for f in fams:
                b = self._best(f)
                if b:
                    self._log(f"Guess  {FAMILY_NAMES[f]:22s} {b[0].load_kg:6.1f} kg  ({b[0].governing})")
            self._report(0.12, "Stage 2/4: evolution search")
            if self.stop.is_set():
                return self._finish(pool)

            # 2. differential evolution, per family
            for fi, f in enumerate(fams):
                self._evolve(pool, f, 0.12 + 0.43 * fi / len(fams), 0.43 / len(fams))
                if self.stop.is_set():
                    return self._finish(pool)

            # 3. brute force around the best designs
            self._report(0.55, "Stage 3/4: brute force")
            tops = []
            for f in fams:
                tops += self._best(f, self.cfg["keep"])
            for ti, c in enumerate(tops):
                self._brute(pool, c, 0.55 + 0.3 * ti / max(1, len(tops)), 0.3 / max(1, len(tops)))
                if self.stop.is_set():
                    return self._finish(pool)

            # 4. polish: coordinate sweeps over every parameter
            self._report(0.85, "Stage 4/4: polishing")
            for f in fams:
                b = self._best(f)
                if b:
                    self._sweep(pool, b[0])
                if self.stop.is_set():
                    break
            return self._finish(pool)
        finally:
            if pool:
                pool.shutdown(cancel_futures=True)
            self._log(f"{self.state.evaluated} designs analysed in {time.time() - t0:.0f} s")

    def _finish(self, pool) -> list[Candidate]:
        self._report(0.95, "Final high-accuracy check")
        # one or two of each family, then the overall best
        picks, keys = [], set()
        for f in self.spec.families:
            for c in self._best(f, 2):
                picks.append(c)
        for c in self._best(None, 6):
            picks.append(c)
        uniq = []
        for c in picks:   # one per distinct layout (ignore tiny width differences)
            k = (c.family,) + tuple(c.params[p.name] for p in SPACES[c.family] if p.kind != "float")
            if k not in keys:
                keys.add(k)
                uniq.append(c)
        final = self._run(pool, [(c.family, c.params) for c in uniq], "final", fit=False, accurate=True)
        final = sorted([c for c in final if c.ok], key=lambda c: -c.score)
        dedup = []
        for c in final:   # drop designs that perform identically (differences that change nothing)
            if not any(d.family == c.family and abs(d.load_kg - c.load_kg) < 0.05 and abs(d.mass - c.mass) < 0.02
                       for d in dedup):
                dedup.append(c)
        final = dedup
        self.state.best = final
        self.progress(1.0, "Done", final[0] if final else None)
        return final

    def _evolve(self, pool, family: str, f0: float, span: float):
        space = SPACES[family]
        d = len(space)
        pop_n, gens = self.cfg["pop"], self.cfg["gens"]
        seeded = [c for c in self._best(family, pop_n // 2)]
        X = [np.array([p.encode(c.params[p.name]) for p in space]) for c in seeded]
        S = [c.score for c in seeded]
        # Latin-hypercube style random fill
        need = pop_n - len(X)
        if need > 0:
            cols = [np.random.default_rng(self.rng.randrange(1 << 30)).permutation(need) for _ in range(d)]
            rand = [np.array([(cols[j][i] + self.rng.random()) / need for j in range(d)]) for i in range(need)]
            res = self._run(pool, [(family, self._decode(space, x)) for x in rand], "evolve")
            for x, c in zip(rand, res):
                X.append(np.array([p.encode(c.params[p.name]) for p in space]) if c.ok else x)
                S.append(c.score)
        X, S = np.array(X), np.array(S)
        for g in range(gens):
            if self.stop.is_set():
                return
            trials = []
            for i in range(len(X)):
                r1, r2, r3 = self.rng.sample([k for k in range(len(X)) if k != i], 3)
                best_i = int(np.argmax(S))
                F = 0.5 + 0.3 * self.rng.random()
                base = X[best_i] if self.rng.random() < 0.3 else X[r1]   # current-to-best now and then
                v = np.clip(base + F * (X[r2] - X[r3]), 0, 1)
                mask = np.array([self.rng.random() < 0.8 for _ in range(d)])
                mask[self.rng.randrange(d)] = True
                trials.append(np.where(mask, v, X[i]))
            res = self._run(pool, [(family, self._decode(space, x)) for x in trials], "evolve")
            for i, (x, c) in enumerate(zip(trials, res)):
                if c.score > S[i]:
                    X[i] = np.array([p.encode(c.params[p.name]) for p in space])
                    S[i] = c.score
            if g % 3 == 0 or g == gens - 1:
                self._report(f0 + span * (g + 1) / gens,
                             f"Evolving {FAMILY_NAMES[family]} — generation {g + 1}/{gens}, best {S.max():.1f}")
        b = self._best(family)
        if b:
            self._log(f"Evolve {FAMILY_NAMES[family]:22s} {b[0].load_kg:6.1f} kg  ({b[0].governing})")

    @staticmethod
    def _decode(space: list[Param], x) -> dict:
        return {p.name: p.decode(v) for p, v in zip(space, x)}

    def _grid_values(self, p: Param, current):
        if p.kind == "choice":
            return list(p.choices)
        if p.kind == "int":
            lo, hi = int(p.lo), int(p.hi)
            if p.name == "n_panels":
                lo, hi = max(lo, current - 3), min(hi, current + 3)
            return list(range(lo, hi + 1))
        return None

    def _brute(self, pool, c: Candidate, f0, span):
        """Every combination of the discrete choices around a good design, each fitted to budget."""
        space = SPACES[c.family]
        disc = [p for p in space if p.kind != "float"]
        grids = [self._grid_values(p, c.params[p.name]) for p in disc]
        total = math.prod(len(g) for g in grids)
        budget = self.cfg["brute"]
        if total <= budget:
            combos = list(itertools.product(*grids))
        else:
            # too many: always try every value of each choice on its own, then random combinations
            combos = set()
            for i, g in enumerate(grids):
                for v in g:
                    base = [c.params[p.name] for p in disc]
                    base[i] = v
                    combos.add(tuple(base))
            while len(combos) < budget:
                combos.add(tuple(self.rng.choice(g) for g in grids))
            combos = list(combos)
        jobs = []
        for combo in combos:
            p2 = dict(c.params)
            p2.update({p.name: v for p, v in zip(disc, combo)})
            jobs.append((c.family, p2))
        n = len(jobs)
        step = max(1, n // 5)
        for k in range(0, n, step):
            if self.stop.is_set():
                return
            self._run(pool, jobs[k:k + step], "brute")
            self._report(f0 + span * min(1.0, (k + step) / n),
                         f"Brute force {FAMILY_NAMES[c.family]}: {min(n, k + step)}/{n} combinations")
        b = self._best(c.family)
        if b:
            self._log(f"Brute  {FAMILY_NAMES[c.family]:22s} {b[0].load_kg:6.1f} kg  ({b[0].governing})")

    def _sweep(self, pool, c: Candidate):
        """Coordinate search: try every value of each parameter in turn, keep improvements."""
        space = SPACES[c.family]
        cur = c
        for rnd in range(self.cfg["sweeps"]):
            improved = False
            for p in space:
                if self.stop.is_set():
                    return
                if p.kind == "float":
                    v0 = cur.params[p.name]
                    width = (p.hi - p.lo) * (0.25 / (rnd + 1))
                    vals = [p.clamp(v0 + width * k / 4) for k in range(-4, 5) if k]
                else:
                    vals = self._grid_values(p, cur.params[p.name])
                jobs = []
                for v in vals:
                    p2 = dict(cur.params)
                    p2[p.name] = v
                    jobs.append((c.family, p2))
                res = self._run(pool, jobs, "polish")
                good = [r for r in res if r.ok]
                if good:
                    bst = max(good, key=lambda r: r.score)
                    if bst.score > cur.score * 1.001:
                        cur, improved = bst, True
            if not improved:
                break
        self._log(f"Polish {FAMILY_NAMES[c.family]:22s} {cur.load_kg:6.1f} kg  ({cur.governing})")
