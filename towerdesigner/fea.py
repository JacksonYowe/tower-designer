"""3D frame finite-element solver: linear static + linear (eigenvalue) buckling.

Each node has 6 degrees of freedom (ux, uy, uz, rx, ry, rz). Members marked frame=True are
Euler-Bernoulli beams; frame=False members are pinned-end truss bars (axial only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .materials import Wood, hankinson
from .structure import Structure

P_REF = 100.0  # N, reference load the analysis is run at (results scale linearly)


@dataclass
class FEModel:
    nodes: np.ndarray
    I: np.ndarray
    J: np.ndarray
    E: np.ndarray
    G: np.ndarray
    A: np.ndarray
    Iy: np.ndarray
    Iz: np.ndarray
    Jt: np.ndarray
    frame: np.ndarray
    L: np.ndarray
    R: np.ndarray           # (Ne, 3, 3) rows = local x, y, z
    member_of: np.ndarray   # element -> member index


def build_model(st: Structure, wood: Wood, subdivide: int = 1) -> FEModel:
    nodes = [*st.nodes]
    I, J, mem = [], [], []
    for mi, m in enumerate(st.members):
        nseg = subdivide if m.frame else 1
        a = m.i
        for s in range(1, nseg + 1):
            if s < nseg:
                nodes.append(st.nodes[m.i] + (st.nodes[m.j] - st.nodes[m.i]) * s / nseg)
                b = len(nodes) - 1
            else:
                b = m.j
            I.append(a)
            J.append(b)
            mem.append(mi)
            a = b
    nodes = np.array(nodes)
    I, J, mem = np.array(I), np.array(J), np.array(mem)
    ms = st.members
    grain = np.array([m.grain_deg for m in ms])[mem]
    E = hankinson(wood.E_L, wood.E_T, grain)
    G = np.full(len(I), wood.G_LT)
    A = np.array([m.sec.A for m in ms])[mem]
    Iy = np.array([m.sec.Iy for m in ms])[mem]
    Iz = np.array([m.sec.Iz for m in ms])[mem]
    Jt = np.array([m.sec.J for m in ms])[mem]
    frame = np.array([m.frame for m in ms])[mem]
    x = nodes[J] - nodes[I]
    L = np.linalg.norm(x, axis=1)
    x = x / L[:, None]
    ref = np.array([m.ref if m.ref is not None else (0.0, 0.0, 1.0) for m in ms], float)[mem]
    y = ref - (np.einsum("ij,ij->i", ref, x))[:, None] * x
    bad = np.linalg.norm(y, axis=1) < 1e-6
    if bad.any():
        alt = np.where(np.abs(x[bad, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])
        y[bad] = alt - (np.einsum("ij,ij->i", alt, x[bad]))[:, None] * x[bad]
    y /= np.linalg.norm(y, axis=1)[:, None]
    z = np.cross(x, y)
    R = np.stack([x, y, z], axis=1)
    return FEModel(nodes, I, J, E, G, A, Iy, Iz, Jt, frame, L, R, mem)


def _k_local(md: FEModel) -> np.ndarray:
    n = len(md.L)
    L, E = md.L, md.E
    k = np.zeros((n, 12, 12))
    X = E * md.A / L
    k[:, 0, 0] = k[:, 6, 6] = X
    k[:, 0, 6] = k[:, 6, 0] = -X
    f = md.frame.astype(float)
    T = f * md.G * md.Jt / L
    k[:, 3, 3] = k[:, 9, 9] = T
    k[:, 3, 9] = k[:, 9, 3] = -T
    for (v, r, Ii, sgn) in ((1, 5, md.Iz, 1.0), (2, 4, md.Iy, -1.0)):
        EI = f * E * Ii
        a, b, c, d = 12 * EI / L ** 3, 6 * EI / L ** 2, 4 * EI / L, 2 * EI / L
        v2, r2 = v + 6, r + 6
        k[:, v, v] = k[:, v2, v2] = a
        k[:, v, v2] = k[:, v2, v] = -a
        for (i, j, val) in ((v, r, sgn * b), (v, r2, sgn * b), (v2, r, -sgn * b), (v2, r2, -sgn * b)):
            k[:, i, j] = k[:, j, i] = val
        k[:, r, r] = k[:, r2, r2] = c
        k[:, r, r2] = k[:, r2, r] = d
    return k


def _kg_local(md: FEModel, N: np.ndarray) -> np.ndarray:
    n = len(md.L)
    L = md.L
    kg = np.zeros((n, 12, 12))
    fr = md.frame
    c = N / (30 * L)
    for (idx, s) in (((1, 5, 7, 11), 1.0), ((2, 4, 8, 10), -1.0)):
        m = np.array([[36, 3 * s, -36, 3 * s], [3 * s, 4, -3 * s, -1], [-36, -3 * s, 36, -3 * s],
                      [3 * s, -1, -3 * s, 4]], float)
        for a in range(4):
            for b in range(4):
                pa = 1 if idx[a] in (1, 2, 7, 8) else 0
                pb = 1 if idx[b] in (1, 2, 7, 8) else 0
                lp = (1 - pa) + (1 - pb)  # number of rotation dofs -> powers of L
                val = np.where(fr, c * m[a, b] * L ** lp, 0.0)
                kg[:, idx[a], idx[b]] += val
    tr = ~fr
    t = np.where(tr, N / L, 0.0)
    for (i, j) in ((1, 7), (2, 8)):
        kg[:, i, i] += t
        kg[:, j, j] += t
        kg[:, i, j] -= t
        kg[:, j, i] -= t
    tor = np.where(fr, N * (md.Iy + md.Iz) / (md.A * L), 0.0)
    kg[:, 3, 3] += tor
    kg[:, 9, 9] += tor
    kg[:, 3, 9] -= tor
    kg[:, 9, 3] -= tor
    return kg


def _transform(md: FEModel) -> np.ndarray:
    n = len(md.L)
    T = np.zeros((n, 12, 12))
    for b in range(4):
        T[:, 3 * b:3 * b + 3, 3 * b:3 * b + 3] = md.R
    return T


def _assemble(md: FEModel, ke: np.ndarray, T: np.ndarray) -> sp.csr_matrix:
    kg = np.einsum("eji,ejk,ekl->eil", T, ke, T)
    dofs = np.concatenate([6 * md.I[:, None] + np.arange(6), 6 * md.J[:, None] + np.arange(6)], axis=1)
    rows = np.repeat(dofs, 12, axis=1).ravel()
    cols = np.tile(dofs, (1, 12)).ravel()
    nd = 6 * len(md.nodes)
    return sp.coo_matrix((kg.ravel(), (rows, cols)), shape=(nd, nd)).tocsr()


@dataclass
class FEResult:
    model: FEModel
    u: np.ndarray
    f_local: np.ndarray       # (Ne, 12) element end forces in local axes, at P_REF
    N: np.ndarray             # per element axial force (tension +)
    lam_global: float         # global buckling load factor (x P_REF)
    mode: np.ndarray | None
    f_imp: np.ndarray         # (Ne, 12) extra forces from an out-of-straight tower, at P_REF


def solve(st: Structure, wood: Wood, lateral: float, uneven: float, imp_amp: float,
          subdivide: int = 1, buckling: bool = True) -> FEResult:
    """Linear solve at P_REF, then buckling, then the effect of a built-in imperfection.

    uneven:  fraction by which the most-loaded leg exceeds its fair share (load plate not flat)
    imp_amp: mm, how far out of true the tower is built, in the shape of its buckling mode
    """
    md = build_model(st, wood, subdivide)
    ke = _k_local(md)
    T = _transform(md)
    K = _assemble(md, ke, T)
    nd = K.shape[0]
    # small rotational springs so nodes joined only by truss bars are not singular
    diag = K.diagonal()
    rot = np.zeros(nd, bool)
    rot[3::6] = rot[4::6] = rot[5::6] = True
    eps = 1e-6 * float(diag[~rot].max())
    K = K + sp.diags(np.where(rot, eps, 0.0))
    fixed = np.zeros(nd, bool)
    for s in st.support_nodes:
        fixed[6 * s:6 * s + 3] = True
    free = ~fixed
    F = np.zeros(nd)
    tops = st.top_nodes
    xs = np.array([md.nodes[t][0] for t in tops])
    r_top = float(np.mean(np.hypot(md.nodes[tops][:, 0], md.nodes[tops][:, 1])))
    ecc = uneven * r_top / 2          # gives the most-loaded leg (1 + uneven) x its share
    sxx = float((xs ** 2).sum()) or 1.0
    for t, xi in zip(tops, xs):
        F[6 * t + 2] -= P_REF / len(tops) + P_REF * ecc * xi / sxx
        F[6 * t + 0] += lateral * P_REF / len(tops)
    Kf = K[free][:, free].tocsc()
    lu = spla.splu(Kf)
    u = np.zeros(nd)
    u[free] = lu.solve(F[free])
    dofs = np.concatenate([6 * md.I[:, None] + np.arange(6), 6 * md.J[:, None] + np.arange(6)], axis=1)
    ue = u[dofs]
    f_local = np.einsum("eij,ejk,ek->ei", ke, T, ue)
    N = f_local[:, 6]
    lam, mode = np.inf, None
    f_imp = np.zeros_like(f_local)
    if buckling:
        kgl = _kg_local(md, N)
        Kg = _assemble(md, kgl, T)
        A = -Kg[free][:, free]
        try:
            Minv = spla.LinearOperator(Kf.shape, matvec=lu.solve, dtype=float)
            # fixed starting vector so the same design always gives the same answer
            v0 = np.ones(A.shape[0]) + 0.01 * np.arange(A.shape[0]) / A.shape[0]
            mu, vec = spla.eigsh(A, k=1, M=Kf, Minv=Minv, which="LA", tol=1e-6, maxiter=8000, v0=v0)
            if mu[0] > 1e-12:
                lam = 1.0 / float(mu[0])
                mode = np.zeros(nd)
                mode[free] = vec[:, 0]
        except (spla.ArpackNoConvergence, RuntimeError):
            lam = np.inf
        if mode is not None and imp_amp > 0:
            trans = np.abs(np.c_[mode[0::6], mode[1::6], mode[2::6]]).max()
            if trans > 0:
                d0 = mode * (imp_amp / trans)
                # forces the load exerts on a crooked tower = -Kg * imperfection
                F_imp = -(Kg @ d0)
                u_imp = np.zeros(nd)
                u_imp[free] = lu.solve(F_imp[free])
                f_imp = np.einsum("eij,ejk,ek->ei", ke, T, u_imp[dofs])
    return FEResult(md, u, f_local, N, lam, mode, f_imp)
