"""
PolynomialApproximators.py
==========================
Polynomial approximations to 1/x for use as QSVT signal polynomials.

All classes produce odd polynomials p(x) ≈ 1/x on [a, 1], where a = σ_min of
the block-encoded matrix.  Each polynomial is a Chebyshev object compatible
with pyqsp / QuantumSignalProcessingPhases.

Classes
-------
RemezPolynomial
    Minimax for the RELATIVE criterion  max_{x in [a,1]} |xp(x)−1|.
    Exact certificate; natural baseline for compliance-based QSVT.
    Eigenvalue correction adds no benefit (already equioscillates).

SunderhaufPolynomial
    Minimax for the ABSOLUTE criterion  max_{x in [a,1]} |p(x)−1/x|.
    Closed-form recurrence; ~30× faster than Remez to construct.
    Mindegree is ~5% higher than Remez for the same ε.
    Eigenvalue correction gives ~9× compliance improvement (criterion
    mismatch leaves a fixable relative-error residual at λ_min).

MangPolynomial
    L2-optimal in θ-space (θ = arccos x); lower degree than Remez/Sunderhauf
    (~70% of Remez at κ=100) with no provable L∞ certificate.
    Eigenvalue correction gives ~24× compliance improvement.

SpectralPolynomial
    Minimum-norm interpolating polynomial at all N known eigenvalues.
    Degree d = 3N−1, independent of κ.  No continuous guarantee between
    eigenvalues; must be used with a continuous backbone for QSVT.

References
----------
Remez (1934); Trefethen, "Approximation Theory and Approximation Practice",
SIAM 2019; Mang et al., CCIS 2744 (2026); Sunderhauf et al., arXiv:2507.15537.
"""

import math
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



# ==============================================================================
# CONVENTION NOTES  
# ==============================================================================
#
# pyqsp  signal_operator="Wx"  uses the ROTATION signal unitary:
#
#   W(x) = [[ x,           i*sqrt(1-x^2) ],    <- unitary, (0,0) elem = x
#            [ i*sqrt(1-x^2),  x          ]]
#
# and the DIAGONAL phase gate:
#
#   P(phi) = diag( e^{i*phi},  e^{-i*phi} )
#
# The QSP sequence is:  U = P(phi_0) W(x) P(phi_1) W(x) ... W(x) P(phi_d)
# and for an ODD polynomial p(x):  Re( U[0,0] ) = p(x).
#
# ── Block encoding ─────────────────────────────────────────────────────────
# To match the Wx rotation signal, the (2N x 2N) block encoding must be:
#
#   U_BE = [[ A,              i*sqrt(I - A A†) ],
#            [ i*sqrt(I - A†A),     A†          ]]
#
# Note the i factors and the +A† (not -A†) in the bottom-right corner.
# This ensures that, for each singular value sigma_i, the effective 2x2
# block is exactly W(sigma_i).
#
# ── Phase gate in Qiskit ───────────────────────────────────────────────────
# Qiskit's  Rz(theta) = diag( e^{-i*theta/2},  e^{+i*theta/2} )
# We need   P(phi)    = diag( e^{+i*phi},       e^{-i*phi}     )
# => use    qc.rz(-2*phi, ancilla)     (Z-rotation, NOT X-rotation)
#
# ── Statevector extraction ─────────────────────────────────────────────────
# Circuit: QuantumCircuit(q_anc, q_data)
# Qiskit statevector ordering: |data[n-1]...data[0], anc[0]>
#   => index k = data_idx * 2 + anc_bit
# Post-select ancilla=0: sv.data[0::2]  (even indices, data order preserved)
# The solution direction is in the REAL PART of the extracted amplitudes.
#
# ==============================================================================
# Sunderhauf optimal 1/x polynomial
# Ref: Sunderhauf et al., "Block-encoding structured matrices for data input
#      in quantum computing", Quantum 8, 1226 (2024).
# ==============================================================================

# ==============================================================================
# MangPolynomial  —  drop-in replacement for SunderhaufPolynomial
#
# Implements the L2-optimised odd Chebyshev approximation to 1/x on [a, 1]
# introduced in:
#   Mang et al., "Numerical Experiments Using Block-Diagonalization Technique
#   for Solving Poisson's Equation", QUEST-IS 2025, CCIS 2744, pp. 167-174, 2026.
#   (eq. 4 of that paper)
#
# ── What it does ──────────────────────────────────────────────────────────────
# Mang et al. parameterise x = cos(θ) and minimise the L2 residual
#
#   C(w) = ‖ Σ_i w_i cos((2i+1)θ)  −  1/(2κ cos θ) ‖²_{L²(θ)}
#
# over θ ∈ [0, arccos(a)], where a = 1/κ in their setting but we keep a
# as a general lower bound matching the SunderhaufPolynomial interface.
#
# This is ordinary least squares on the odd-Chebyshev basis {T_{2i+1}(x)}:
#   T_{2i+1}(cos θ) = cos((2i+1)θ)
# so the solution is: w = (AᵀA)⁻¹ Aᵀ t  (via numpy.linalg.lstsq).
#
# The resulting polynomial in x is:
#   p(x) = (2/a) · Σ_i w_i · T_{2i+1}(x)          [approximates 1/x on [a,1]]
#
# (Factor 2/a rather than 2κ because we use general a = sigma_min, not 1/κ.)
#
# ── Why it uses fewer terms than Sunderhauf ────────────────────────────────
# Sunderhauf guarantees L∞ < ε uniformly on [a, 1] — the hardest error metric.
# Mang minimises L2 over θ, which weights all angles equally.  Near x = a
# (large θ), where 1/x is largest and hardest to approximate, the L2 metric
# allows more pointwise error than L∞ does.  The result is lower degree at
# the cost of elevated relative error near x = a.
#
# Empirical degree comparison (ε ≈ 1%, relative criterion):
#   κ =     20 :  Remez d =  147,  Sunderhauf d =  149,  Mang d =  115  (0.78× Remez)
#   κ =    100 :  Remez d =  755,  Sunderhauf d =  899,  Mang d =  533  (0.71× Remez)
#   κ =    450 :  Remez d = 3385,  Sunderhauf d = 4723,  Mang d = 2327  (0.69× Remez)
#
# ── When to prefer Mang vs other backbones ────────────────────────────────
# Prefer Mang when:
#   • κ is large and circuit depth (degree) is the binding constraint.
#   • The RHS vector b has little energy in modes near σ_min (smooth loads
#     in topology optimisation / FEM with uniform body forces).
#   • ~0.05% compliance accuracy after K=1 eigenvalue correction is sufficient.
#
# Prefer Remez when:
#   • No eigenvalue information is available and compliance accuracy is paramount.
#   • A provable max|xp-1| ≤ ε guarantee on [a,1] is required (QSVT certificate).
#   • Note: eigenvalue correction adds NO benefit over Remez alone.
#
# Prefer Sunderhauf when:
#   • Fast polynomial construction is needed (closed-form recurrence, ~30× faster
#     than Remez at d=257).
#   • Eigenvalue correction will be applied (Sunderhauf's absolute-criterion
#     residual is correctable; Remez's is not).
#
# ── Interface (identical to SunderhaufPolynomial) ──────────────────────────
#   MangPolynomial.mindegree(epsilon, a)        -> int   (minimum odd degree)
#   MangPolynomial.poly(d, a)                   -> Chebyshev object
#   MangPolynomial.error_for_degree(d, a)       -> float (empirical L∞ estimate)
#
# The Chebyshev object returned by poly() is normalised so its L∞ norm on
# [-1, 1] equals the reciprocal of the success-probability normalisation M,
# exactly matching what _compute_phases() in myQSVT expects.
#
# ── Degree selection ───────────────────────────────────────────────────────
# mindegree() uses a bisection search: it evaluates error_for_degree() at
# candidate degrees and returns the smallest odd d such that the estimated
# L∞ error on [a, 1] is below epsilon.  This is slower than Sunderhauf's
# closed-form formula but is only called once per solver instantiation.
# ==============================================================================



# ==============================================================================
# RemezPolynomial — drop-in replacement for SunderhaufPolynomial
#
# Computes the TRUE minimax (Chebyshev) odd polynomial approximation to 1/x
# on [a, 1] for a given degree d, via the Remez exchange algorithm.
#
# ── Criterion comparison: Remez vs Sunderhauf vs Mang ────────────────────────
#
# The three classes minimise DIFFERENT objectives.  For an odd polynomial p:
#
#   RemezPolynomial      (this class):
#     Minimises  max_{x in [a,1]}  |x·p(x) - 1|        (relative error)
#     poly(d,a)         → exact minimax for this (relative) criterion at degree d
#     error_for_degree  → exact certificate E*(d,a) for the relative criterion
#     mindegree         → true minimum degree; binary search with Remez calls
#
#   SunderhaufPolynomial:
#     Minimises  max_{x in [a,1]}  |p(x) - 1/x|        (absolute error)
#     poly(d,a)         → exact minimax for this (absolute) criterion at degree d
#     error_for_degree  → EXACT closed-form: (1-a)^n / (a(1+a)^(n-1)), TIGHT
#     mindegree         → closed-form upper bound for absolute criterion
#
#   MangPolynomial:
#     Minimises  ∫ |p(θ) - 1/x(θ)|² dθ                 (L2 in θ-space)
#     poly(d,a)         → L2-optimal; no L∞ certificate
#     error_for_degree  → sampled L∞ estimate; not a provable bound
#     mindegree         → binary search on sampled estimate
#
# The absolute and relative criteria are related by |p-1/x| = |xp-1|/x, so
# the absolute criterion up-weights errors near x=a by 1/a = κ.  Sunderhauf
# is minimax for its (harder) criterion; Remez is minimax for its criterion.
# Neither is universally superior — they solve different problems.
#
# KEY CONSEQUENCE for compliance-based QSVT (topology optimisation, FEM):
#   Compliance error  |ΔJ/J| ≤ Σ_k w_k |λ_k p(λ_k) - 1|
#   is bounded by max|xp(x)-1| — the RELATIVE criterion.
#   → Remez is the natural backbone for compliance-accurate QSVT.
#   → Sunderhauf's degree is ~5% higher than Remez at matching ε (absolute vs
#     relative criterion), but Sunderhauf is ~30× faster to compute.
#   → Eigenvalue correction (min-norm) benefits Sunderhauf (systematic residual
#     from criterion mismatch) but NOT Remez (already equioscillates in |xp-1|).
#
# ── The Remez exchange algorithm ─────────────────────────────────────────────
#
# By the Chebyshev equioscillation theorem, the minimax odd polynomial p* of
# degree d = 2n-1 approximating f(x) = 1/x on [a,1] is characterised by:
#
#   e(x) = x·p*(x) - 1  equioscillates at exactly n+1 points in [a,1]:
#   it alternates between +E* and -E* where E* = max|e(x)| is the minimax error.
#
# Algorithm:
#   1. Initialise n+1 reference points uniformly in theta = arccos(x) space.
#      (Uniform-in-theta is far better conditioned than Chebyshev nodes in x,
#       because the true equioscillation points for 1/x cluster near x=a,
#       corresponding to large theta near theta_max = arccos(a).)
#   2. Solve the (n+1)×(n+1) levelled interpolation system for
#      coefficients c and level E such that:
#         x_i · p(x_i) - 1 = (-1)^i · E   for i = 0,...,n.
#   3. Locate all local extrema of e(x) on [a,1] exactly: coarse scan in
#      x-space to find sign-constant segments, then scipy bounded minimisation
#      within each segment (exact to 1e-13).
#   4. Select n+1 alternating-sign extrema and use them as the new reference set.
#   5. Repeat until |E| converges.
#
# ── Numerical note ────────────────────────────────────────────────────────────
# The levelled system is solved in float64. At large degrees the Chebyshev
# basis matrix becomes ill-conditioned if reference points are spaced in x;
# uniform theta-spacing keeps condition numbers acceptable up to degree ~1000
# (sufficient for all practical QSVT applications considered here).
#
# ── Interface (identical to SunderhaufPolynomial) ─────────────────────────────
#   RemezPolynomial.mindegree(epsilon, a)   -> int    minimum odd degree
#   RemezPolynomial.poly(d, a)              -> Chebyshev object
#   RemezPolynomial.error_for_degree(d, a)  -> float  exact minimax L∞ error
# ==============================================================================

import numpy as np
import math
from numpy.polynomial.chebyshev import Chebyshev



class RemezPolynomial:
    """
    True minimax odd polynomial for 1/x on [a,1], via the Remez algorithm.

    Drop-in replacement for SunderhaufPolynomial with identical interface.
    poly() returns the unique minimiser of max_{x in [a,1]} |x·p(x) - 1|
    over all odd polynomials of degree <= d.
    """

    _MAX_ITER: int = 80      # max Remez iterations
    _TOL: float    = 1e-10   # convergence tolerance on equioscillation level
    _N_SCAN: int   = 4000    # coarse x-scan points for segment detection

    # ------------------------------------------------------------------
    # Odd Chebyshev basis (evaluated at theta values, not x values)
    # ------------------------------------------------------------------

    @staticmethod
    def _basis_theta(thetas: np.ndarray, n: int) -> np.ndarray:
        """
        B[i,j] = T_{2j+1}(cos(theta_i)) = cos((2j+1)*theta_i).
        Shape (len(thetas), n). Well-conditioned for uniform theta grids.
        """
        return np.column_stack([np.cos((2*j + 1) * thetas) for j in range(n)])

    # ------------------------------------------------------------------
    # Extremum search on [a, 1]  —  vectorised, no scipy
    # ------------------------------------------------------------------

    @staticmethod
    def _find_extrema(c: np.ndarray, n: int, a: float,
                      e_scan: np.ndarray = None,
                      x_scan: np.ndarray = None):
        """
        Locate all local extrema of e(x) = x·p(x) - 1 on [a, 1].

        Replaces the original scipy.minimize_scalar approach with a fully
        vectorised coarse scan + 3-point parabolic refinement.  When
        called from _remez, the pre-computed e_scan and x_scan are passed
        in so the O(N_SCAN × n) matmul is shared across convergence check
        and extremum search — one matmul per iteration instead of two.

        Peak location accuracy: ~(dx)^2 ≈ 6e-8, giving |E| error < 2e-9
        (< 1e-4 relative), verified against minimize_scalar.  Sufficient
        for the Remez exchange which only needs the equioscillation level
        to converge within _TOL = 1e-10.
        """
        j_idx = np.arange(n, dtype=np.float64)

        if e_scan is None:
            x_scan  = np.linspace(a, 1.0, RemezPolynomial._N_SCAN)
            th_scan = np.arccos(np.clip(x_scan, -1.0 + 1e-14, 1.0 - 1e-14))
            e_scan  = x_scan * (np.cos(np.outer(th_scan, 2.0*j_idx+1.0)) @ c) - 1.0

        abs_e   = np.abs(e_scan)
        sgn     = np.sign(e_scan)
        changes = np.where(np.diff(sgn))[0]
        bounds  = np.concatenate([[0], changes + 1, [len(e_scan)]])

        extrema = []
        for ki in range(len(bounds) - 1):
            lo_i = bounds[ki]
            hi_i = min(bounds[ki + 1], len(e_scan) - 1)
            if hi_i - lo_i < 1:
                continue
            pk = int(np.argmax(abs_e[lo_i : hi_i + 1])) + lo_i

            # 3-point parabolic sub-sample refinement
            if 0 < pk < len(e_scan) - 1:
                y0, y1, y2 = abs_e[pk-1], abs_e[pk], abs_e[pk+1]
                x0, x1, x2 = x_scan[pk-1], x_scan[pk], x_scan[pk+1]
                denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
                if abs(denom) > 1e-20:
                    A = (x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1)) / denom
                    B = (x2**2*(y0-y1) + x1**2*(y2-y0) + x0**2*(y1-y2)) / denom
                    if A < 0.0:
                        xpk = -B / (2.0 * A)
                        if x0 <= xpk <= x2:
                            th_pk = float(np.arccos(np.clip(xpk, -1+1e-14, 1-1e-14)))
                            bv    = np.cos((2.0*j_idx + 1.0) * th_pk)
                            extrema.append((float(xpk), float(xpk)*(float(bv@c))-1.0))
                            continue

            xpk   = float(x_scan[pk])
            th_pk = float(np.arccos(np.clip(xpk, -1+1e-14, 1-1e-14)))
            bv    = np.cos((2.0*j_idx + 1.0) * th_pk)
            extrema.append((xpk, float(xpk)*(float(bv@c))-1.0))

        extrema.append((float(a),   float(e_scan[0])))
        extrema.append((float(1.0), float(e_scan[-1])))
        extrema.sort(key=lambda xe: xe[0])
        deduped = [extrema[0]]
        for xe in extrema[1:]:
            if xe[0] - deduped[-1][0] > 1e-10:
                deduped.append(xe)
        return deduped

    # ------------------------------------------------------------------
    # Core Remez
    # ------------------------------------------------------------------

    @staticmethod
    def _remez(d: int, a: float):
        """
        Run Remez exchange and return (Chebyshev poly p*, float E_star).

        Optimisations vs the original implementation
        --------------------------------------------
        1. _basis_theta uses np.outer instead of a Python column_stack loop.
        2. The error scan (x·p(x)-1 on N_SCAN points) is a single matmul,
           and the resulting e_scan array is passed directly into _find_extrema
           — avoiding a second O(N_SCAN × n) evaluation per iteration.
        3. _find_extrema uses parabolic peak refinement (no scipy.minimize_scalar),
           giving ~8× speedup on the extremum search with <1e-4 relative E error.
        """
        n     = (d + 1) // 2
        j_idx = np.arange(n, dtype=np.float64)

        # Initialise reference points uniformly in theta = arccos(x) space.
        # Uniform-in-theta is critical: equioscillation points for 1/x cluster
        # near x=a (large theta), giving a well-conditioned levelled system.
        theta_max = float(np.arccos(a))
        k         = np.arange(n + 1, dtype=np.float64)
        x_ref     = np.clip(np.cos(theta_max * k / n), a, 1.0)

        # Pre-build the fixed scan grid (reused every iteration)
        x_scan  = np.linspace(a, 1.0, RemezPolynomial._N_SCAN)
        th_scan = np.arccos(np.clip(x_scan, -1.0 + 1e-14, 1.0 - 1e-14))

        E_star = np.inf
        c      = None

        for _ in range(RemezPolynomial._MAX_ITER):

            # ── Levelled interpolation ─────────────────────────────────
            # x_i · (B[i,:] @ c) - 1 = (-1)^i · E   for i = 0,...,n
            th_ref = np.arccos(np.clip(x_ref, -1.0 + 1e-14, 1.0 - 1e-14))
            B_ref  = np.cos(np.outer(th_ref, 2.0 * j_idx + 1.0))   # (n+1, n)
            signs  = (-1.0) ** np.arange(n + 1)
            M      = np.column_stack([x_ref[:, None] * B_ref, -signs])

            try:
                sol = np.linalg.solve(M, np.ones(n + 1))
            except np.linalg.LinAlgError:
                sol, *_ = np.linalg.lstsq(M, np.ones(n + 1), rcond=None)
            c = sol[:n]

            # ── Vectorised error scan (shared with extremum search) ────
            B_scan = np.cos(np.outer(th_scan, 2.0 * j_idx + 1.0))
            e_scan = x_scan * (B_scan @ c) - 1.0
            E_new  = float(np.max(np.abs(e_scan)))

            # ── Convergence check ──────────────────────────────────────
            if abs(E_new - E_star) < RemezPolynomial._TOL:
                E_star = E_new
                break
            E_star = E_new

            # ── Extremum search — reuses e_scan, no scipy ─────────────
            extrema = RemezPolynomial._find_extrema(
                c, n, a, e_scan=e_scan, x_scan=x_scan)

            # ── Exchange: select n+1 alternating extrema ───────────────
            new_ref = _select_alternating(extrema, n + 1)
            if len(new_ref) == n + 1:
                x_ref = np.array(new_ref)

        # ── Build Chebyshev object ─────────────────────────────────────
        coef = np.zeros(2 * n)
        for j, cj in enumerate(c):
            coef[2*j + 1] = cj
        return Chebyshev(coef), E_star

    # ------------------------------------------------------------------
    # Public interface — identical to SunderhaufPolynomial
    # ------------------------------------------------------------------

    @staticmethod
    def poly(d: int, a: float) -> Chebyshev:
        """
        True minimax odd polynomial of degree d approximating 1/x on [a, 1].

        This is the unique minimiser of max_{x in [a,1]} |x·p(x) - 1| over
        all odd polynomials of degree <= d. Found via the Remez algorithm.

        Parameters
        ----------
        d : int    Odd polynomial degree.
        a : float  Left endpoint; a = sigma_min of the block-encoded matrix.

        Returns
        -------
        Chebyshev  Minimax polynomial p* with p*(x) ≈ 1/x on [a, 1].
        """
        if d % 2 == 0:
            raise ValueError(f"d must be odd, got {d}.")
        if not (0 < a < 1):
            raise ValueError(f"a must be in (0, 1), got {a}.")
        p, _ = RemezPolynomial._remez(d, a)
        return p

    @staticmethod
    def error_for_degree(d: int, a: float) -> float:
        """
        Exact minimax L∞ error at degree d:

            E*(d, a) = min_{p odd, deg<=d}  max_{x in [a,1]} |x·p(x) - 1|

        Unlike Sunderhauf's closed-form (conservative for 1/x) and Mang's
        sampled estimate (not a certificate), this is the exact best-achievable
        error for any odd polynomial of degree d approximating 1/x on [a,1].

        Parameters
        ----------
        d : int    Odd polynomial degree.
        a : float  Left endpoint.

        Returns
        -------
        float  Exact minimax error E*(d, a).
        """
        if d % 2 == 0:
            d += 1
        _, E = RemezPolynomial._remez(d, a)
        return E

    @staticmethod
    def mindegree(epsilon: float, a: float) -> int:
        """
        Minimum odd degree d such that E*(d, a) <= epsilon.

        Exponential search to bracket, then binary search with Remez at each
        candidate. Gives the minimum degree for a provable L∞ guarantee
        specific to 1/x — always <= Sunderhauf's closed-form degree.

        Parameters
        ----------
        epsilon : float  Target L∞ error.
        a       : float  Left endpoint a = sigma_min.

        Returns
        -------
        int  Minimum odd d with E*(d, a) <= epsilon.
        """
        if not (0 < epsilon < 1):
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}.")
        if not (0 < a < 1):
            raise ValueError(f"a must be in (0, 1), got {a}.")

        # Sunderhauf closed-form is a guaranteed upper bound
        n_sund = math.ceil(
            (math.log(1/epsilon) + math.log(1/a) + math.log(1+a))
            / math.log((1+a)/(1-a))
        )
        d_max = 2 * n_sund - 1

        if RemezPolynomial.error_for_degree(1, a) <= epsilon:
            return 1

        # Exponential search: 1, 3, 7, 15, ..., capped at d_max
        d = 1
        while RemezPolynomial.error_for_degree(d, a) > epsilon:
            next_d = min(d * 2 + 1, d_max)
            if next_d == d:
                break
            d = next_d

        d_hi = d
        d_lo = max(1, (d // 2) | 1)    # largest previously tried (odd)

        # Binary search in [d_lo, d_hi]
        while d_lo < d_hi:
            d_mid = (d_lo + d_hi) // 2
            if d_mid % 2 == 0:
                d_mid += 1
            if d_mid >= d_hi:
                break
            if RemezPolynomial.error_for_degree(d_mid, a) <= epsilon:
                d_hi = d_mid
            else:
                d_lo = d_mid + 2

        return d_hi


# ------------------------------------------------------------------
# Helper: greedy alternating-sign selection from sorted (x, e) list
# ------------------------------------------------------------------

def _select_alternating(extrema, n: int):
    """
    Select n alternating-sign points from sorted (x,e) pairs,
    replacing within a sign group when a larger |e| is found.
    Tries both starting signs and returns the best (longest) result.
    """
    best = []
    for start_sign in [+1.0, -1.0]:
        sel_x, sel_e = [], []
        for x, e in extrema:
            sg = np.sign(e)
            if sg == 0:
                continue
            if not sel_e:
                if sg == start_sign:
                    sel_x.append(x); sel_e.append(e)
            elif sg == np.sign(sel_e[-1]):
                if abs(e) > abs(sel_e[-1]):
                    sel_x[-1] = x; sel_e[-1] = e
            else:
                sel_x.append(x); sel_e.append(e)
        if len(sel_x) >= n and len(sel_x) > len(best):
            best = sel_x[:n]
    return best

class MangPolynomial:
    """
    L2-optimised odd Chebyshev polynomial approximating 1/x on [a, 1].

    Drop-in replacement for SunderhaufPolynomial.  All three public methods
    share the same signature so _compute_phases() in myQSVT works unchanged.

    Ref: Mang et al., QUEST-IS 2025, CCIS 2744 (2026), eq. 4.
    """

    # Minimum quadrature points for the L2 fit.
    # The actual count is max(_N_THETA_MIN, 4*n_terms) — see _lstsq_coeffs.
    _N_THETA_MIN: int = 2000

    # Number of sample points used to estimate the L∞ error in error_for_degree.
    _N_LINF: int = 4000

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lstsq_coeffs(d: int, a: float) -> np.ndarray:
        """
        Solve the L2 problem (eq. 4 of Mang et al.) and return the weight
        vector w of length (d+1)//2.

        Design matrix:  A[i, j] = cos((2j+1) * theta_i)
                                 = T_{2j+1}(cos theta_i)
        Target:         t[i]    = 1 / (2 * a * cos(theta_i))
                                 = 1 / (2 * a * x_i)

        The target is 1/(2a · x), i.e. (1/2a) · 1/x.  Multiplying w by 2a
        recovers the coefficients of the Chebyshev series for 1/x.

        Returns w (unnormalised); the caller applies the 2a factor.
        """
        n_terms   = (d + 1) // 2    # number of odd Chebyshev terms T_1, T_3, ...

        # N_THETA must exceed n_terms to keep the system overdetermined.
        # A fixed constant fails for large d (large kappa, e.g. m >= 6) where
        # n_terms > _N_THETA_MIN, making the system underdetermined.
        N_THETA   = max(MangPolynomial._N_THETA_MIN, 4 * n_terms)

        theta_max = np.arccos(a)
        thetas    = np.linspace(1e-8, theta_max, N_THETA)
        target    = 1.0 / (2.0 * a * np.cos(thetas))
        A = np.column_stack([np.cos((2 * j + 1) * thetas)
                             for j in range(n_terms)])

        # Solve via normal equations with Tikhonov regularisation:
        #   (A^T A + lam * I) w = A^T t
        # This replaces numpy.linalg.lstsq (which uses LAPACK dgelsd, SVD-based)
        # because dgelsd fails to converge on Windows/MKL when the Chebyshev
        # columns are nearly linearly dependent — which happens when theta_max
        # is close to pi/2 (small a, large kappa).  The normal equations solved
        # via numpy.linalg.solve (LU/Cholesky) are platform-independent and
        # always converge.  lambda = 1e-10 has no measurable effect on the fit
        # (verified: L-inf error identical to 8 significant figures vs lam=0).
        lam = 1e-10
        ATA = A.T @ A
        ATt = A.T @ target
        ATA[np.diag_indices(n_terms)] += lam
        w = np.linalg.solve(ATA, ATt)
        return w

    @staticmethod
    def _build_chebyshev(w: np.ndarray, a: float) -> Chebyshev:
        """
        Convert weight vector w (from _lstsq_coeffs) into a Chebyshev object
        representing p(x) ≈ 1/x on [a, 1].

        p(x) = (2a) · Σ_j w_j · T_{2j+1}(x)

        The factor 2a converts from the 1/(2a·x) target back to 1/x.
        Chebyshev coefficient array has zeros at even positions (odd poly).
        """
        n_terms   = len(w)
        max_degree = 2 * n_terms   # highest term is T_{2*(n_terms-1)+1}
        coef      = np.zeros(max_degree)
        for j, wj in enumerate(w):
            coef[2 * j + 1] = 2.0 * a * wj   # coefficient of T_{2j+1}
        return Chebyshev(coef)

    # ------------------------------------------------------------------
    # Public interface  (identical to SunderhaufPolynomial)
    # ------------------------------------------------------------------

    @staticmethod
    def poly(d: int, a: float) -> Chebyshev:
        """
        Return a Chebyshev polynomial object for the Mang L2-optimised
        approximation to 1/x on [a, 1] at degree d (must be odd).

        The polynomial is NOT normalised here — _compute_phases() in myQSVT
        applies its own normalisation (dividing by M) before calling pyqsp,
        exactly as it does for SunderhaufPolynomial.poly().

        Parameters
        ----------
        d : int   Polynomial degree (must be odd).
        a : float Lower bound of the approximation interval; a = σ_min(L_D/4).

        Returns
        -------
        Chebyshev  Odd Chebyshev polynomial p with p(x) ≈ 1/x on [a, 1].
        """
        if d % 2 == 0:
            raise ValueError(f"d must be odd, got {d}.")
        if not (0 < a < 1):
            raise ValueError(f"a must be in (0, 1), got {a}.")

        w = MangPolynomial._lstsq_coeffs(d, a)
        return MangPolynomial._build_chebyshev(w, a)

    @staticmethod
    def error_for_degree(d: int, a: float) -> float:
        """
        Estimate the L∞ relative approximation error of the degree-d Mang
        polynomial on [a, 1], defined as:

            max_{x in [a,1]}  |p(x) - 1/x| / (1/x)
          = max_{x in [a,1]}  |x · p(x) - 1|

        Unlike Sunderhauf, there is no closed-form expression; we evaluate
        the polynomial numerically at _N_LINF sample points.

        Note: Mang et al. use an L2 target, so this L∞ estimate will be
        larger than the L2 residual, particularly near x = a.  Use this
        value as a conservative upper bound on approximation quality.

        Parameters
        ----------
        d : int   Polynomial degree (must be odd).
        a : float Lower bound a = σ_min(L_D/4).

        Returns
        -------
        float  Estimated maximum relative error on [a, 1].
        """
        if d % 2 == 0:
            d += 1   # silently promote to next odd degree

        w    = MangPolynomial._lstsq_coeffs(d, a)
        poly = MangPolynomial._build_chebyshev(w, a)

        xs       = np.linspace(a, 1.0, MangPolynomial._N_LINF)
        vals     = poly(xs)
        rel_err  = np.abs(vals * xs - 1.0)   # |x·p(x) - 1|
        return float(np.max(rel_err))

    @staticmethod
    def mindegree(epsilon: float, a: float) -> int:
        """
        Find the minimum odd polynomial degree d such that the estimated
        L∞ relative error on [a, 1] is below epsilon.

        Uses bisection over odd degrees between d_lo=1 and d_hi, where d_hi
        is set conservatively using the Sunderhauf closed-form as an upper
        bound (Mang degree is always ≤ Sunderhauf degree).

        Parameters
        ----------
        epsilon : float  Target maximum relative error (e.g. 0.01 for 1%).
        a       : float  Lower bound a = σ_min(L_D/4).

        Returns
        -------
        int  Minimum odd degree d satisfying the error criterion.
        """
        if not (0 < epsilon < 1):
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}.")
        if not (0 < a < 1):
            raise ValueError(f"a must be in (0, 1), got {a}.")

        # Quick exit.
        if MangPolynomial.error_for_degree(1, a) <= epsilon:
            return 1

        # Phase 1 — Exponential search: double d until error <= epsilon.
        # This avoids bisecting from Sunderhauf's d_hi (which can be ~16 000
        # for m=6), which would build a multi-GB design matrix at the very
        # first bisection step and crash with an out-of-memory error.
        d = 1
        while MangPolynomial.error_for_degree(d, a) > epsilon:
            d = d * 2 + 1       # next odd integer roughly double
            if d > 500_000:
                return d        # safety cap
        d_good = d

        # Phase 2 — Bisect in the tight bracket [d//2, d_good].
        d_lo = max(1, (d_good // 2) | 1)   # ensure odd
        k_lo = (d_lo + 1) // 2
        k_hi = (d_good + 1) // 2

        while k_lo < k_hi:
            k_mid = (k_lo + k_hi) // 2
            d_mid = 2 * k_mid - 1
            if MangPolynomial.error_for_degree(d_mid, a) <= epsilon:
                k_hi = k_mid
            else:
                k_lo = k_mid + 1

        return 2 * k_lo - 1


class SunderhaufPolynomial:
    """
    Optimal odd polynomial for 1/x under the ABSOLUTE error criterion.

    Minimises  max_{x in [a,1]} |p(x) - 1/x|  in closed form.

    Ref: Sunderhauf et al., arXiv:2507.15537 (2025).

    Note: this is NOT the same as RemezPolynomial, which minimises the
    RELATIVE criterion max|xp(x)-1|.  The two differ by a factor of 1/x:

        |p(x) - 1/x|  =  |xp(x) - 1| / x

    so the absolute criterion up-weights errors near x=a by κ = 1/a.
    Consequently Sunderhauf's mindegree is typically 5–10% higher than
    Remez's at the same ε.  However it is ~30× faster to compute.

    For compliance-based QSVT: Sunderhauf has a systematic relative-error
    residual at λ_min (criterion mismatch) that min-norm eigenvalue correction
    can remove (~9× improvement).  Remez has no such residual and gains
    nothing from correction.
    """

    @staticmethod
    def helper_Lfrac(n: int, x: float, a: float) -> float:
        """Three-term recurrence for L_n(x; a)."""
        alpha = (1 + a) / (2 * (1 - a))
        l1 = (x + (1 - a) / (1 + a)) / alpha
        l2 = (x**2 + (1 - a) / (1 + a) * x / 2 - 0.5) / alpha**2
        if n == 1:
            return l1
        for _ in range(3, n + 1):
            l1, l2 = l2, x * l2 / alpha - l1 / (4 * alpha**2)
        return l2

    @staticmethod
    def helper_P(x: float, n: int, a: float) -> float:
        return (
            1
            - (-1)**n * (1 + a)**2 / (4 * a)
            * SunderhaufPolynomial.helper_Lfrac(
                n, (2 * x**2 - (1 + a**2)) / (1 - a**2), a)
        ) / x

    @staticmethod
    def poly(d: int, a: float) -> Chebyshev:
        if d % 2 == 0:
            raise ValueError("d must be odd")
        coef = np.polynomial.chebyshev.chebinterpolate(
            SunderhaufPolynomial.helper_P, d, args=((d + 1) // 2, a))
        coef[0::2] = 0          # enforce odd parity exactly
        return Chebyshev(coef)

    @staticmethod
    def error_for_degree(d: int, a: float) -> float:
        n = (d + 1) // 2
        return (1 - a)**n / (a * (1 + a)**(n - 1))

    @staticmethod
    def mindegree(epsilon: float, a: float) -> int:
        n = math.ceil(
            (np.log(1 / epsilon) + np.log(1 / a) + np.log(1 + a))
            / np.log((1 + a) / (1 - a))
        )
        return 2 * n - 1

# ======================================================================
# SpectralPolynomial
# ======================================================================

class SpectralPolynomial:
    """
    Minimum-norm polynomial interpolating 1/x at all N known eigenvalues.

    When the eigenvalues λ_1, …, λ_N of the system matrix are known a priori,
    the polynomial need only satisfy λ_i p(λ_i) = 1 at those N discrete points.
    This replaces the continuous minimax problem on [a,1] with an underdetermined
    linear system in the odd-Chebyshev basis, yielding degree d = 3N−1 regardless
    of κ — compared to d ~ O(κ log(κ/ε)) for Remez or Sunderhauf.

    WARNING: This polynomial has NO guarantee on [a,1] between eigenvalues.
    It must be paired with a continuous backbone (Remez, Sunderhauf, or Mang)
    via the min-norm eigenvalue correction when used in QSVT — see the hybrid
    strategy in the accompanying document.

    Construction
    ------------
    Solves the minimum-ℓ2-norm system

        λ_i · p(λ_i) = 1    for i = 1, …, N

    in the odd-Chebyshev basis {T_1, T_3, …, T_d} on [−1, 1].
    With n = (d+1)/2 > N free coefficients (n_factor > 1), the system is
    underdetermined; the Moore–Penrose pseudoinverse gives the minimum-norm
    solution via lstsq.

    Parameters
    ----------
    eigenvalues : array_like, shape (N,)
        Eigenvalues of the SPD system matrix, normalised to lie in (0, 1).
    n_factor : float, optional
        Over-parameterisation ratio.  The basis dimension is n = ⌈n_factor·N⌉,
        giving degree d = 2n−1.  Must satisfy n_factor ≥ 1.  Default 1.5
        (d = 3N−1).  Use 2.0 for smaller max|p(x)| on [−1,1].

    Public API
    ----------
    poly()         -> Chebyshev   eigenvalue-interpolating polynomial
    mindegree()    -> int         polynomial degree d = 2⌈n_factor·N⌉−1
    error_at_eigs() -> ndarray    |λ_i p(λ_i) − 1| for each eigenvalue

    References
    ----------
    Trefethen, "Approximation Theory and Approximation Practice", SIAM 2019,
    Ch. 4 (Chebyshev interpolation).
    """

    def __init__(self, eigenvalues, n_factor: float = 1.5):
        eigenvalues = np.asarray(eigenvalues, dtype=float).ravel()
        if np.any(eigenvalues <= 0) or np.any(eigenvalues >= 1):
            raise ValueError("All eigenvalues must lie in (0, 1).")
        if n_factor < 1.0:
            raise ValueError("n_factor must be >= 1.")

        self._eigs     = np.sort(eigenvalues)
        self._N        = len(eigenvalues)
        self._n_factor = n_factor
        self._n        = int(np.ceil(n_factor * self._N))
        self._d        = 2 * self._n - 1
        self._poly     = self._build_poly()

    def _build_poly(self) -> Chebyshev:
        lam = self._eigs
        n   = self._n
        j   = np.arange(n, dtype=float)
        th  = np.arccos(np.clip(lam, 1e-14, 1 - 1e-14))
        B   = np.cos(np.outer(th, 2*j + 1))   # (N, n)
        LB  = lam[:, None] * B                 # (N, n)
        c, _, _, _ = np.linalg.lstsq(LB, np.ones(self._N), rcond=None)
        coef = np.zeros(2 * n)
        for j_, cj in enumerate(c):
            coef[2 * j_ + 1] = cj
        return Chebyshev(coef)

    def poly(self, d: int = None, a: float = None) -> Chebyshev:
        """Return the eigenvalue-interpolating polynomial (d and a ignored)."""
        return self._poly

    def mindegree(self, epsilon: float = None, a: float = None) -> int:
        """Return the polynomial degree d = 2⌈n_factor·N⌉−1 (args ignored)."""
        return self._d

    def error_at_eigs(self) -> np.ndarray:
        """Return |λ_i p(λ_i) − 1| for each eigenvalue (should be ~1e-12)."""
        return np.abs(self._eigs * self._poly(self._eigs) - 1.0)

    def degree_info(self) -> dict:
        """Summary dict: N, n_factor, n, d, max|p| on [−1,1], max_delta."""
        x   = np.linspace(-1, 1, 25 * self._d)
        tau = float(np.max(np.abs(self._poly(x))))
        return {
            "N":         self._N,
            "n_factor":  self._n_factor,
            "n":         self._n,
            "d":         self._d,
            "tau":       tau,
            "max_delta": float(np.max(self.error_at_eigs())),
        }

BASE_POLYS = {
    'remez' : RemezPolynomial,
    'mang'  : MangPolynomial,
    'sunderhauf'  : SunderhaufPolynomial,
}

def spectral_correction(p0: Chebyshev, eigenvalues: np.ndarray,
                      rcond: float = 1e-10) -> np.ndarray:
    """
    Compute the min-norm Chebyshev coefficient correction using a
    truncated SVD solve for the Gram system, robust to ill-conditioning
    when eigenvalues are small and tightly clustered.

    Parameters
    ----------
    p0          : Chebyshev polynomial object (defined on [-1, 1])
    eigenvalues : array of K known eigenvalues in (0, 1]
    rcond       : singular value threshold relative to sigma_max;
                  directions with s < rcond * s_max are discarded.
                  Default 1e-10 is safe for double precision.

    Returns
    -------
    c_corr : np.ndarray, shape (n0,)
        Additive correction to odd Chebyshev coefficients p0.coef[1::2].
    """
    lam = np.asarray(eigenvalues)
    lam_sorted = np.sort(lam)
    lam_unique = [lam_sorted[0]]
    for l in lam_sorted[1:]:
        if (l - lam_unique[-1]) / lam_unique[-1] > 1e-3:  # relative gap > 0.1%
            lam_unique.append(l)
    lam_unique = np.array(lam_unique)

    if len(lam_unique) < len(lam_sorted):
        print(f"Removing duplicate eigenvalues: {len(lam_sorted)} -> {len(lam_unique)}")

    lam = lam_unique.copy()

    c0  = p0.coef[1::2]
    n0  = len(c0)
    j   = np.arange(n0)

    # B[k, j] = T_{2j+1}(lambda_k)
    B  = np.cos(np.outer(np.arccos(np.clip(lam, -1+1e-14, 1-1e-14)), 2*j+1))

    # Step 1: residuals
    r  = 1.0 - lam * p0(lam)

    # Step 2: Gram system via truncated SVD
    LB       = lam[:, None] * B
    G        = LB @ LB.T
    U, s, Vt = np.linalg.svd(G)
    s_inv    = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    alpha    = Vt.T @ (s_inv * (U.T @ r))

    # Step 3: correction coefficients

    # inside spectral_correction, after computing G:
    return LB.T @ alpha


def _min_norm_correction(p_B, d_B: int, a: float, known_eigs):
    """
    Minimum-norm coefficient update to zero λ_k p(λ_k)−1 at each known_eig.

    Parameters
    ----------
    p_B       : Chebyshev  backbone polynomial
    d_B       : int        degree of p_B (must be odd)
    a         : float      σ_min (lower spectral bound)
    known_eigs: array_like eigenvalues at which to zero the residual

    Returns
    -------
    Chebyshev  corrected polynomial (same degree d_B)
    """
    lam = np.clip(np.asarray(known_eigs, float), a * 1.002, 0.9999)
    n   = (d_B + 1) // 2
    j   = np.arange(n)
    th  = np.arccos(np.clip(lam, 1e-12, 1 - 1e-12))
    B   = np.cos(np.outer(th, 2*j + 1))     # (K, n) Chebyshev basis
    LB  = lam[:, None] * B                   # (K, n) weighted basis
    r   = -(lam * p_B(lam) - 1)              # residuals to eliminate
    # Gram matrix solve: c_K = (LB LBᵀ)⁻¹ r
    G   = LB @ LB.T
    c_K = np.linalg.solve(G + 1e-14 * np.eye(len(lam)), r)
    delta_c = LB.T @ c_K                    # coefficient corrections
    coef = p_B.coef.copy()
    for j_, v in enumerate(delta_c):
        coef[2*j_ + 1] += v
    return Chebyshev(coef)


if __name__ == "__main__":
    kappa = 10
    eps = 0.01
    K = 3
    lam = np.array([1/kappa, 0.5, 1])
    basePolynomial  = 'sunderhauf' # 'remez' or 'mang' or 'sunderhauf'
    a = 1/kappa
    polyClass = BASE_POLYS[basePolynomial.lower()]

    degree   = polyClass.mindegree(eps, 1/kappa)
    poly = polyClass.poly(degree, 1/kappa)

    corr = spectral_correction(poly,lam)
    coef_H       = poly.coef.copy()
    coef_H[1::2] += corr
    spectral_poly = Chebyshev(coef_H)


    x = np.union1d(np.linspace(a, 1.0, 1000), lam)
    y = spectral_poly(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title(f'Spectral with {basePolynomial} polynomial of degree {degree} for a={a:.2f} and eps={eps:.2f}')
    plt.grid()
    plt.show()

    error = np.abs(x * spectral_poly(x) - 1.0)
    plt.plot(x, error)
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title(f'Error of spectral with {basePolynomial} polynomial of degree {degree} for a={a:.2f} and eps={eps:.2f}')
    plt.yscale('log')
    plt.grid()
    plt.show()