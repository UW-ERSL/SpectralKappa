"""
MinimaxCorrection.py
====================
Minimax tau-controlled spectral correction for QSVT.

Given a base polynomial p0 (from PolynomialApproximators.py) and K known
eigenvalues normalized to (0,1], this module finds a correction Delta_c that:

  1. Enforces  lambda_k * p_SC(lambda_k) = 1  for k = 1, ..., K
  2. Minimizes tau = max_{x in [-1,1]} |p_SC(x)|

subject to the parity constraint (p_SC remains odd).

The corrected polynomial p_SC = p0 + p_corr has the SAME degree as p0.
This is the key difference from spectral_correction() in PolynomialApproximators.py
(Gram system solve), which does not control tau explicitly.

Background
----------
- SpectralCorrection.tex  : base spectral correction (Gram system)
- Chapter18.tex (QSVT)    : polynomial degree d = O(sqrt(kappa) log(1/eps))
- SpectralKappa project   : exploit K known eigenvalues to reduce kappa_eff
                            while bounding tau via minimax LP

Formulation
-----------
Variables : Delta_c in R^{n0},  t in R  (t is the tau upper bound)
Minimize  : t
Subject to:
  p0(x_i) + B(x_i) @ Delta_c <=  t   for all Chebyshev grid points x_i
  p0(x_i) + B(x_i) @ Delta_c >= -t   for all Chebyshev grid points x_i
  lambda_k * (p0(lambda_k) + B(lambda_k) @ Delta_c) = 1,  k = 1,...,K

where B[i,j] = T_{2j+1}(x_i) is the odd Chebyshev basis matrix.

The grid is a Chebyshev grid cos(j*pi/(n-1)) on [-1,1], which captures
polynomial extrema exactly.

Parity is preserved by construction: p0 is odd (per QSVT convention),
p_corr is in the same odd basis, so p_SC is odd.

Interface conventions (matching PolynomialApproximators.py)
-----------------------------------------------------------
- p0 is a numpy.polynomial.chebyshev.Chebyshev object
- Odd Chebyshev coefficients are at p0.coef[1::2]
- Eigenvalues are normalized to (0, 1]
  (for 1D Poisson: lams = eigs_1d_poisson(m) / eigs_1d_poisson(m).max())
- Returned p_SC is also a Chebyshev object, usable directly in QSVTSolvers.py

Dependencies
------------
  numpy, cvxpy, numpy.polynomial.chebyshev

Usage
-----
  See SpectralKappaExperiments.ipynb for worked examples on 1D/2D Poisson.
"""

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
import cvxpy as cp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _odd_cheb_basis(x, n0):
    """
    Evaluate the odd Chebyshev basis at points x.

    Returns B of shape (len(x), n0) where B[i, j] = T_{2j+1}(x[i]).
    The odd polynomial p(x) = sum_j c_j T_{2j+1}(x) = B @ c,
    where c = p0.coef[1::2] in the PolynomialApproximators convention.

    Parameters
    ----------
    x  : array-like, shape (m,)
    n0 : int   number of odd Chebyshev terms; degree d = 2*n0 - 1

    Returns
    -------
    B  : ndarray, shape (m, n0)
    """
    x = np.asarray(x, dtype=float)
    B = np.zeros((len(x), n0))
    for j in range(n0):
        coeffs = np.zeros(2 * j + 2)
        coeffs[2 * j + 1] = 1.0
        B[:, j] = Chebyshev(coeffs)(x)
    return B


def effective_kappa(all_eigenvalues, K):
    """
    Compute the effective condition number after correcting the K smallest
    eigenvalues.

    After correction the base polynomial only needs to approximate 1/x on
    [lambda_{K+1}, lambda_max].  The effective condition number is:

        kappa_eff = lambda_max / lambda_{K+1}

    and the predicted degree reduction factor from the sqrt(kappa) term is:

        factor = sqrt(kappa / kappa_eff)

    Parameters
    ----------
    all_eigenvalues : array-like   full eigenvalue spectrum, normalized to (0,1]
    K               : int          number of corrected eigenvalues (smallest K)

    Returns
    -------
    dict with keys:
        kappa                   original condition number lambda_max / lambda_min
        kappa_eff               effective condition number lambda_max / lambda_{K+1}
        degree_reduction_factor sqrt(kappa / kappa_eff)
        lambda_min              lambda_1
        lambda_K1               lambda_{K+1} (new effective left endpoint)
    """
    lams  = np.sort(np.asarray(all_eigenvalues, dtype=float))
    kappa = lams[-1] / lams[0]

    if K >= len(lams):
        lam_K1    = lams[-1]
        kappa_eff = 1.0
    else:
        lam_K1    = lams[K]          # 0-indexed: lams[K] is lambda_{K+1}
        kappa_eff = lams[-1] / lam_K1

    return {
        'kappa'                  : kappa,
        'kappa_eff'              : kappa_eff,
        'degree_reduction_factor': np.sqrt(kappa / kappa_eff),
        'lambda_min'             : lams[0],
        'lambda_K1'              : lam_K1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class MinimaxCorrection:
    """
    Minimax tau-controlled spectral correction for QSVT.

    Parameters
    ----------
    p0 : Chebyshev
        Base polynomial from PolynomialApproximators.py.
        Obtain via e.g. MangPolynomial.poly(d, a) where a = lambda_{K+1}.
        Must be an odd polynomial; odd coefficients at p0.coef[1::2].

    eigenvalues : array-like, shape (K,)
        K known eigenvalues normalized to (0, 1], sorted ascending.
        For 1D Poisson:
            lams_raw  = eigs_1d_poisson(m)
            lams_norm = lams_raw / lams_raw.max()
            eigenvalues = lams_norm[:K]

    n_grid : int, optional (default 2000)
        Number of Chebyshev grid points for LP discretization.
        Rule of thumb: n_grid >= 10 * degree.

    Notes
    -----
    The interpolation constraints lambda_k * p_SC(lambda_k) = 1 pin the
    polynomial to 1/lambda_k ~ kappa at the smallest eigenvalues, so
    tau >= kappa is a hard lower bound.  The LP minimizes tau above this
    floor by suppressing oscillations in [lambda_1, lambda_K].

    The critical question of SpectralKappa: does the LP keep tau near
    kappa, or does it blow up?  Run tau_vs_K_sweep() to answer this
    numerically for your specific problem.
    """

    def __init__(self, p0: Chebyshev, eigenvalues, n_grid=2000):
        self.p0          = p0
        self.c0          = p0.coef[1::2].copy()   # odd Chebyshev coefficients
        self.n0          = len(self.c0)
        self.d0          = 2 * self.n0 - 1
        self.eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
        self.K           = len(self.eigenvalues)
        self.n_grid      = n_grid

        # Chebyshev grid on [-1, 1]
        self.grid = np.cos(np.linspace(0, np.pi, n_grid))

        # Basis matrices — computed once, reused in solve()
        self.B_grid = _odd_cheb_basis(self.grid,        self.n0)
        self.B_eig  = _odd_cheb_basis(self.eigenvalues, self.n0)

        # Base polynomial on grid and at eigenvalues
        self.p0_grid = self.B_grid @ self.c0
        self.p0_eig  = self.B_eig  @ self.c0

    # ── Main solver ──────────────────────────────────────────────────────────

    def solve(self, tikhonov=0.0, verbose=False):
        """
        Solve the minimax LP for the tau-optimal spectral correction.

        Parameters
        ----------
        tikhonov : float, optional (default 0.0)
            Tikhonov regularization weight on ||Delta_c||^2.
            tikhonov=0 : pure LP  (CLARABEL/ECOS)
            tikhonov>0 : QP       (OSQP/CLARABEL)
            Set > 0 to trade exact interpolation for tighter tau control.

        verbose : bool, optional

        Returns
        -------
        result : dict with keys
            status     str          cvxpy solver status
            p_SC       Chebyshev    corrected polynomial (drop-in for p0)
            delta_c    ndarray(n0)  odd coefficient correction
            tau        float        max|p_SC(x)| on [-1,1]
            tau_base   float        max|p0(x)|   on [-1,1]
            tau_ratio  float        tau / tau_base
            residuals  ndarray(K)   |lambda_k p_SC(lambda_k) - 1|
        """
        delta_c   = cp.Variable(self.n0)
        t         = cp.Variable()
        p_SC_grid = self.p0_grid + self.B_grid @ delta_c

        if tikhonov > 0.0:
            objective = cp.Minimize(t + tikhonov * cp.sum_squares(delta_c))
        else:
            objective = cp.Minimize(t)

        constraints = [p_SC_grid <=  t,
                       p_SC_grid >= -t]

        for k in range(self.K):
            lam    = self.eigenvalues[k]
            p_SC_k = self.p0_eig[k] + self.B_eig[k, :] @ delta_c
            constraints.append(lam * p_SC_k == 1.0)

        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=verbose)

        if prob.status not in ('optimal', 'optimal_inaccurate'):
            return {'status': prob.status, 'p_SC': None,
                    'tau': np.nan, 'tau_base': np.nan}

        dc  = delta_c.value
        tau = float(t.value)

        # Build corrected Chebyshev — same convention as PolynomialApproximators
        coef_SC        = self.p0.coef.copy()
        coef_SC[1::2] += dc
        p_SC = Chebyshev(coef_SC)

        p_SC_eig  = self.p0_eig + self.B_eig @ dc
        residuals = np.abs(self.eigenvalues * p_SC_eig - 1.0)
        tau_base  = float(np.max(np.abs(self.p0_grid)))

        return {
            'status'   : prob.status,
            'p_SC'     : p_SC,
            'delta_c'  : dc,
            'tau'      : tau,
            'tau_base' : tau_base,
            'tau_ratio': tau / tau_base if tau_base > 0 else np.nan,
            'residuals': residuals,
        }

    # ── Sweep ────────────────────────────────────────────────────────────────

    def tau_vs_K_sweep(self, all_eigenvalues, p0_builder, K_values,
                       tikhonov=0.0):
        """
        Sweep over K (number of corrected eigenvalues) and record tau.

        At each K, the K smallest eigenvalues are corrected using the
        base polynomial returned by p0_builder(K).

        Parameters
        ----------
        all_eigenvalues : array-like    full sorted normalized spectrum
        p0_builder      : callable      p0_builder(K) -> Chebyshev
                          Example:
                            def p0_builder(K):
                                a = lams_norm[K]      # new left endpoint
                                d = MangPolynomial.mindegree(eps, a)
                                return MangPolynomial.poly(d, a)
        K_values        : list of int   values of K to test
        tikhonov        : float         regularization weight (default 0)

        Returns
        -------
        results : list of dicts, one per K, each containing:
            K, tau, tau_base, tau_ratio, kappa_eff, kappa,
            deg_factor, residuals, status, degree
        """
        lams    = np.sort(np.asarray(all_eigenvalues, dtype=float))
        results = []

        for K in K_values:
            eigs_K = lams[:K]
            p0_K   = p0_builder(K)
            solver = MinimaxCorrection(p0_K, eigs_K, self.n_grid)
            res    = solver.solve(tikhonov=tikhonov)
            eff    = effective_kappa(lams, K)

            n0_K   = len(p0_K.coef[1::2])
            row = {
                'K'         : K,
                'degree'    : 2 * n0_K - 1,
                'tau'       : res.get('tau', np.nan),
                'tau_base'  : res.get('tau_base', np.nan),
                'tau_ratio' : res.get('tau_ratio', np.nan),
                'kappa_eff' : eff['kappa_eff'],
                'kappa'     : eff['kappa'],
                'deg_factor': eff['degree_reduction_factor'],
                'residuals' : res.get('residuals'),
                'status'    : res.get('status'),
            }
            results.append(row)
            print(f"K={K:3d} | {res['status']:20s} | "
                  f"d={2*n0_K-1:4d} | "
                  f"tau={res.get('tau', float('nan')):8.2f} | "
                  f"kappa_eff={eff['kappa_eff']:8.2f} | "
                  f"factor={eff['degree_reduction_factor']:.2f}x")

        return results

    # ── Plotting ─────────────────────────────────────────────────────────────

    def plot_polynomials(self, result, ax=None):
        """
        Plot p0, p_SC, and 1/x on [lambda_min/2, 1].
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        lam_min = self.eigenvalues[0]
        x_plot  = np.linspace(lam_min * 0.5, 1.0, 1000)

        ax.plot(x_plot, 1.0 / x_plot,              'k--', lw=1.5, label=r'$1/x$')
        ax.plot(x_plot, self.p0(x_plot),            'b-',  lw=1.5, label=r'$p_0$')
        ax.plot(x_plot, result['p_SC'](x_plot),     'r-',  lw=1.5, label=r'$p_{SC}$')
        ax.axhline(result['tau'],      color='r', ls=':', lw=1,
                   label=fr"$\tau_{{SC}}={result['tau']:.2f}$")
        ax.axhline(result['tau_base'], color='b', ls=':', lw=1,
                   label=fr"$\tau_0={result['tau_base']:.2f}$")
        ax.scatter(self.eigenvalues, 1.0 / self.eigenvalues,
                   zorder=5, color='k', s=30, label='corrected eigenvalues')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$p(x)$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_compliance_error(self, result, ax=None):
        """
        Plot |x p(x) - 1| for p0 and p_SC on [lambda_min/2, 1].
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        lam_min = self.eigenvalues[0]
        x_plot  = np.linspace(lam_min * 0.5, 1.0, 1000)

        ax.semilogy(x_plot, np.abs(x_plot * self.p0(x_plot) - 1.0),
                    'b-', lw=1.5, label=r'$|x\,p_0(x)-1|$')
        ax.semilogy(x_plot, np.abs(x_plot * result['p_SC'](x_plot) - 1.0),
                    'r-', lw=1.5, label=r'$|x\,p_{SC}(x)-1|$')
        ax.scatter(self.eigenvalues, result['residuals'],
                   zorder=5, color='k', s=30, label='eigenvalue residuals')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$|x\,p(x)-1|$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax