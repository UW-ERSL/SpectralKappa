import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, Operator
from numpy.polynomial import Chebyshev
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from PolynomialApproximators import (SunderhaufPolynomial,
                            RemezPolynomial,
                            MangPolynomial, SpectralPolynomial,spectral_correction)

from PoissonFunctions import (build_1d_poisson, eigs_1d_poisson)
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
import time


# ==============================================================================
# QSVT linear solver
# ==============================================================================
class StandardQSVT:
    def __init__(self, A, b, kappa=None, nShots=1000, target_error=None, degree_override=None,
                 polyMethod='Remez'):
        """
        Parameters
        ----------
        A            : (N, N) real matrix; all singular values must be in (0, 1).
        b            : (N,) right-hand side; normalised
        kappa        : condition number estimate (if None computed here).
        nShots       : shots for QASM simulator (unused in statevector mode).
        target_error : target L-inf error for the 1/x Chebyshev approximation.
        polyMethod   : polynomial method for 1/x approximation.  Options:
                         'remez', 'RemezCutoff',
                         'sunderhauf', 'SunderhaufCutoff',
                         'mang', 'MangCutoff',
                         'Eigenvalue'  (requires all singular values of A).
        sigma_cutoff : cutoff parameter b for '*Cutoff' methods; ignored otherwise.
        eigenvalues  : (N,) array of ALL singular values of A, required (or
                       auto-computed from A) when polyMethod='Eigenvalue'.
        n_factor     : over-parameterisation ratio for 'Eigenvalue'.
                       n = ceil(n_factor * N), d = 2n-1.  Default 1.5.
        degree_override : if given, use this degree for all polynomials (overrides target_error).
        """
        self.A = A
        self.b = b
        self.nShots = nShots
        self.n = int(np.log2(len(b)))
        self.ancilla_qubits = 1
        self.degree_override = degree_override

        # ── Polynomial method selection ───────────────────────────────
        if polyMethod.lower() == 'remez':
            self.polyMethod = RemezPolynomial
        elif polyMethod.lower() == 'sunderhauf':
            self.polyMethod = SunderhaufPolynomial
        elif polyMethod.lower() == 'mang':
            self.polyMethod = MangPolynomial
        else:
            raise ValueError(f"Unknown polyMethod '{polyMethod}'.")
 
        if kappa is None:
            s = np.linalg.svd(A, compute_uv=False)
            self.kappa = s[0] / s[-1]
            print(f"Computed κ = {self.kappa:.4f}")
        else:
            self.kappa = kappa

        self.dataOK = True # = self._validate_input()
        self.target_error = target_error

        self.angles, self.tau, self.achieved_error = \
            self._get_inverse_phases(self.kappa, target_error=target_error)

        print(f"Generated {len(self.angles)} phase angles for degree {len(self.angles) - 1}")

    # ------------------------------------------------------------------
    # Phase computation
    # ------------------------------------------------------------------
    def _get_inverse_phases(self, kappa, target_error=None):
        a = 1.0 / kappa

        
        if self.degree_override is not None:
            degree = self.degree_override
        elif target_error is not None:
            degree = self.polyMethod.mindegree(target_error, a)
            self.degree = degree
        else:
            degree = max(self.degree, 1)
            if degree % 2 == 0:
                degree += 1
           

        poly = self.polyMethod.poly(degree, a)
        achieved_error = None # placeholder; self.polyMethod.error_for_degree(degree, a)
   
        N_sample = 25 * degree
        x_s    = np.linspace(-1, 1, N_sample)
        M      = np.max(np.abs(poly(x_s))) / np.cos(np.pi * degree / (2 * N_sample))

        tau            = M               # scaling to recover unnormalised A^{-1}b
        poly_normalised = Chebyshev(poly.coef / M)

        max_val = np.max(np.abs(poly_normalised(np.linspace(-1, 1, 2000))))
        if max_val > 0.999:
            scale           = 0.999 / max_val
            poly_normalised = Chebyshev(poly_normalised.coef * scale)
            tau            /= scale


        phases = QuantumSignalProcessingPhases(poly_normalised, signal_operator="Wx")
        return [float(phi) for phi in phases], tau, achieved_error

    # ------------------------------------------------------------------
    # Block encoding  -- ROTATION form to match pyqsp Wx convention
    # ------------------------------------------------------------------
    def get_block_encoding(self):
        """
        One would use effcient block-encoding constructions for sparse or structured A, but here we
        build the (2N x 2N) block-encoding unitary matching pyqsp's Wx signal:

            U_BE = [[ A,                i*sqrt(I - A A†) ],
                    [ i*sqrt(I - A†A),       A†          ]]

        This ensures the effective 2x2 sub-unitary for each singular value
        sigma_i is exactly W_pyqsp(sigma_i) = [[sigma_i, i*sqrt(1-sigma_i^2)], ...].
        """
        N     = self.A.shape[0]
        I     = np.eye(N)
        A_dag = self.A.conj().T

        sqrt_r = scipy.linalg.sqrtm(I - self.A @ A_dag)
        sqrt_l = scipy.linalg.sqrtm(I - A_dag @ self.A)

        U_matrix = np.block([[self.A,   1j * sqrt_r],
                              [1j * sqrt_l, A_dag  ]])

        err = np.max(np.abs(U_matrix @ U_matrix.conj().T - np.eye(2 * N)))
        if err > 1e-10:
            print(f"Warning: block encoding not unitary, max error = {err:.2e}")


        return Operator(U_matrix)

    # ------------------------------------------------------------------
    # Phase gate on ancilla  (diagonal Rz, NOT Rx)
    # ------------------------------------------------------------------
    def _apply_projector_phase(self, circuit, phi, anc_qubit):
        """
        Apply P(phi) = diag(e^{i*phi}, e^{-i*phi}) on the ancilla.

        Qiskit Rz(theta) = diag(e^{-i*theta/2}, e^{+i*theta/2})
        => Rz(-2*phi)    = diag(e^{+i*phi},      e^{-i*phi})     ✓
        """
        circuit.rz(-2.0 * phi, anc_qubit)   # Z-rotation, NOT X-rotation

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def construct_qsvt_circuit(self):
        """
        QSVT sequence: P(phi_0), U_BE, P(phi_1), U_BE, ..., U_BE, P(phi_d)

        Gate appended as list(q_data) + list(q_anc) so that Qiskit places
        q_anc as the most-significant-bit (MSB) block selector, matching
        the mathematical block-encoding convention.
        """
        q_anc  = QuantumRegister(self.ancilla_qubits, 'anc')
        q_data = QuantumRegister(self.n, 'b')
        c      = ClassicalRegister(self.n + self.ancilla_qubits, 'meas')
        qc     = QuantumCircuit(q_anc, q_data, c)

        qc.prepare_state(Statevector(self.b), q_data)
        qc.barrier()

        U_op   = self.get_block_encoding()
        U_gate = U_op.to_instruction()

        for i in range(len(self.angles) - 1):
            self._apply_projector_phase(qc, self.angles[i], q_anc[0])
            qc.append(U_gate, list(q_data) + list(q_anc))

        self._apply_projector_phase(qc, self.angles[-1], q_anc[0])
        qc.barrier()
        qc.measure(range(qc.num_qubits), range(qc.num_qubits))

        print(f"Circuit width: {qc.width()}, depth: {qc.depth()}")
        return qc

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def _validate_input(self):
        s = np.linalg.svd(self.A, compute_uv=False)
        if np.any(s >= 1.0):
            print("Warning: all singular values must be strictly < 1.")
            return False
        return True

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, stateVector=True):
        """
        Run QSVT and return the unit-normalised solution direction and success
        probability.

        Returns
        -------
        u_dir : (N,) ndarray
            Unit-normalised solution direction  Re(sv[0::2]) / ||Re(sv[0::2])||,
            proportional to p(A)|b>.
        success_prob : float
            Post-selection probability P = ||sv[0::2]||^2  (complex norm, includes
            both the real/target part and the imaginary QSP-completion part).
        norm_real : float
            ||Re(sv[0::2])||  — the real-part norm only.  Use this (not
            sqrt(success_prob)) to recover the physical compliance:

                C_qsvt = (b @ u_dir) * solver.tau * norm_real
                C_true = b @ np.linalg.solve(A, b)
                rel_compliance_err = abs(C_qsvt - C_true) / C_true

            Derivation: p_norm = p / tau, so Re(sv[0::2]) = p_norm(A)|b> / ||b||
            (with ||b||=1), giving ||p(A)b|| = tau * norm_real.
            sqrt(success_prob) overestimates this because success_prob includes
            the imaginary QSP-completion polynomial, which is orthogonal to the
            target but lives in the same ancilla=0 subspace.

        Statevector index: k = data_idx * 2 + anc_bit
        Ancilla=0 subspace: sv.data[0::2]  (even indices, correct data order).
        Solution direction: real part of the extracted amplitudes.
        """
        if not self.dataOK:
            return None

        qc = self.construct_qsvt_circuit()

        if stateVector:
            print("Running statevector simulation...")
            qc_sv = qc.copy()
            qc_sv.remove_final_measurements()
            sv = Statevector.from_instruction(qc_sv)
            u_qsvt = sv.data[0::2]          # ancilla=0 => even indices
            success_prob = float(np.sum(np.abs(u_qsvt)**2))
        else:
            print("Running QASM simulation...")
            backend = Aer.get_backend('qasm_simulator')
            t_qc    = transpile(qc, backend)
            counts  = backend.run(t_qc, shots=self.nShots).result().get_counts()

            # Qiskit bitstring: rightmost char = qubit 0 = ancilla
            success_counts = {k: v for k, v in counts.items() if k[-1] == '0'}
            total_success  = sum(success_counts.values())
            if total_success == 0:
                print("ERROR: no shots with ancilla=0.")
                return np.zeros(2**self.n)

            success_prob = total_success / self.nShots
            u_qsvt = np.zeros(2**self.n, dtype=complex)
            for bitstr, count in success_counts.items():
                idx          = int(bitstr[:-1], 2)
                u_qsvt[idx]  = np.sqrt(count / total_success)

        # Solution direction is in the REAL PART (imaginary is the QSP completion).
        # The ancilla=0 subspace sv[0::2] has complex amplitudes:
        #   Re(sv[0::2]) ~ p_norm(A)|b>   (target polynomial)
        #   Im(sv[0::2]) ~ completion poly (orthogonal, irrelevant)
        # success_prob = ||sv[0::2]||^2 = ||Re||^2 + ||Im||^2  (COMPLEX norm).
        # For physical compliance recovery we need ||Re(sv[0::2])|| = norm_real,
        # because ||p(A)b|| = tau * norm_real * ||b||  (with ||b||=1).
        # Using sqrt(success_prob) would overestimate this by the imaginary contribution.
        u_real    = u_qsvt.real
        norm_real = float(np.linalg.norm(u_real))
        if norm_real < 1e-12:
            print("ERROR: real part of extracted state has near-zero norm.")
            return None
        return u_real / norm_real, success_prob, norm_real


class PureSpectralQSVT(StandardQSVT):
    """
    QSVT solver using the pure spectral polynomial, which interpolates
    1/x exactly at all known eigenvalues without a base polynomial.

    This is optimal when K = N (all eigenvalues known), yielding the
    lowest possible degree d = 2*ceil(n_factor*N) - 1. When K < N,
    use SpectrallyBootstrappedQSVT instead.

    Parameters
    ----------
    A          : (N, N) real matrix; all singular values in (0, 1).
    b          : (N,) right-hand side; normalised.
    eigenvalues: (K,) known eigenvalues, normalised to (0, 1).
    kappa      : condition number estimate.
    n_factor   : over-parameterisation ratio (default 1.5, giving d = 3N-1).
                 Higher values reduce tau at the cost of higher degree.
    """

    def __init__(self, A, b, eigenvalues, kappa=None, n_factor=1.5, **kwargs):
        self.eigenvalues = np.asarray(eigenvalues)
        self.n_factor = n_factor
        # Build the spectral polynomial before calling super().__init__,
        # which triggers _get_inverse_phases
        self._spectral_poly = SpectralPolynomial(self.eigenvalues,
                                                  n_factor=n_factor)
        # Don't pass polyMethod to super — we override _get_inverse_phases
        # Pass a dummy polyMethod to satisfy the base class
        super().__init__(A, b, kappa=kappa, polyMethod='Mang', **kwargs)

    def _get_inverse_phases(self, kappa, target_error=None):
        """
        Override: use the pure spectral polynomial directly.
        No base polynomial, no correction.
        """
        poly = self._spectral_poly.poly()
        degree = self._spectral_poly.mindegree()
        self.degree = degree

        # ── normalisation (identical to base class) ───────────────────
        N_sample        = 25 * degree
        x_s             = np.linspace(-1, 1, N_sample)
        M               = (np.max(np.abs(poly(x_s)))
                           / np.cos(np.pi * degree / (2 * N_sample)))
        tau             = M
        poly_normalised = Chebyshev(poly.coef / M)

        max_val = np.max(np.abs(poly_normalised(np.linspace(-1, 1, 2000))))
        if max_val > 0.999:
            scale           = 0.999 / max_val
            poly_normalised = Chebyshev(poly_normalised.coef * scale)
            tau            /= scale

        phases = QuantumSignalProcessingPhases(poly_normalised,
                                               signal_operator="Wx")

        return [float(phi) for phi in phases], tau, None
    

class SpectrallyBootstrappedQSVT(StandardQSVT):
    """
    Extends StandardQSVT to apply the min-norm hybrid correction before
    computing QSP phase angles.  All other pipeline steps are unchanged.

    Parameters
    ----------
    lam_K : np.ndarray
        K known eigenvalues (singular values of A, already normalised
        to lie in (0, 1]) to be corrected to machine precision.
    rcond : float
        SVD truncation threshold for the Gram system (default 1e-10).
    """

    def __init__(self, A, b, lam_K, rcond=1e-10, degree_override=None, **kwargs):
        self.lam_K = np.asarray(lam_K)
        self.rcond = rcond
        super().__init__(A, b, degree_override=degree_override, **kwargs)

    def _get_inverse_phases(self, kappa, target_error=None):
        """
        Override: build base polynomial, apply hybrid correction,
        then proceed with normalisation and QSP phase computation.
        """
        

        a  = 1.0 / kappa
        if self.degree_override is not None:
            degree = self.degree_override
        else:
            degree = self.polyMethod.mindegree(target_error, a)

        print("degree override: ", self.degree_override)
        print("computed degree: ", degree)
        self.degree = degree
        # ── base polynomial ───────────────────────────────────────────
        p0 = self.polyMethod.poly(degree, a)

        # ── hybrid correction ─────────────────────────────────────────
        c_corr        = spectral_correction(p0, self.lam_K, rcond=self.rcond)
        coef_H        = p0.coef.copy()
        coef_H[1::2] += c_corr
        poly          = Chebyshev(coef_H)

        # ── normalisation (identical to base class) ───────────────────
        N_sample        = 25 * degree
        x_s             = np.linspace(-1, 1, N_sample)
        M               = (np.max(np.abs(poly(x_s)))
                           / np.cos(np.pi * degree / (2 * N_sample)))
        tau             = M
        poly_normalised = Chebyshev(poly.coef / M)

        max_val = np.max(np.abs(poly_normalised(np.linspace(-1, 1, 2000))))
        if max_val > 0.999:
            scale           = 0.999 / max_val
            poly_normalised = Chebyshev(poly_normalised.coef * scale)
            tau            /= scale

        phases = QuantumSignalProcessingPhases(poly_normalised,
                                               signal_operator="Wx")

        return [float(phi) for phi in phases], tau, None




# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":

    A = np.array([[0.5, 0.1], [0.1, 0.3]])
    b = np.array([1.0, 0.0])
    solver = StandardQSVT(A, b, polyMethod='Remez', target_error=0.01)
    u_dir, success_prob, norm_real = solver.solve(stateVector=True)
    print(f"StandardQSVT solution direction: {u_dir}, success probability: {success_prob:.4f}, norm_real: {norm_real:.4f}")
