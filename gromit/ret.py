'''
Retardance and diattenuation extraction from Jones matrices, with error propagation.
'''
import numpy as np
from numpy.linalg import svd
from scipy.linalg import sqrtm, inv
# Normalize J by max(|J|)
def _normalize_J(J):
    J = np.asarray(J, dtype=complex)
    scale = np.max(np.abs(J))
    if scale == 0:
        return J
    return J / scale

# Polar decomposition unitary factor
def _unitary_polar(J):
    """
    J = U H  (U unitary, H Hermitian positive)
    """
    H = sqrtm(J.conj().T @ J)
    U = J @ inv(H)
    return U

# Retardance  and diattenuation from Jones matrix
def retardance_from_jones(J, return_axis=False):

    J = _normalize_J(J)

    U = _unitary_polar(J)

    U = U / np.sqrt(np.linalg.det(U))

    # eigenvalues of unitary factor
    eigvals, eigvecs = np.linalg.eig(U)

    phases = np.angle(eigvals)

    delta = phases[0] - phases[1]

    # wrap to [0, π]
    if delta > np.pi:
        delta = 2*np.pi - delta

    # diattenuation from Hermitian part
    H = sqrtm(J.conj().T @ J)
    s = np.linalg.svd(H, compute_uv=False)

    D = (s[0] - s[1]) / (s[0] + s[1] + 1e-15)

    if return_axis:
        return delta, eigvecs, D

    return delta, D



#-----------------------------
# Error Propagation

# Diatteuation sensitivity
def delta_diattenuation(S, dS):
    s1, s2 = S
    ds1, ds2 = dS

    denom = (s1 + s2)**2 + 1e-15

    dD = ((ds1 - ds2)*(s1 + s2) - (s1 - s2)*(ds1 + ds2)) / denom
    return np.real(dD)


# Singular value sensitivity
def delta_singular_values(U, S, Vh, dJ):
    V = Vh.conj().T

    ds = np.array([
        np.real(U[:, i].conj().T @ dJ @ V[:, i])
        for i in range(2)
    ])

    return ds


# Retardance error
def delta_retardance(J, dJ):
    """
    First-order perturbation of trace-based retardance.
    Much more stable than eigenvector method.
    """

    J = _normalize_J(J)
    dJ = np.asarray(dJ, dtype=complex)

    U, S, Vh = svd(J)
    W = U @ Vh

    # perturbation of unitary part (stable projection)
    dW = dJ - W @ (W.conj().T @ dJ + dJ.conj().T @ W) / 2

    # trace sensitivity
    tr = np.trace(W)
    dtr = np.trace(dW)

    denom = np.sqrt(1 - (np.abs(tr)/2)**2 + 1e-15)

    ddelta = np.real(
        -2 * np.real(np.conj(tr) * dtr) / (2 * denom + 1e-15)
    )

    delta = 2 * np.arccos(np.clip(np.abs(tr)/2, -1, 1))

    return np.real(delta), np.real(ddelta)


# retardance and diattenuation from Jones matrix with error matrix
def retardance_from_jones_err(J, dJ):

    J = _normalize_J(J)
    dJ = np.asarray(dJ, dtype=complex)

    # SVD
    U, S, Vh = svd(J)

    # diattenuation propagation
    ds = delta_singular_values(U, S, Vh, dJ)
    D = (S[0] - S[1]) / (S[0] + S[1] + 1e-15)
    dD = delta_diattenuation(S, ds)

    # retardance propagation (robust)
    delta, ddelta = delta_retardance(J, dJ)

    return np.abs(delta), np.abs(ddelta), np.abs(D), np.abs(dD)