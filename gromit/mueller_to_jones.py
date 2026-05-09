'''
Mueller to Jones eigendecomposition with error propagation.
'''
import numpy as np

# Constants

sigma = np.array([
    [[1, 0], [0, 1]],
    [[1, 0], [0, -1]],
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]]
], dtype=np.complex128)

U = np.array([
    [1, 0, 0, 1],
    [1, 0, 0, -1],
    [0, 1, 1, 0],
    [0, 1j, -1j, 0]
], dtype=np.complex128)

# Precompute Pi tensors
Pi = []
for i in range(4):
    for j in range(4):
        kron_product = np.kron(sigma[i], np.conjugate(sigma[j]))
        Pi.append(0.5 * U @ kron_product @ np.conjugate(U.T))
Pi = np.stack(Pi, axis=2)  # (4,4,16)


# Mueller to Jones eigendecomposition

def mueller_to_jones(M, remove_global_phase=True):
    """
    Extract dominant (non-depolarizing) Jones matrix from Mueller matrix.

    Parameters
    ----------
    M : (4,4) array
        Mueller matrix
    remove_global_phase : bool
        Removes global phase ambiguity

    Returns
    -------
    J_dom : (2,2) complex ndarray
        Dominant Jones matrix
    eigenvalues : (4,) array
        Coherency eigenvalues (depolarization weights)
    """

    M = np.array(M, dtype=np.complex128)

    # Normalize
    M = M / M[0, 0]

    # Build coherency matrix
    M_flat = M.flatten()
    H = np.sum(Pi * M_flat.reshape(1, 1, -1), axis=2) / 4

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Build Jones matrices from eigenvectors
    JonesMatrices = []
    for k in range(4):
        v = eigenvectors[:, k]
        J = sum(v[j] * sigma[j] for j in range(4))
        JonesMatrices.append(J)

    # Take dominant component
    J_dom = JonesMatrices[0]

    # Remove global phase ambiguity
    if remove_global_phase:
        phi = np.angle(J_dom[0, 0])
        if np.abs(J_dom[0, 0]) > 0:
            J_dom = J_dom * np.exp(-1j * phi)

    return np.array(J_dom)

 # Eigenvector sensitivity
def delta_eigenvector(H, dH, eps=1e-12):
    """
    First-order perturbation of dominant eigenvector.
    """

    evals, evecs = np.linalg.eigh(H)

    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    v0 = evecs[:, 0]
    l0 = evals[0]

    dv0 = np.zeros_like(v0, dtype=np.complex128)

    for k in range(1, 4):
        vk = evecs[:, k]
        lk = evals[k]

        denom = (l0 - lk) + eps

        coeff = (vk.conj().T @ dH @ v0) / denom

        dv0 += coeff * vk

    return dv0, v0

# Mueller to Jones with error propagation
def mueller_to_jones_err(M, dM, eps=1e-12):
    """
    Analytic first-order propagation of Mueller error → Jones error
    """

    M = np.array(M, dtype=np.complex128)
    dM = np.array(dM, dtype=np.complex128)

    # normalize
    M = M / (M[0, 0] + eps)
    dM = dM / (M[0, 0] + eps)

    # ---- build H ----
    M_flat = M.flatten()
    dM_flat = dM.flatten()

    H = np.sum(Pi * M_flat.reshape(1, 1, -1), axis=2) / 4
    dH = np.sum(Pi * dM_flat.reshape(1, 1, -1), axis=2) / 4

    # ---- eigen perturbation ----
    dv0, v0 = delta_eigenvector(H, dH, eps)

    # ---- nominal Jones ----
    J = sum(v0[i] * sigma[i] for i in range(4))

    # ---- Jones perturbation ----
    dJ = sum(dv0[i] * sigma[i] for i in range(4))

    # global phase normalization (propagated approximately)
    if np.abs(J[0, 0]) > 0:
        phase = np.angle(J[0, 0])
        J = J * np.exp(-1j * phase)
        dJ = dJ * np.exp(-1j * phase)

    return J, dJ

def mueller_to_jones_batch(M_list):
    """
    Convert multiple Mueller matrices to Jones matrices.
    Useful for Monte Carlo error propagation.
    """
    Js = []
    evals = []

    for M in M_list:
        J = mueller_to_jones(M)
        Js.append(J)

    return np.array(Js)


def mueller_pupil_to_jones(M_pupil, remove_global_phase=True, eps=1e-8):
    """
    Convert Mueller pupil (Nx, Ny, 4, 4) → Jones pupil (Nx, Ny, 2, 2)
    Ignores pixels where M00 ~ 0 (outside pupil)
    """

    Nx, Ny = M_pupil.shape[:2]

    J_pupil = np.full((Nx, Ny, 2, 2), np.nan, dtype=np.complex128)
    mask = np.abs(M_pupil[:, :, 0, 0]) > eps  # valid pupil region
    for i in range(Nx):
        for j in range(Ny):

            if not mask[i, j]:
                continue  # skip outside pupil

            M_pixel = M_pupil[i, j]

            J = mueller_to_jones(M_pixel, remove_global_phase)

            J_pupil[i, j] = J
    max_amp = np.nanmax(np.abs(J_pupil))
    J_pupil = J_pupil / max_amp
    return J_pupil



# Mueller to Jones with error propagation


def mueller_to_jones_err(M, dM, eps=1e-12):
    """
    Analytic first-order propagation of Mueller error → Jones error
    """

    M = np.array(M, dtype=np.complex128)
    dM = np.array(dM, dtype=np.complex128)

    # normalize
    M = M / (M[0, 0] + eps)
    dM = dM / (M[0, 0] + eps)

    # ---- build H ----
    M_flat = M.flatten()
    dM_flat = dM.flatten()

    H = np.sum(Pi * M_flat.reshape(1, 1, -1), axis=2) / 4
    dH = np.sum(Pi * dM_flat.reshape(1, 1, -1), axis=2) / 4

    # ---- eigen perturbation ----
    dv0, v0 = delta_eigenvector(H, dH, eps)

    # ---- nominal Jones ----
    J = sum(v0[i] * sigma[i] for i in range(4))

    # ---- Jones perturbation ----
    dJ = sum(dv0[i] * sigma[i] for i in range(4))

    # global phase normalization (propagated approximately)
    if np.abs(J[0, 0]) > 0:
        phase = np.angle(J[0, 0])
        J = J * np.exp(-1j * phase)
        dJ = dJ * np.exp(-1j * phase)

    return J, dJ


def mueller_to_jones_error_batch(M_list, dM_list):

    Js = []
    dJs = []

    for M, dM in zip(M_list, dM_list):
        J, dJ = mueller_to_jones_err(M, dM)
        Js.append(J)
        dJs.append(dJ)

    return np.array(Js), np.array(dJs)