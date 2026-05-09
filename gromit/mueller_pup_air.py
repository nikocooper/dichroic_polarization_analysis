'''
Plots Mueller and Jones Pupil of data from an air measurement. Calibrated system optic parameters are used to construct 
the Mueller matrix from power measurements, then converted to Jones pupil. The function full_mueller_polarimetry_err was
added to the base katsu.polarimetry for error propagation through to the Mueller matrix.
'''

import numpy as np
from katsu import mueller as ks
from katsu import polarimetry as pol
import matplotlib.pyplot as plt
import os
from PIL import Image
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import mueller_to_jones

def circular_mask(shape):
    """
    Create a boolean circular mask for a 2D array shape (H, W)
    """
    H, W = shape
    y, x = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2
    r = min(H, W) // 2

    mask = (x - cx)**2 + (y - cy)**2 <= r**2
    return mask

def jones_pupil_image(J, vmin_mag=None, vmax_mag=None):
    """
    Plot magnitude and phase of 2x2 Jones pupil

    J shape must be (H, W, 2, 2)

    Layout:

        Magnitude        Phase
        |J00| |J01|   ∠J00 ∠J01
        |J10| |J11|   ∠J10 ∠J11
    """

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    fig.suptitle("640 nm Air Jones Pupil", fontsize=20)
    for a in ax.flat:
        a.set_facecolor('black')
    for i in range(2):
        for j in range(2):

            mag = np.abs(J[..., i, j])
            phase = np.angle(J[..., i, j])
            mask = circular_mask(mag.shape)

            alpha = (mask).astype(float)
            # ---------- magnitude (left 2x2 block) ----------
            im1 = ax[i, j].imshow(
                mag,
                cmap="grey",
                vmin=vmin_mag,
                vmax=vmax_mag,
                alpha=alpha
            )

            im2 = ax[i, j+2].imshow(
                phase,
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
                alpha=alpha
            )

            ax[i, j+2].set_title(f"∠J{i}{j}", fontsize=16)

            div2 = make_axes_locatable(ax[i, j+2])
            cax2 = div2.append_axes("right", size="7%", pad="2%")
            fig.colorbar(im2, cax=cax2)

            ax[i, j].set_title(f"|J{i}{j}|", fontsize=16)
            div1 = make_axes_locatable(ax[i, j])
            cax1 = div1.append_axes("right", size="7%", pad="2%")
            fig.colorbar(im1, cax=cax1)

            # remove ticks
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j+2].set_xticks([])
            ax[i, j+2].set_yticks([])

    plt.tight_layout()
    plt.show()

def mm_image(M, normalized=True, vmin=None, vmax=None, mask_func=None):
    """
    Plot 4x4 Mueller matrix with optional circular cropping like Jones pupil.
    
    Parameters
    ----------
    M : ndarray (H, W, 4, 4)
    mask_func : function -> returns (H, W) boolean mask
    """

    fig, ax = plt.subplots(ncols=4, nrows=4, figsize=[9, 7.5])

    # assume shared mask across all components
    mask = circular_mask(M.shape[:2])
    alpha = mask.astype(float)


    for i in range(4):
        for j in range(4):

            ax[i, j].set_title(f"M{i}{j}")

            data = M[..., i, j]

            if normalized:
                if (i == 0) and (j == 0):
                    im = ax[i, j].imshow(
                        data,
                        vmin=vmin,
                        vmax=vmax,
                        cmap='gray',
                        alpha=alpha
                    )
                else:
                    im = ax[i, j].imshow(
                        data / (M[..., 0, 0] + 1e-12),
                        vmin=-1,
                        vmax=1,
                        cmap='bwr',
                        alpha=alpha
                    )
            else:
                im = ax[i, j].imshow(
                    data,
                    vmin=vmin,
                    vmax=vmax,
                    cmap='coolwarm',
                    alpha=alpha
                )

            div = make_axes_locatable(ax[i, j])
            cax = div.append_axes("right", size="7%", pad="2%")
            fig.colorbar(im, cax=cax)

            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    plt.tight_layout()
    plt.show()

def load_images(
    directory,
    extension=".tiff"
):
    """
    Load numbered images from a directory.

    Images must be named like: 1.tiff, 2.tiff, ...

    Parameters
    ----------
    directory : str
        Folder containing numbered images.
    extension : str
        File extension (default ".tiff").

    Returns
    -------
    list of 2D numpy arrays
        Images in numeric order. images[i] corresponds to (i+1).tiff
    """

    # collect numeric filenames
    file_numbers = []
    for fname in os.listdir(directory):
        if fname.lower().endswith(extension):
            try:
                num = int(os.path.splitext(fname)[0])
                file_numbers.append(num)
            except ValueError:
                pass

    file_numbers.sort()

    images = []

    for num in file_numbers:
        path = os.path.join(directory, f"{num}{extension}")
        img = np.array(Image.open(path), dtype=np.float64)
        images.append(img)

    return np.array(images)
NMEAS=40
deg_step = 180 / NMEAS
psg_angles = np.arange(NMEAS+1) * deg_step * np.pi / 180
psa_angles = psg_angles * 5

ims = load_images(r"gromit\042426_air610")
power = ims[:, 465:675, 1085:1295]
power = np.moveaxis(power,0,-1)

def normalize_mueller(M):
    """
    Normalize Mueller matrix per pixel using M00.
    No masking applied.
    """

    M00 = M[..., 0, 0].copy()

    # avoid division by zero
    M00[M00 == 0] = np.nan

    M_norm = np.zeros_like(M, dtype=np.float64)

    for i in range(4):
        for j in range(4):
            M_norm[..., i, j] = M[..., i, j] / M00

    return M_norm

# build Mueller matrix per pixel from power measurements with calibrated system optic parameters
M = pol.full_mueller_polarimetry(psg_angles, psa_angles, power,
                             return_condition_number = False,
                             Min = None,
                             starting_angles={'psg_polarizer': np.radians(-4.709931733314937),
                                            'psg_waveplate': np.radians(-0.9251182115509248),
                                            'psa_waveplate': np.radians(-41.39100065087486),
                                            'psa_polarizer': np.radians(2.0259920107428897)},
                            starting_polarization={'psg_Tmin': 1e-4,
                                                'psg_ret': np.radians(92.09251026647912),
                                                'psa_Tmin': 1e-5,
                                                'psa_ret': np.radians(92.2559884330113)},
                            starting_anglestep={'psg_step': 1,
                                                'psa_step': 1})


R = ks.mueller_rotation(np.radians(90)) # rotate from lab frame to pupil frame
M_r = R @ (M) @ R.T
M_r = normalize_mueller(M_r)
mask = circular_mask(M_r.shape[:2]) # mask outside pupil
M_r[~mask] = 0.0
mm_image(M_r, normalized=False, vmin=-1, vmax=1)
J = mueller_to_jones.mueller_pupil_to_jones(M_r)
J[~mask] = 0.0
norm1 = np.mean(np.abs(J[..., 0, 0][mask])) + np.std(np.abs(J[..., 0, 0][mask])) # normalize by the max diagonal pupil element
norm2 = np.mean(np.abs(J[..., 1, 1][mask])) + np.std(np.abs(J[..., 1, 1][mask]))
norm = max(norm1, norm2)
J /= norm
J[~mask] = np.nan
jones_pupil_image(np.array(J))


# def full_mueller_polarimetry_err(
#     psg_thetas, psa_thetas, power,

#     # ---- random per-step angle noise ----
#     dtheta_psg=0.0,
#     dtheta_psa=0.0,

#     # ---- systematic offset angle errors ----
#     dtheta0_psg_pol=0.0,
#     dtheta0_psg_wp=0.0,
#     dtheta0_psa_wp=0.0,
#     dtheta0_psa_pol=0.0,

#     # ---- other system errors ----
#     extinction_psg=0.0,
#     extinction_psa=0.0,
#     dret_psg=0.0,
#     dret_psa=0.0,

#     detector_alpha=None,
#     return_error=False,

#     # ---- system configuration ----
#     starting_angles={'psg_polarizer': 0,
#                      'psg_waveplate': 0,
#                      'psa_waveplate': 0,
#                      'psa_polarizer': 0},

#     starting_polarization={'psg_Tmin': 0,
#                            'psg_ret': np.pi / 2,
#                            'psa_Tmin': 0,
#                            'psa_ret': np.pi / 2},

#     starting_anglestep={'psg_step': 1,
#                         'psa_step': 1}
# ):

#     nmeas = len(psg_thetas)
#     psg_angles = psg_thetas * starting_anglestep['psg_step']
#     psa_angles = psa_thetas * starting_anglestep['psa_step']

#     frame_shape = power.shape[:-1] if power.ndim > 1 else ()
#     eps = 1e-6

#     # ============================================================
#     # NOMINAL ANGLES
#     # ============================================================

#     psg_theta = starting_angles['psg_waveplate'] + psg_angles
#     psa_theta = starting_angles['psa_waveplate'] + psa_angles

#     # ============================================================
#     # BUILD W
#     # ============================================================

#     def build_W(psg_t, psa_t, Tmin_psg, Tmin_psa,
#                 psg_ret, psa_ret,
#                 psg_pol_angle, psa_pol_angle):

#         Mg = (linear_retarder(psg_t, psg_ret, shape=[*frame_shape, nmeas]) @
#               linear_diattenuator(psg_pol_angle, Tmin_psg,
#                                   shape=[*frame_shape, nmeas]))

#         Ma = (linear_diattenuator(psa_pol_angle, Tmin_psa,
#                                   shape=[*frame_shape, nmeas]) @
#               linear_retarder(psa_t, psa_ret,
#                               shape=[*frame_shape, nmeas]))

#         PSA = Ma[..., 0, :]
#         PSG = Mg[..., :, 0]

#         W = broadcast_kron(PSA[..., None], PSG[..., None])
#         return W.reshape(*W.shape[:-2], 16)

#     W0 = build_W(psg_theta, psa_theta,
#                  starting_polarization['psg_Tmin'],
#                  starting_polarization['psa_Tmin'],
#                  starting_polarization['psg_ret'],
#                  starting_polarization['psa_ret'],
#                  starting_angles['psg_polarizer'],
#                  starting_angles['psa_polarizer'])

#     # ============================================================
#     # RANDOM ANGLE JACOBIANS (PER-MEASUREMENT)
#     # ============================================================

#     def angle_jacobian(psg_shift, psa_shift):
#         grads = []
#         for i in range(nmeas):
#             dpsg = np.zeros_like(psg_theta)
#             dpsa = np.zeros_like(psa_theta)

#             dpsg[i] = psg_shift
#             dpsa[i] = psa_shift

#             Wi = build_W(psg_theta + dpsg,
#                          psa_theta + dpsa,
#                          starting_polarization['psg_Tmin'],
#                          starting_polarization['psa_Tmin'],
#                          starting_polarization['psg_ret'],
#                          starting_polarization['psa_ret'],
#                          starting_angles['psg_polarizer'],
#                          starting_angles['psa_polarizer'])

#             grads.append((Wi - W0) / eps)

#         return np.stack(grads, axis=-1)

#     J_psg = angle_jacobian(eps, 0)
#     J_psa = angle_jacobian(0, eps)

#     dW_psg = np.sqrt(np.sum((J_psg * dtheta_psg)**2, axis=-1))
#     dW_psa = np.sqrt(np.sum((J_psa * dtheta_psa)**2, axis=-1))

#     # ============================================================
#     # SYSTEMATIC OFFSET JACOBIANS
#     # ============================================================

#     def offset(psg_wp=0, psa_wp=0, psg_pol=0, psa_pol=0):
#         Wi = build_W(psg_theta + psg_wp,
#                      psa_theta + psa_wp,
#                      starting_polarization['psg_Tmin'],
#                      starting_polarization['psa_Tmin'],
#                      starting_polarization['psg_ret'],
#                      starting_polarization['psa_ret'],
#                      starting_angles['psg_polarizer'] + psg_pol,
#                      starting_angles['psa_polarizer'] + psa_pol)
#         return (Wi - W0) / eps

#     dW_offset = (
#         offset(psg_wp=eps) * dtheta0_psg_wp +
#         offset(psa_wp=eps) * dtheta0_psa_wp +
#         offset(psg_pol=eps) * dtheta0_psg_pol +
#         offset(psa_pol=eps) * dtheta0_psa_pol
#     )

#     # ============================================================
#     # EXTINCTION / RETARDANCE
#     # ============================================================

#     def param_deriv(param, idx):
#         args = [
#             starting_polarization['psg_Tmin'],
#             starting_polarization['psa_Tmin'],
#             starting_polarization['psg_ret'],
#             starting_polarization['psa_ret']
#         ]
#         args[idx] += eps

#         Wi = build_W(psg_theta, psa_theta,
#                      args[0], args[1], args[2], args[3],
#                      starting_angles['psg_polarizer'],
#                      starting_angles['psa_polarizer'])
#         return (Wi - W0) / eps

#     W_ext_psg = param_deriv('ext_psg', 0)
#     W_ext_psa = param_deriv('ext_psa', 1)
#     W_ret_psg = param_deriv('ret_psg', 2)
#     W_ret_psa = param_deriv('ret_psa', 3)

#     # ============================================================
#     # TOTAL dW
#     # ============================================================

#     dW = (
#         dW_psg +
#         dW_psa +
#         dW_offset +
#         W_ext_psg * extinction_psg +
#         W_ext_psa * extinction_psa +
#         W_ret_psg * dret_psg +
#         W_ret_psa * dret_psa
#     )

#     # ============================================================
#     # RECONSTRUCTION
#     # ============================================================

#     Winv = np.linalg.pinv(W0)

#     M_vec = Winv @ power[..., None]
#     M_vec = M_vec[..., 0]
#     M = M_vec.reshape(*M_vec.shape[:-1], 4, 4)

#     # ============================================================
#     # NORMALIZATION
#     # ============================================================

#     M00 = M[..., 0, 0:1, None]
#     M00_safe = np.where(M00 == 0, 1.0, M00)

#     M_norm = M / M00_safe

#     # ============================================================
#     # ERROR PROPAGATION
#     # ============================================================

#     dM_W = -Winv @ (dW @ M_vec[..., None])
#     dM_W = np.abs(dM_W[..., 0].reshape(*M.shape))

#     if detector_alpha is None:
#         detector_alpha = np.zeros_like(power)

#     dI = detector_alpha * power**2
#     dM_det = Winv @ dI[..., None]
#     dM_det = np.abs(dM_det[..., 0].reshape(*M.shape))

#     dM_total = np.abs(dM_W) + np.abs(dM_det)

#     dM_total_norm = dM_total / M00_safe
#     dM_W_norm = dM_W / M00_safe
#     dM_det_norm = dM_det / M00_safe

#     if return_error:
#         return M_norm, dM_total_norm, dM_W_norm, dM_det_norm

#     return M_norm