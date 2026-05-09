import numpy as np

def n_mgf2_cauchy(lam_um):
    """
    MgF2 Cauchy from OptiLayer:
    n = A0 + A1/lambda^2 + A2/lambda^4
    """
    A0 = 1.38400
    A1 = -3.65100e-3
    A2 = 6.42900e-4
    lam2 = lam_um**2
    return A0 + A1 / lam2 + A2 / (lam2**2)


def n_tio2_cauchy(lam_um):
    """
    TiO2 Cauchy from OptiLayer:
    n = A0 + A1/lambda^2 + A2/lambda^4
    """
    A0 = 2.22540
    A1 = 2.33800e-3
    A2 = 7.68800e-3
    lam2 = lam_um**2
    return A0 + A1 / lam2 + A2 / (lam2**2)


def n_al2o3_sellmeier3(lam_um):
    """
    Al2O3 Sellmeier-3 formula from OptiLayer.
    Using:
    n^2 = A0
        + A1*lambda^2/(lambda^2 - A2)
        + A3*lambda^2/(lambda^2 - A4)
        + A5*lambda^2/(lambda^2 - A6)
    """
    A0 = 1.00000
    A1 = 1.02380
    A2 = 3.77588e-3
    A3 = 1.05826
    A4 = 1.22544e-2
    A5 = 5.28079
    A6 = 3.21362e2

    lam2 = lam_um**2
    n2 = (
        A0
        + A1 * lam2 / (lam2 - A2)
        + A3 * lam2 / (lam2 - A4)
        + A5 * lam2 / (lam2 - A6)
    )
    return np.sqrt(n2)


def n_fused_silica_sellmeier(lam_um):
    """
    Fused silica Sellmeier formula:
    n^2 - 1 =
        0.6961663 lambda^2 / (lambda^2 - 0.0684043^2)
      + 0.4079426 lambda^2 / (lambda^2 - 0.1162414^2)
      + 0.8974794 lambda^2 / (lambda^2 - 9.896161^2)
    """
    lam2 = lam_um**2
    n2 = (
        1.0
        + 0.6961663 * lam2 / (lam2 - 0.0684043**2)
        + 0.4079426 * lam2 / (lam2 - 0.1162414**2)
        + 0.8974794 * lam2 / (lam2 - 9.896161**2)
    )
    return np.sqrt(n2)

def n_hfo2_sellmeier(lam_um):
    """
    Simple Sellmeier model for HfO2
    lambda in microns from Al-Kuhaili 2004
    """
    lam2 = lam_um**2

    n2 = 1 + (1.9603 * lam2) / (lam2 - 0.15494**2)

    return np.sqrt(n2)