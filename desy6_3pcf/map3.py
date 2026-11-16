"""Aperture-mass skewness from binned shear 3PCF natural components.

The integration kernels below are adapted from TreeCorr
(``GGGCorrelation._calculateT`` and ``GGGCorrelation.calculateMap3``),
Copyright (c) 2003-2024 by Mike Jarvis, distributed under a BSD-style
licence (see https://github.com/rmjarvis/TreeCorr/blob/main/TreeCorr_LICENSE).
They are rewritten here so that a *theory* prediction of the natural
components on the measurement bins is integrated exactly like the measured
ones, which makes the binning of the estimator part of the model.

References: Jarvis, Bernstein & Jain (2004, JBJ) and Schneider, Kilbinger &
Lombardi (2005, SKL).
"""
import numpy as np


def _calculate_T(s, t, k2, k3):
    """Kernels ``T_0..T_3`` for apertures ``(R, k2 R, k3 R)`` (SKL eqs. 62-69).

    ``s`` and ``t`` are the two triangle sides as complex numbers in units of
    ``R``, with ``s`` on the real axis.  For ``k2 = k3 = 1`` this reduces to
    JBJ eqs. 51-52.
    """
    q1 = (s + t) / 3.0
    q2 = q1 - t
    q3 = q1 - s

    a1, a2, a3 = np.abs(q1) ** 2, np.abs(q2) ** 2, np.abs(q3) ** 2
    a123 = a1 * a2 * a3
    b1 = np.conjugate(q1) ** 2 * q2 * q3
    b2 = np.conjugate(q2) ** 2 * q1 * q3
    b3 = np.conjugate(q3) ** 2 * q1 * q2

    k1sq, k2sq, k3sq = 1.0, k2 * k2, k3 * k3
    Theta2 = np.sqrt((k1sq * k2sq + k1sq * k3sq + k2sq * k3sq) / 3.0)
    k1sq, k2sq, k3sq = k1sq / Theta2, k2sq / Theta2, k3sq / Theta2
    Theta4 = Theta2 * Theta2
    Theta6 = Theta4 * Theta2
    S = k1sq * k2sq * k3sq

    Z = (
        (2 * k2sq + 2 * k3sq - k1sq) * a1
        + (2 * k3sq + 2 * k1sq - k2sq) * a2
        + (2 * k1sq + 2 * k2sq - k3sq) * a3
    ) / (6 * Theta2)
    expfactor = -S * np.exp(-Z) / Theta4

    f1 = (k2sq + k3sq) / 2 + (k2sq - k3sq) * (q2 - q3) / (6 * q1)
    f2 = (k3sq + k1sq) / 2 + (k3sq - k1sq) * (q3 - q1) / (6 * q2)
    f3 = (k1sq + k2sq) / 2 + (k1sq - k2sq) * (q1 - q2) / (6 * q3)
    f1c, f2c, f3c = np.conjugate(f1), np.conjugate(f2), np.conjugate(f3)

    g1c = np.conjugate(k2sq * k3sq + (k3sq - k2sq) * k1sq * (q2 - q3) / (3 * q1))
    g2c = np.conjugate(k3sq * k1sq + (k1sq - k3sq) * k2sq * (q3 - q1) / (3 * q2))
    g3c = np.conjugate(k1sq * k2sq + (k2sq - k1sq) * k3sq * (q1 - q2) / (3 * q3))

    T0 = expfactor * a123 * f1c**2 * f2c**2 * f3c**2 / (24.0 * Theta6)
    T1 = expfactor * (
        a123 * f1**2 * f2c**2 * f3c**2 / (24 * Theta6)
        - b1 * f1 * f2c * f3c * g1c / (9 * Theta4)
        + (b1**2 * g1c**2 + 2 * k2sq * k3sq * a2 * a3 * b1 * f2c * f3c) / (a123 * 27 * Theta2)
    )
    T2 = expfactor * (
        a123 * f1c**2 * f2**2 * f3c**2 / (24 * Theta6)
        - b2 * f1c * f2 * f3c * g2c / (9 * Theta4)
        + (b2**2 * g2c**2 + 2 * k1sq * k3sq * a1 * a3 * b2 * f1c * f3c) / (a123 * 27 * Theta2)
    )
    T3 = expfactor * (
        a123 * f1c**2 * f2c**2 * f3**2 / (24 * Theta6)
        - b3 * f1c * f2c * f3 * g3c / (9 * Theta4)
        + (b3**2 * g3c**2 + 2 * k1sq * k2sq * a1 * a2 * b3 * f1c * f2c) / (a123 * 27 * Theta2)
    )
    return T0, T1, T2, T3


class Map3Transform:
    """Linear map from binned natural components to ``<Map^3>``.

    The kernels depend only on the 3PCF bins and the aperture radii, so they
    are computed once and reused for every redshift combination and every
    likelihood evaluation.

    Parameters
    ----------
    d2, d3 : array
        Representative values of the two side-length bins (same units as
        ``filters``).
    phi : array
        Centres of the opening-angle bins, in radians.
    dlnd : float
        Logarithmic width of the side-length bins.
    dphi : float
        Width of the opening-angle bins, in radians.
    filters : array, shape (3, nfilter)
        Aperture radii ``(R1, R2, R3)`` of every data point.
    """

    def __init__(self, d2, d3, phi, dlnd, dphi, filters):
        filters = np.atleast_2d(np.asarray(filters, dtype=float))
        if filters.shape[0] != 3:
            raise ValueError("filters must have shape (3, nfilter)")
        D2, D3, PHI = (x.ravel() for x in np.meshgrid(d2, d3, phi, indexing="ij"))
        self.shape = (len(d2), len(d3), len(phi))
        self.filters = filters

        weights = np.empty((filters.shape[1], 4, D2.size), dtype=complex)
        for i, (R1, R2, R3) in enumerate(filters.T):
            s = D2 / R1
            d3r = D3 / R1
            t = d3r * np.exp(1j * PHI)
            # area element s ds d^2t / (2 pi) in the logarithmic/linear bins
            measure = (s * s * dlnd) * (d3r * d3r * dlnd * dphi / (2 * np.pi))
            # SAS binning visits each triangle once per vertex but only in one
            # orientation; adding the kernel with apertures 2 and 3 swapped
            # accounts for the mirrored triangles.
            T = np.array(_calculate_T(s, t, R2 / R1, R3 / R1))
            T += np.array(_calculate_T(s, t, R3 / R1, R2 / R1))
            weights[i] = T * measure
        self._weights = weights.reshape(filters.shape[1], -1)

    def __call__(self, gamma):
        """Return ``<Map^3>`` for natural components of shape ``(4, nd2, nd3, nphi)``."""
        gamma = np.asarray(gamma)
        if gamma.shape != (4,) + self.shape:
            raise ValueError(f"expected natural components of shape {(4,) + self.shape}, got {gamma.shape}")
        return 0.25 * np.real(self._weights @ gamma.ravel())
