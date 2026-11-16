"""Container and FITS format of the ``<Map^3>`` data vector.

A file holds one binary table (``EXTNAME = map3``) with columns ``BIN1``,
``BIN2``, ``BIN3`` (tomographic bins, 1-based, sorted), ``THETA1``,
``THETA2``, ``THETA3`` (aperture radii in arcmin) and ``VALUE``, optionally
followed by an image extension ``COVMAT`` with the covariance and the number
of realizations it was estimated from (``NSIM``).  The layout is the one of
the DES Y3 analysis, so Y3 files can still be read.
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table


class Map3Data:
    def __init__(self, z, theta, value, cov=None, nsim=0, name="map3"):
        self.z = np.sort(np.atleast_2d(np.asarray(z, dtype=int)), axis=0)
        self.theta = np.atleast_2d(np.asarray(theta, dtype=float))
        self.value = np.atleast_1d(np.asarray(value, dtype=float)).copy()
        if not (self.z.shape == self.theta.shape == (3, self.value.size)):
            raise ValueError("z and theta must have shape (3, n) and value shape (n,)")
        self.cov = None if cov is None else np.asarray(cov, dtype=float)
        if self.cov is not None and self.cov.shape != (self.size, self.size):
            raise ValueError("covariance shape does not match the data vector")
        self.nsim = int(nsim)
        self.name = name

    @property
    def size(self):
        return self.value.size

    @classmethod
    def from_bins(cls, triplets, filters, **kwargs):
        """Empty data vector for every redshift triplet times filter triplet."""
        triplets = np.atleast_2d(triplets)
        filters = np.atleast_2d(filters)
        nf = filters.shape[1]
        z = np.repeat(triplets.T, nf, axis=1)
        theta = np.tile(filters, (1, len(triplets)))
        return cls(z, theta, np.zeros(z.shape[1]), **kwargs)

    @classmethod
    def from_fits(cls, filename):
        with fits.open(filename) as hdul:
            hdu = hdul[1]
            d = hdu.data
            z = [d["BIN1"], d["BIN2"], d["BIN3"]]
            theta = [d["THETA1"], d["THETA2"], d["THETA3"]]
            cov, nsim = None, 0
            if "COVMAT" in hdul:
                cov = np.array(hdul["COVMAT"].data, dtype=float)
                nsim = hdul["COVMAT"].header.get("NSIM", 0)
            return cls(z, theta, d["VALUE"], cov=cov, nsim=nsim, name=hdu.header["EXTNAME"].lower())

    def to_fits(self, filename, overwrite=True):
        table = Table(
            [*self.z, *self.theta, self.value],
            names=["BIN1", "BIN2", "BIN3", "THETA1", "THETA2", "THETA3", "VALUE"],
        )
        header = fits.Header()
        header["EXTNAME"] = self.name
        header["BIN_TYPE"] = "SSS"
        for i in (1, 2, 3):
            header[f"KERNEL_{i}"] = "nz_source"
        header["SORTZ"] = True
        header["3PT_DATA"] = True
        hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(table, header=header)])
        if self.cov is not None:
            header = fits.Header()
            header["EXTNAME"] = "COVMAT"
            header["NSIM"] = self.nsim
            header["3PT_DATA"] = True
            header["STRT_0"] = 0
            header["NAME_0"] = self.name
            hdul.append(fits.ImageHDU(self.cov, header=header))
        hdul.writeto(filename, overwrite=overwrite)

    def copy(self):
        cov = None if self.cov is None else self.cov.copy()
        return Map3Data(self.z.copy(), self.theta.copy(), self.value, cov=cov, nsim=self.nsim, name=self.name)

    def tomographic_triplets(self):
        """Unique redshift-bin triplets, shape ``(ntriplet, 3)``."""
        return np.unique(self.z, axis=1).T

    def select(self, mask):
        """Restrict the data vector (and covariance) to ``mask`` in place."""
        mask = np.asarray(mask)
        self.z = self.z[:, mask]
        self.theta = self.theta[:, mask]
        self.value = self.value[mask]
        if self.cov is not None:
            self.cov = self.cov[np.ix_(mask, mask)]
        return self

    def triplet_mask(self, triplet):
        return np.all(self.z == np.sort(triplet)[:, None], axis=0)

    def theta_mask(self, theta_min=0.0, theta_max=np.inf):
        """Points whose three aperture radii all lie in ``[theta_min, theta_max]``."""
        return np.all((self.theta >= theta_min) & (self.theta <= theta_max), axis=0)

    def filters(self, triplet):
        """Aperture radii, shape ``(3, n)``, of the points of one redshift triplet."""
        return self.theta[:, self.triplet_mask(triplet)]

    def indices(self, triplet, filters):
        """Positions of the points ``(triplet, filters[:, i])`` in the data vector."""
        filters = np.atleast_2d(filters)
        zmask = self.triplet_mask(triplet)
        out = np.empty(filters.shape[1], dtype=int)
        for i, f in enumerate(filters.T):
            hit = np.flatnonzero(zmask & np.all(np.isclose(self.theta, f[:, None]), axis=0))
            if hit.size != 1:
                raise KeyError(f"triplet {tuple(triplet)} with filters {tuple(f)} matches {hit.size} data points")
            out[i] = hit[0]
        return out

    def inverse_covariance(self):
        if self.cov is None:
            raise ValueError("this data vector has no covariance")
        return np.linalg.inv(self.cov)
