"""Benchmark"""
import numpy as np
import scipy.signal

from . import metric


class Benchmark:
    """Benchmark class"""
    def __init__(self, t, ground, platform):
        """Constructor

        Parameters
        ----------
        t : array
            Time array
        ground : array
            Ground motion time series. (velocity)
        platform : array
            Isolation platform motion time series. (velocity)
        """
        self.t = t
        self.ground = ground
        self.platform = platform

    def run(self, **kwargs):
        """Run benchmark

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to scipy.signal.welch() and
            metric functions.

        Returns
        -------
        results : dict
            Benchmark results in metric:score pairs.
        """
        # Calculate amplitude spectral densities
        dt = self.t[1] - self.t[0]
        fs = 1 / dt
        
        if "nperseg" not in kwargs.keys() or kwargs["nperseg"] is None:
            kwargs["nperseg"] = int(len(self.t)/5)

        f, psd_ground = scipy.signal.welch(self.ground, fs=fs, **kwargs)
        _, psd_platform = scipy.signal.welch(self.platform, fs=fs, **kwargs)

        asd_ground = psd_ground**.5
        asd_platform = psd_platform**.5


        # Calculate performance metrics
        ## Root-mean-squared values
        rms_ground = metric.rms(self.ground)
        rms_platform = metric.rms(self.platform)

        ## Suppression ratio
        suppression_ratio = metric.suppression_ratio(
            self.ground, self.platform)

        ## Maximum suppression
        maximum_suppression = metric.maximum_suppression(
            asd_ground, asd_platform)

        ## Band limited suppression ratio
        band_limited_suppression = metric.band_limited_suppression(
            f, asd_ground, asd_platform, **kwargs)

        ## Auto band limited suppression ratio
        auto_band_limited_suppression = metric.auto_band_limited_suppression(
            f, asd_ground, asd_platform, **kwargs)


        # Put perfomance metrics into results
        results = {
            "rms_ground": rms_ground,
            "rms_platform": rms_platform,
            "suppression_ratio": suppression_ratio,
            "maximum_suppression": maximum_suppression,
            "band_limited_suppression": band_limited_suppression,
            "auto_band_limited_suppression": auto_band_limited_suppression,
        }

        return results
        

