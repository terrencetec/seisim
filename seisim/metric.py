"""Performance metrics"""
import numpy as np


def rms(signal):
    """Root mean square value

    Parameters
    ----------
    signal : array
        Time series.
    
    Return
    ------
    value : float
        RMS value
    """
    value = np.std(signal)

    return value

def blrms(signal):
    """Band-limited RMS"""
    pass


def suppression_ratio(ground, platform):
    """Suppression Ratio
    
    Parameters
    ----------
    ground : array
        Ground motion.
    platform : array
        Isolation platform motion.

    Returns
    -------
    ratio : float
        Suppression ratio, defined as the RMS ground velocity
        divided by the RMS platformd motion
    """
    rms_ground = np.std(ground)
    rms_platform = np.std(platform)
    ratio = rms_ground / rms_platform
    
    return ratio
    

def maximum_suppression(asd_ground, asd_platform):
    """Maximum suppression in amplitude spectrum
    
    Parameters
    ----------
    asd_ground : array
        Amplitude spectral density of ground motion.
    asd_platform : array
        Amplitude spectral density of platform motion

    Returns
    -------
    ms : float
        Maximum ASD suppression.
    """
    suppression = asd_ground / asd_platform
    ms = max(suppression)

    return ms


def band_limited_suppression(
        f, asd_ground, asd_platform, f1=0.05, f2=0.5, **kwargs):
    """Band limited suppression ratio.

    Parameters
    ----------
    f : array
        Frequency array.
    asd_ground : array
        Amplitude spectral density of ground motion.
    asd_platform : array
        Amplitude spectral density of platform motion.
    f1 : float, default 0.05
        Lower frequency limit
    f2 : float, default 0.5
        Upper frequency limit

    Returns
    -------
    blsup : float
        Band-limited suppression ratio
    """
    f1_ = f1
    f2_ = f2
    if f1 > f2:
        f1 = f2_
        f2 = f1_
    mask = (f>f1) & (f<f2)
    f_ = f[mask]
    asd_ground_ = asd_ground[mask]
    asd_platform_ = asd_platform[mask]
    rms_ground = np.sqrt(np.trapezoid(asd_ground_**2, f_))
    rms_platform = np.sqrt(np.trapezoid(asd_platform_**2, f_))
    blsup = rms_ground / rms_platform

    return blsup


def auto_band_limited_suppression(
        f, asd_ground, asd_platform, threshold=2., **kwargs):
    """Auto determined band-limited suppression ratio

    Parameters
    ----------
    f : array
        Frequency array.
    asd_ground : array
        Amplitude spectral density of ground motion.
    asd_platform : array
        Amplitude spectral density of platform motion.
    threshold : float, default 2.
        The suppression threshold.
        Only measures suppression for suppression ratio > threshold.

    Returns
    -------
    blsup : float
        Band-limited suppression ratio
    """
    ratio = asd_ground / asd_platform
    mask = ratio > threshold

    f_ = f[mask]
    asd_ground_ = asd_ground[mask]
    asd_platform_ = asd_platform[mask]
    rms_ground = np.sqrt(np.trapezoid(asd_ground_**2, f_))
    rms_platform = np.sqrt(np.trapezoid(asd_platform_**2, f_))
    blsup = rms_ground / rms_platform

    return blsup


def find_suppression_bands(f, asd_ground, asd_platform, threshold=2):
    """
    Find frequency bands where ground/platform ratio > threshold.

    Parameters
    ----------
    f : array
        Frequency array
    ground_asd : array
        Ground motion ASD
    platform_asd : array
        Platform motion ASD
    threshold : float
        Suppression threshold (default = 2)

    Returns
    -------
    bands : list of tuples
        List of (f_start, f_end) bands
    """

    ratio = asd_ground / asd_platform
    mask = ratio > threshold

    bands = []
    start = None

    for i in range(len(f)):
        if mask[i] and start is None:
            start = f[i]

        if not mask[i] and start is not None:
            bands.append((start, f[i-1]))
            start = None

    if start is not None:
        bands.append((start, f[-1]))

    return bands
