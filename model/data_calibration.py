import pandas as pd
import numpy as np
import pywt


def calibrate_single_phase(phases):
    """
    Calibrate phase data from the single time moment
    Based on:
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/sys031fp.pdf
        https://github.com/ermongroup/Wifi_Activity_Recognition/.../phase_calibration.m

    :param phases: phase in the single time moment, np.array of shape(1, num of subcarriers)
    :return: calibrate phase, np.array of shape(1, num of subcarriers)
    """

    phases = np.array(phases)
    difference = 0

    calibrated_phase, calibrated_phase_final = np.zeros_like(phases), np.zeros_like(phases)
    calibrated_phase[0] = phases[0]

    phases_len = phases.shape[0]

    for i in range(1, phases_len):
        temp = phases[i] - phases[i - 1]

        if abs(temp) > np.pi:
            difference = difference + 1 * np.sign(temp)

        calibrated_phase[i] = phases[i] - difference * 2 * np.pi

    k = (calibrated_phase[-1] - calibrated_phase[0]) / (phases_len - 1)
    b = np.mean(calibrated_phase)

    for i in range(phases_len):
        calibrated_phase_final[i] = calibrated_phase[i] - k * i - b

    return calibrated_phase_final


def calibrate_phase(phases):
    """
    Calibrate phase data based on the following method:
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/sys031fp.pdf
        https://github.com/ermongroup/Wifi_Activity_Recognition/.../phase_calibration.m

    :param phases: np.array of shape(data len, num of subcarries)
    :return: calibrated phases: np.array of shape(data len, num of subcarriers)
    """

    calibrated_phases = np.zeros_like(phases)

    for i in range(phases.shape[0]):
        calibrated_phases[i] = calibrate_single_phase(np.unwrap(phases[i]))

    return calibrated_phases


def calibrate_amplitude(amplitudes, rssi=1):
    """
    Simple amplitude normalization, that could be multiplied by rsii
    ((data - min(data)) / (max(data) - min(data))) * rssi

    :param amplitudes: np.array of shape(data len, num of subcarriers)
    :param rssi: number
    :return: normalized_amplitude: np.array of shape(data len, num of subcarriers)
    """

    amplitudes = np.array(amplitudes)
    return ((amplitudes - np.min(amplitudes)) / (np.max(amplitudes) - np.min(amplitudes))) * rssi


def calibrate_amplitude_custom(amplitudes, min_val, max_val, rssi=1):
    amplitudes = np.array(amplitudes)
    return ((amplitudes - min_val) / (max_val - min_val)) * rssi


def dwn_noise(vals):
    data = vals.copy()
    threshold = 0.06  # Threshold for filtering

    w = pywt.Wavelet('sym5')
    maxlev = pywt.dwt_max_level(data.shape[0], w.dec_len)

    coeffs = pywt.wavedec(data, 'sym5', level=maxlev)

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))

    datarec = pywt.waverec(coeffs, 'sym5')

    return datarec


def hampel(vals_orig, k=7, t0=3):
    # Make copy so original not edited
    vals = pd.Series(vals_orig.copy())

    # Hampel Filter
    L = 1.4826

    rolling_median = vals.rolling(k).median()
    difference = np.abs(rolling_median - vals)
    median_abs_deviation = difference.rolling(k).median()
    threshold = t0 * L * median_abs_deviation
    outlier_idx = difference > threshold
    vals[outlier_idx] = rolling_median

    # print("vals: ", vals.shape)
    return vals.to_numpy()
