from typing import Any

import mne
import mne.export
import numpy as np
import pandas as pd
import scipy.fft
from sklearn.experimental import enable_iterative_imputer
import sklearn.ensemble
import sklearn.impute

# import matplotlib.pyplot as plt
# import mne.time_frequency
# import scipy.fft
# import sklearn.metrics

import eeg.default


def clean_with_fft(
    raw: mne.io.Raw,
    artifact_intervals: list[tuple[float, float, str]],
    artifact_specs: dict[str, dict[str, list]],
    sample_freq: int,
) -> pd.DataFrame:
    # sample_freq = raw.info["sfreq"]
    raw_df: pd.DataFrame = raw.to_data_frame()
    # .to_data_frame returns in microvolts, need to convert to volts
    # for future operations
    raw_df.loc[:, raw_df.columns != "time"] = (
        raw_df.loc[:, raw_df.columns != "time"] / 1e6
    )
    n_samples = raw_df.shape[0]
    # fft_discretization = 32767
    # zeroing_value = 0.001 * fft_discretization
    zeroing_value = 0.0001

    # ch_means = raw_df.mean()
    # ch_maxes = raw_df.max()

    # Максимальная частота составляет половину частоты дискретизации
    fft_freqs = scipy.fft.rfftfreq(n_samples, 1 / sample_freq)
    samples_per_freq = len(fft_freqs) / (sample_freq / 2)
    for start, stop, action in artifact_intervals:
        artifact_channels = artifact_specs.get(action, None)
        if artifact_channels is None:
            continue

        # Get data from the artifact interval
        start -= 2 / sample_freq
        stop += 2 / sample_freq
        interval_mask = raw_df["time"].between(start, stop)
        interval_df = raw_df.loc[interval_mask]

        for chan, freqs in artifact_channels.items():
            for lo_freq, hi_freq in freqs:
                # # Discretize data and run FFT
                # normalized_tone = np.int16(
                #     ((interval_df[chan].values - ch_means[chan]) / ch_maxes[chan])
                #     * fft_discretization
                # )
                # fft_data = scipy.fft.rfft(normalized_tone)
                fft_data = scipy.fft.rfft(interval_df[chan].values)

                # Clean artifact frequencies from data
                min_target_idx = (
                    int(samples_per_freq * lo_freq) - 2 if lo_freq is not None else None
                )
                max_target_idx = (
                    int(samples_per_freq * hi_freq) + 2 if hi_freq is not None else None
                )
                fft_data[min_target_idx:max_target_idx] = zeroing_value

                # Reverse FFT
                signal_data = scipy.fft.irfft(fft_data)
                if len(signal_data) < raw_df.loc[interval_mask, chan].shape[0]:
                    signal_data = np.insert(signal_data, 0, zeroing_value)

                # signal_data = (
                #     signal_data / fft_discretization * ch_maxes[chan] + ch_means[chan]
                # )

                raw_df.loc[interval_mask, chan] = signal_data

    raw_df = raw_df.drop(columns=["time"])

    return raw_df


# def clean_with_fft(raw: mne.io.Raw) -> mne.io.Raw:
#     raw_df = raw.to_data_frame()
#     raw_df = raw_df.iloc[:-1, 1:]
#     ch_names = raw_df.columns.tolist()
#     ch_types = ["eeg"] * len(ch_names)
#     SAMPLE_RATE = raw.info["sfreq"]
#     N = raw_df.shape[0]

#     filtered_df = raw_df.copy()
#     for channel in ch_names:
#         chan = raw_df[channel].values
#         normalized_tone = np.int16(((chan - chan.mean()) / chan.max()) * 32767)
#         xf = scipy.fft.rfftfreq(N, 1 / SAMPLE_RATE)
#         yf = scipy.fft.rfft(normalized_tone)

#         # Максимальная частота составляет половину частоты дискретизации
#         points_per_freq = len(xf) / (SAMPLE_RATE / 2)

#         TARGET_MIN_FREQ = 0.1
#         TARGET_MAX_FREQ = 8
#         min_target_idx = int(points_per_freq * TARGET_MIN_FREQ)
#         max_target_idx = int(points_per_freq * TARGET_MAX_FREQ)

#         # Обнулим yf для индексов около целевой частоты
#         yf[min_target_idx - 2 : max_target_idx + 2] = 0

#         new_sig = scipy.fft.irfft(yf)
#         filtered_df[channel] = new_sig / 43623701.42716775

#     new_info = mne.create_info(ch_names, sfreq=SAMPLE_RATE, ch_types=ch_types)
#     filtered_raw = mne.io.RawArray(filtered_df.values.T, new_info)
#     # filtered_raw = raw.copy().filter(l_freq=8, h_freq=None)

#     return filtered_raw


def remove_artifacts(
    raw_eeg_df: pd.DataFrame, artifact_intervals: list[tuple[float, float, str]]
) -> pd.DataFrame:
    ALL_CHANNELS = ["O1", "T3", "Fp1", "Fp2", "T4", "O2"]
    ACTION_CHANNEL_MAP: dict[str, tuple[str]] = {
        "HEAD_LEFT": ("Fp1", "Fp2"),
        "STAND": ("O1", "T3", "Fp1", "Fp2", "T4", "O2"),
        "BLINK": ("O1", "O2"),
    }

    cleaned_df = raw_eeg_df.copy()
    del raw_eeg_df

    # cleaned_df.loc[cleaned_df["time"].between(artifact_intervals[0][0], artifact_intervals[0][1])]

    for start, stop, action in artifact_intervals:
        channels = ACTION_CHANNEL_MAP.get(action)
        if channels is None:
            channels = ALL_CHANNELS
        # for channel in channels:
        cleaned_df.loc[cleaned_df["time"].between(start, stop), channels] = np.nan

    return cleaned_df


def impute_missing_channel_data(eeg_df: pd.DataFrame) -> pd.DataFrame:
    # estimator = sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=0)
    # imputer = sklearn.impute.IterativeImputer(estimator, max_iter=10, random_state=0)

    # Uses bayesian ridge regression by default, not very good results by the look of them. But subsecond fast
    imputer = sklearn.impute.IterativeImputer(max_iter=10, random_state=0)
    imputed_values = imputer.fit_transform(eeg_df)

    imputed_df = pd.DataFrame(imputed_values, columns=eeg_df.columns)
    return imputed_df


def make_mne_raw(data: np.ndarray, info: mne.Info) -> mne.io.Raw:
    return mne.io.RawArray(data.T, info)


def clean_eeg(
    eeg_raw: mne.io.Raw,
    artifact_intervals: list[tuple[float, float, str]],
    method: str = "fft",
    artifact_specs: dict[
        str, dict[str, list]
    ] = eeg.default.DEFAULT_ACTION_ARTIFACT_FREQS,
    total_clean_freqs: list[list[float]] = [(0.1, 8)],
    total_clean_coef: float = 0.001,
) -> mne.io.Raw:
    """Clean EEG data from artifacts.

    Args:
        eeg_raw: Raw EEG data.
        artifact_intervals: Intervals of seconds with actions that are considered artifacts.
            Used in selective channel cleaning.
            Example: [(1.0, 1.5, "HEAD_LEFT"), (5.2, 6.4, "STAND"), (9.1, 9.4, "BLINK")].
        method: Method of cleaning the EEG data. "iterative" or "fft" are supported.
            "iterative" uses multiple imputation to fill in missing data.
            "fft" uses fast fourier transform to clean the data based on frequencies.
            Defaults to "fft".
        artifact_specs: Dictionary of artifact types and their respective channels
            and frequencies to clean.
            The frequency is a list of tuples (freq1, freq2) of inclusive frequency ranges.
                If the freq1 < freq2, all frequencies in the range are cleaned.
                If the freq1 == freq2, only that frequency is cleaned.
                If freq1 is None, all frequencies up to the freq2 are cleaned.
                If freq2 is None, all frequencies from the freq1 are cleaned.
                If both freq1 and freq2 are None, all frequencies are cleaned.
                Pass an empty list to clean nothing.
            Defaults to eeg.default.DEFAULT_ACTION_ARTIFACT_FREQS.
            Example: {
                "HEAD_LEFT": {"Fp1": [(0.1, 8)], "Fp2": [(0.1, 8)]},
                "STAND": {"O1": [(0.1, 8)], "T3": [(0.1, 8)]},
                "BLINK": {"O1": [(0.1, 8)], "O2": [(0.1, 8)]},
            }.
        total_clean_freqs: Frequencies to clean from all channels.
            Same rules are applied as for artifact_specs frequencies.
            Defaults to [(0.1, 8)].
            Example: [(0.1, 8), (50, 50)].
        total_clean_coef: Coefficient to apply to all channels for data with
            frequency matching those in total_clean_freqs.
            Defaults to 0.001.

    Raises:
        ValueError: Raised if an unknown method is passed (not "iterative" or "fft").

    Returns:
        mne.io.Raw: Cleaned EEG data.
    """
    # Selective channel cleaning based on detected artifacts
    if method == "iterative":
        raw_df = eeg_raw.to_data_frame()
        clean_df = remove_artifacts(raw_df, artifact_intervals)
        imputed_df = impute_missing_channel_data(clean_df)
        cleaned_eeg = imputed_df.drop(columns=["time"])
        new_info = eeg_raw.info
    elif method == "fft":
        cleaned_eeg = clean_with_fft(
            eeg_raw,
            artifact_intervals,
            artifact_specs,
            sample_freq=eeg_raw.info["sfreq"],
        )
        new_info = mne.create_info(
            eeg_raw.info["ch_names"],
            sfreq=eeg_raw.info["sfreq"],
            ch_types=eeg_raw.info.get_channel_types(),
        )
    else:
        raise ValueError("Unknown method")

    cleaned_eeg = make_mne_raw(cleaned_eeg.values, new_info)

    # Total cleaning of all channels

    return cleaned_eeg


if __name__ == "__main__":
    eeg_filepath = r"C:\Users\shata\Downloads\YaDiskDL\Компьютер DESKTOP-CJ9FOG3 (2)\Калининград 19.10.2022\Final\Cohort1\table1\Round1\P1\2022.10.19-14.11.55.144.edf"

    # Example of video artifact detection output - intervals of seconds with actions
    artifact_intervals = [
        (1.0, 1.5, "HEAD_LEFT"),
        (5.2, 6.4, "STAND"),
        (9.1, 9.4, "BLINK"),
    ]

    eeg_raw = clean_eeg(eeg_filepath, artifact_intervals)
