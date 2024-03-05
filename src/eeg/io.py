import mne


def read_edf_file(file_path: str) -> mne.io.Raw:
    return mne.io.read_raw_edf(file_path, preload=True)


def write_edf_file(raw: mne.io.Raw, file_path: str) -> None:
    # # Must have extension .fif or .fif.gz
    # raw.save(file_path, overwrite=True)

    # edf export requires EDFlib-Python to be installed
    mne.export.export_raw(file_path, raw, fmt="edf", overwrite=True)
