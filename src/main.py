import datetime
from typing import Any

import eeg.filtering
import eeg.io

import video.mediapipe1


if __name__ == "__main__":
    # video_file_path = r"C:\path\to\video.mp4"
    # eeg_filepath = r"C:\path\to\eeg.edf"
    video_file_path = r"C:\Projects\Panga\eeg-artifact-filtering\data\ANMR002.mp4"
    eeg_filepath = (
        r"C:\Projects\Panga\eeg-artifact-filtering\data\2024.01.17-13.45.38.269.edf"
    )

    start_dttm = datetime.datetime.now()

    # video_stream_widget = video.mediapipe1.VideoStreamWidget(
    #     src=video_file_path, live=False, save_data=True
    # )
    # video_stream_widget.run()
    # artifact_data = video_stream_widget.final_data

    # end_dttm = datetime.datetime.now()
    # print(f"Video action recognition finished in {end_dttm - start_dttm} seconds")
    # start_dttm = datetime.datetime.now()

    # # Make timestamps from video into intervals
    # artifact_intervals = []
    # for item in artifact_data:
    #     ts = item["ts"]
    #     start = ts - 0.25
    #     stop = ts + 0.25
    #     actions: dict = item["actions"]
    #     likeliest_action_name = max(actions, key=actions.get)
    #     artifact_intervals.append((start, stop, likeliest_action_name))

    # # Example of video artifact detection output - intervals of seconds with actions
    artifact_intervals = [
        (1.0, 1.5, "HEAD_LEFT"),
        (5.2, 6.4, "STAND"),
        (9.1, 9.4, "BLINK"),
    ]

    raw_eeg = eeg.io.read_edf_file(eeg_filepath)
    processed_eeg = eeg.filtering.clean_eeg(raw_eeg, artifact_intervals, method="fft")
    eeg.io.write_edf_file(processed_eeg, "cleaned_eeg.edf")

    end_dttm = datetime.datetime.now()
    print(f"EEG artifact cleaning finished in {end_dttm - start_dttm} seconds")

    ## Visualization
    rw = eeg.io.read_edf_file(eeg_filepath)
    cln = eeg.io.read_edf_file("cleaned_eeg.edf")
    rw.crop(tmin=0.5, tmax=10).plot(
        n_channels=6,
        # scalings=0.00005,
        scalings="auto",
        title="Before artifact removal",
    )
    cln.crop(tmin=0.5, tmax=10).plot(
        n_channels=6,
        # scalings=0.00005,
        scalings="auto",
        title="After artifact removal",
    )

    print()

    # fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    # ax1 = axes[0]
    # raw_mod = raw.copy().crop(tmax=5).pick_channels(["T3"])
    # filtered_raw_mod = filtered_raw.copy().crop(tmax=5).pick_channels(["T3"])
    # ax1.plot(raw_mod.times, raw_mod.get_data().T)
    # ax1.plot(filtered_raw_mod.times, filtered_raw_mod.get_data().T)
    # ax1.set_title("No band filter")
