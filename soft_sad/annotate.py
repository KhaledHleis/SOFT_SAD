import json
import numpy as np
import soundfile as sf
from scipy.signal import medfilt
from pathlib import Path
import os, shutil
import argparse
import yaml
from tqdm import tqdm
import pandas as pd


# region Helper functions
def generate_vad_annotation(
    utt_id,
    audio_path,
    positive_label="speech",
    category="audio_books",
    frame_ms=30,
    hop_ms=10,
    energy_threshold_ratio=0.08,
    min_activity_ms=150,
    min_silence_ms=200,
    smoothing_kernel=5,
):
    """
    Generate JSON annotation using energy-based Voice Activity Detection (VAD)
    without machine learning.

    Parameters
    ----------
    audio_path : str
        Path to audio file.

    positive_label : str
        Label for detected activity (e.g. "speech", "nonspeech").

    category : str
        Category associated with positive activity.

    frame_ms : int
        Frame size in milliseconds.

    hop_ms : int
        Hop size in milliseconds.

    energy_threshold_ratio : float
        Relative threshold compared to max energy.

    min_activity_ms : int
        Minimum duration for positive activity.

    min_silence_ms : int
        Minimum silence duration.

    smoothing_kernel : int
        Median filter kernel size.

    Returns
    -------
    dict
        Annotation dictionary.
    """

    # ---------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------
    signal, sr = sf.read(audio_path)

    # Convert stereo to mono
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    duration = len(signal) / sr

    # ---------------------------------------------------------
    # Framing
    # ---------------------------------------------------------
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    energies = []
    timestamps = []

    for start in range(0, len(signal) - frame_len, hop_len):
        frame = signal[start : start + frame_len]

        # RMS energy
        rms = np.sqrt(np.mean(frame**2) + 1e-10)

        energies.append(rms)
        timestamps.append(start / sr)

    energies = np.array(energies)

    # ---------------------------------------------------------
    # Energy thresholding
    # ---------------------------------------------------------
    threshold = energy_threshold_ratio * np.max(energies)

    vad = (energies > threshold).astype(np.int32)

    # ---------------------------------------------------------
    # Median filtering for smoothing
    # ---------------------------------------------------------
    if smoothing_kernel > 1:
        vad = medfilt(vad, kernel_size=smoothing_kernel)

    # ---------------------------------------------------------
    # Remove short activity bursts
    # ---------------------------------------------------------
    min_activity_frames = int(min_activity_ms / hop_ms)

    start = None
    for i in range(len(vad)):
        if vad[i] == 1 and start is None:
            start = i

        elif vad[i] == 0 and start is not None:
            if (i - start) < min_activity_frames:
                vad[start:i] = 0
            start = None

    # ---------------------------------------------------------
    # Fill short silence gaps
    # ---------------------------------------------------------
    min_silence_frames = int(min_silence_ms / hop_ms)

    start = None
    for i in range(len(vad)):
        if vad[i] == 0 and start is None:
            start = i

        elif vad[i] == 1 and start is not None:
            if (i - start) < min_silence_frames:
                vad[start:i] = 1
            start = None

    # ---------------------------------------------------------
    # Convert frame decisions to intervals
    # ---------------------------------------------------------
    intervals = []

    current_state = vad[0]
    segment_start = timestamps[0]

    for i in range(1, len(vad)):
        if vad[i] != current_state:

            segment_end = timestamps[i]

            if current_state == 1:
                intervals.append(
                    {
                        "start": round(segment_start, 4),
                        "end": round(segment_end, 4),
                        "label": positive_label,
                        "category": category,
                    }
                )
            else:
                intervals.append(
                    {
                        "start": round(segment_start, 4),
                        "end": round(segment_end, 4),
                        "label": "silence",
                    }
                )

            segment_start = timestamps[i]
            current_state = vad[i]

    # Last segment
    final_end = duration

    if current_state == 1:
        intervals.append(
            {
                "start": round(segment_start, 4),
                "end": round(final_end, 4),
                "label": positive_label,
                "category": category,
            }
        )
    else:
        intervals.append(
            {
                "start": round(segment_start, 4),
                "end": round(final_end, 4),
                "label": "silence",
            }
        )

    # ---------------------------------------------------------
    # Final annotation
    # ---------------------------------------------------------
    annotation = {
        "utt_id": utt_id,
        "duration": round(duration, 4),
        "sample_rate": sr,
        "intervals": intervals,
    }

    return annotation


# endregion


# ============================================================
# main script
# ============================================================
def main():
    # arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_root = Path(cfg["data"]["root"])
    wav_dir = os.path.join(data_root, "wavs")
    csv_path = os.path.join(data_root, "ALL.csv")
    annotations_dir = os.path.join(data_root, "annotations")
    # for every wav file in the dataset construct and save its json file
    # get all files from the csv
    df = pd.read_csv(csv_path)
    wave_files_stem = df["utt_id"]
    classes = df["class"]
    categories = df["category"]

    if not os.path.exists(annotations_dir):
        os.mkdir(annotations_dir)
    else:
        shutil.rmtree(annotations_dir)
        os.mkdir(annotations_dir)

    for file_stem, posclass, category in tqdm(
        zip(wave_files_stem, classes, categories),
        total=len(df),
        desc="Generating annotations",
        unit="file",
    ):
        file_path = os.path.join(wav_dir, file_stem + ".wav")
        with open(os.path.join(annotations_dir, file_stem + ".json"), "w") as f:
            json.dump(
                generate_vad_annotation(
                    file_stem,
                    audio_path=file_path,
                    positive_label=posclass,
                    category=category,
                ),
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
