"""Rebuild Unit 4's small browser CSV from the original Sleep-EDF recordings.

Usage: python prepare_sleep_data.py --source .local/unit4-sleep-pilot
Extra preparation dependency: pip install pyedflib
The website itself needs only the bundled CSV, numpy, pandas and matplotlib.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30)]
STAGES = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
          "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}
FEATURES = []
for channel in ["eeg_fpz", "eeg_pz", "eog"]:
    for band in ["delta", "theta", "alpha", "sigma", "beta"]:
        FEATURES.append(f"{channel}_{band}")


def prepare(source):
    """Verify the original files, then extract features without using stage labels."""
    # Pin the original file identities after the first export; a rebuild may not
    # silently replace the published dataset with different source recordings.
    provenance_path = ROOT / "website/data/sleep_edf_provenance.json"
    if provenance_path.exists():
        manifest = json.loads(provenance_path.read_text())["source_files"]
    else:
        manifest = json.loads((source / "download_manifest.json").read_text())
    assert len(manifest) == 16
    for entry in manifest:
        digest = hashlib.sha256((source / entry["file"]).read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"Source checksum mismatch: {entry['file']}"

    frames = []
    for subject in range(8):
        prefix = f"SC40{subject}1"
        signal_path = next(source.glob(prefix + "*-PSG.edf"))
        annotation_path = next(source.glob(prefix + "*-Hypnogram.edf"))
        with pyedflib.EdfReader(str(signal_path)) as signals, pyedflib.EdfReader(str(annotation_path)) as annotations:
            assert signals.getStartdatetime() == annotations.getStartdatetime()
            epochs = int(signals.file_duration // 30)
            labels = np.full(epochs, -1, dtype=int)
            onset, duration, description = annotations.readAnnotations()
            for start, length, text in zip(onset, duration, description):
                assert np.isclose(start / 30, round(start / 30))
                assert np.isclose(length / 30, round(length / 30))
                lo = max(0, round(start / 30))
                hi = min(epochs, round((start + length) / 30))
                if text in STAGES:
                    assert np.all(labels[lo:hi] == -1)
                    labels[lo:hi] = STAGES[text]
            features = []
            for channel in CHANNELS:
                index = signals.getSignalLabels().index(channel)
                assert signals.getSampleFrequency(index) == 100
                samples = signals.readSignal(index)[:epochs * 3000].reshape(epochs, 3000)
                assert np.isfinite(samples).all()
                frequency, power = welch(samples, fs=100, nperseg=256, axis=1)
                for low, high in BANDS:
                    mask = (frequency >= low) & (frequency < high)
                    bandpower = power[:, mask].sum(axis=1) * (frequency[1] - frequency[0])
                    features.append(np.log(np.maximum(bandpower, 1e-12)))
        frame = pd.DataFrame(np.column_stack(features), columns=FEATURES)
        assert np.isfinite(frame.values).all()
        frame.insert(0, "stage", labels)
        frame.insert(0, "seconds", np.arange(epochs) * 30)
        frame.insert(0, "epoch", np.arange(epochs))
        frame.insert(0, "subject", f"SC40{subject}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    assert len(data) == 21948 and (data.stage >= 0).sum() == 21947
    csv_path = ROOT / "website/data/sleep_edf.csv"
    data.to_csv(csv_path, index=False, float_format="%.12g")
    provenance = {
        "name": "Sleep-EDF Expanded 1.0.0 — eight-person teaching subset",
        "source": "https://physionet.org/content/sleep-edfx/1.0.0/",
        "license": "Open Data Commons Attribution License v1.0",
        "credit": "Bob Kemp and colleagues; PhysioNet. See the source page for required citations.",
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "rows": len(data), "scored_rows": int((data.stage >= 0).sum()),
        "subjects": list(data.subject.unique()), "epoch_seconds": 30,
        "stages": {"-1": "Movement/unscored", "0": "Wake", "1": "N1", "2": "N2", "3": "N3 (original stages 3 and 4)", "4": "REM"},
        "features": FEATURES,
        "preparation": "First night per person; nonoverlapping 30-second epochs. Welch power spectrum at 100 Hz, nperseg=256, default Hann window, constant detrending and 50% internal Welch overlap. Sum density bins in [0.5,4), [4,8), [8,12), [12,16), [16,30) Hz, multiply by frequency spacing, floor at 1e-12 and take natural log. Source signals are in microvolts. No fitting or label-dependent feature selection.",
        "source_files": manifest,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {csv_path.name}: {len(data):,} epochs, 8 people, 15 signal features")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".local/unit4-sleep-pilot")
    prepare(parser.parse_args().source)
