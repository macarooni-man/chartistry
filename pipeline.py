# pipeline.py

import os
import numpy as np
from typing import List
from chart_parser import Chart
from ml_transcriber import load_model, transcribe_drums, tick_to_time, _build_timing_map, Transcription, Note
from sklearn.metrics import precision_recall_fscore_support

# 1) configuration
MODEL_PATH = "drum_detector.pth"
THRESHOLDS = {c: 0.3 for c in ["kick", "snare", "tom1", "tom2", "tom3", "cymbal1", "cymbal2", "cymbal3"]}
TARGET_F1 = 0.90

# 2) load GT Transcription from Chart
def load_ground_truth(chart_path: str) -> Transcription:
    chart = Chart(chart_path)
    # build timing_map for tick→time
    timing_map = _build_timing_map(chart)

    groups = {}
    for note in chart.tracks["ExpertDrums"].notes:
        t = tick_to_time(note.tick, timing_map, chart.resolution)
        groups.setdefault(note.tick, []).append(
            Note(time=t, tick=note.tick, category=note.category)
        )
    return Transcription(type="ExpertDrums", notes=[groups[t] for t in sorted(groups)])

# 3) turn Transcription → frame labels
def transcription_to_frame_labels(
    trans: Transcription,
    sr: int,
    hop_length: int,
    model_classes: List[str]
):
    """
    Build (T, K) binary array: 1 if label present in frame.
    """
    # find all hit times
    all_times = []
    all_labels = []
    for group in trans.notes:
        for n in group:
            all_times.append(n.time)
            all_labels.append(n.category)
    # determine total frames
    T = int(np.ceil(max(all_times) * sr / hop_length)) + 1
    K = len(model_classes)
    lbl_map = {c: i for i, c in enumerate(model_classes)}
    Y = np.zeros((T, K), dtype=int)
    # mark frames within ±1 hop of each hit
    for t, cat in zip(all_times, all_labels):
        f = int(round(t * sr / hop_length))
        if cat in lbl_map and 0 <= f < T:
            Y[f, lbl_map[cat]] = 1
    return Y

# 4) evaluation
def evaluate(gt: Transcription, pred: Transcription, sr: int, hop_length: int, classes):
    Y_true = transcription_to_frame_labels(gt, sr, hop_length, classes).reshape(-1, len(classes))
    # we need predicted frame labels too
    # (re‐run transcribe with the same frame hop → P, then threshold → Y_pred)
    # For brevity, assume `pred` was built similarly via `transcribe_drums`
    Y_pred = transcription_to_frame_labels(pred, sr, hop_length, classes)
    p,r,f1,_ = precision_recall_fscore_support(
        Y_true, Y_pred, average="macro", zero_division=0
    )
    return f1

# 5) main loop
def main(root_folder: str):
    model = load_model(MODEL_PATH)
    classes = model.classes
    all_f1 = []

    for song in os.listdir(root_folder):
        fol = os.path.join(root_folder, song)
        chart_file = os.path.join(fol, "notes.chart")
        audio_file = os.path.join(fol, "drums.wav")
        if not os.path.isfile(chart_file) or not os.path.isfile(audio_file):
            continue

        gt   = load_ground_truth(chart_file)
        pred = transcribe_drums(gt_chart:=Chart(chart_file),
                                audio_file,
                                model,
                                THRESHOLDS)
        f1   = evaluate(gt, pred, sr=44100, hop_length=512, classes=classes)
        print(f"{song:30s}  F1={f1:.3f}")
        all_f1.append(f1)

    avg = sum(all_f1)/len(all_f1) if all_f1 else 0.0
    print(f"\nAverage F1: {avg:.3f}")
    if avg < TARGET_F1:
        print("Below target—adjust thresholds or retrain and re-run.")
    else:
        print("✅ Target reached!")

if __name__=="__main__":
    main("path/to/your/song_folders")
