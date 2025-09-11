import os, sys, glob

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.curdir, 'dataset')

AUDIO_NAMES = ["drums"]
AUDIO_EXTS  = [".wav", ".ogg", ".mp3", ".flac", ".m4a"]
CHART_NAMES = ["notes"]
CHART_EXTS  = [".chart"]

def pick_file(dirpath, bases, exts):
    # case-insensitive match like drums.wav / Drums.WAV etc.
    for b in bases:
        for e in exts:
            hits = glob.glob(os.path.join(dirpath, f"{b}{e}"), recursive=False)
            hits += glob.glob(os.path.join(dirpath, f"{b.upper()}{e}"))
            hits += glob.glob(os.path.join(dirpath, f"{b.capitalize()}{e}"))
            if hits:
                return hits[0]
    return None

dirs = [os.path.join(ROOT, d) for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
dirs.sort()

found, missing = [], []
for d in dirs:
    chart = None
    # accept any *.chart first (some packs name them arbitrarily)
    any_chart = glob.glob(os.path.join(d, "*.chart"))
    if any_chart:
        chart = any_chart[0]
    else:
        chart = pick_file(d, CHART_NAMES, CHART_EXTS)

    audio = None
    # prefer drums.*; else mix.*; else audio.*
    for group in [ ["drums"], ["mix"], ["audio"] ]:
        audio = pick_file(d, group, AUDIO_EXTS)
        if audio: break

    if chart and audio:
        found.append((d, chart, audio))
    else:
        reason = []
        if not chart: reason.append("no .chart file found (looked for *.chart, notes.chart, chart.chart)")
        if not audio: reason.append("no audio (looked for drums/mix/audio with .wav/.ogg/.mp3/.flac/.m4a)")
        missing.append((d, "; ".join(reason)))

print(f"Found {len(found)} folders, Missing {len(missing)}")
if missing:
    print("\nMissing list:")
    for d, why in missing[:20]:
        print(f" - {os.path.basename(d)}: {why}")
    if len(missing) > 20:
        print(f"... and {len(missing)-20} more")
