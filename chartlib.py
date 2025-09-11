from __future__ import annotations

from typing import Dict, List, Tuple, Iterable, Optional
from dataclasses import dataclass, field
import bisect
import re
import io



# ------------------------------------------- Legend ----------------------------------------------------
# - tick: integer time unit used inside .chart files; musical time depends on Resolution (ticks/quarter).
# - ms: milliseconds; real-clock time.
# - Resolution (TPQN): ticks per quarter note (from [Song].Resolution).
# - SyncTrack: defines tempo changes over time; "B <us_per_qn>" gives microseconds per quarter note.
# - TempoMap: piecewise-linear mapping between ticks and milliseconds built from SyncTrack.
# - Pro Drums cymbal flags: additional "N 66/67/68" events at the same tick as Y/B/G to mark cymbals.



# ----------------------------------------- Data Objects ------------------------------------------------

# Currently, all charts in the dataset are set to 960
dataset_chart_res = 960

@dataclass
class Note:
    tick: int                      # Tick position in the chart
    lane: int                      # Note lane - 0: kick, 1: red, 2: yellow, 3: blue, 4: green
    length: int = 0                # Sustain length in ticks (usually 0 for drums)
    dyn: int = 1                   # Dynamics - 0: ghost, 1: normal, 2: accent
    time_ms: Optional[int] = None  # Filled later via TempoMap.attach_times()

@dataclass
class Special:
    tick: int
    type: int
    length: int

@dataclass
class TrackEvent:
    tick: int
    text: str

@dataclass
class SyncEvent:
    tick: int
    kind: str          # 'B' (tempo), 'TS' (time signature), 'A' (anchor)
    values: List[int]  # B: [us_per_qn]; TS: [num, den? or just num in some files]; A: [0]

@dataclass
class Track:
    notes: List[Note] = field(default_factory=list)
    specials: List[Special] = field(default_factory=list)
    events: List[TrackEvent] = field(default_factory=list)



# ----------------------------------------- Parser / Model ----------------------------------------

class Chart:
    """
    Load and manipulate a .chart file.
    Exposes:
      - metadata: Dict[str, str|int|float]
      - resolution: int  (ticks per quarter note)
      - sync: List[SyncEvent]
      - events: List[TrackEvent] (global [Events] section)
      - tracks: Dict[str, Track] keyed by section name (e.g., 'ExpertDrums')

    Adds:
      - tempo: internal TempoMap instance
      - tick_to_ms()/ms_to_tick(): convenience that defer to tempo
      - build_grid_ticks(subdivision=12, track_hint='ExpertDrums'): 48ths by default
    """
    _section_re = re.compile(
        r'^\[(?P<section>[^\]]+)\]\s*\{\s*(?P<body>.*?)\s*\}',
        re.MULTILINE | re.DOTALL
    )



    # ----------------------- Nested TempoMap (internal to Chart) -----------------------

    class TempoMap:
        """
        Piecewise-linear mapping between ticks and milliseconds.

        Built from:
          - resolution (ticks per quarter note)
          - sync events with kind 'B' (tempo): value is 'us_per_qn' (microseconds per quarter note)

        Stored as segments: start_tick[i], start_ms[i], ms_per_tick[i]
        ms_per_tick = (us_per_qn / 1000.0) / resolution
        """
        def __init__(self, resolution: int, sync_events: List[SyncEvent]):
            self.resolution = resolution
            tempos: List[Tuple[int, int]] = []  # (tick, us_per_qn)
            for event in sorted(sync_events, key=lambda e: e.tick):

                if event.kind == "B" and event.values:
                    tempos.append((event.tick, int(event.values[0])))

            if not tempos or tempos[0][0] != 0:
                tempos.insert(0, (0, 500_000))  # default 120 BPM

            self.start_tick: List[int] = []
            self.start_ms: List[float] = []
            self.ms_per_tick: List[float] = []

            acc_ms = 0.0
            last_tick = tempos[0][0]
            last_us_qn = tempos[0][1]
            last_mpt = (last_us_qn / 1000.0) / self.resolution

            self.start_tick.append(last_tick)
            self.start_ms.append(acc_ms)
            self.ms_per_tick.append(last_mpt)

            for i in range(1, len(tempos)):
                tick_i, us_qn_i = tempos[i]
                delta_ticks = tick_i - last_tick
                acc_ms += delta_ticks * last_mpt

                self.start_tick.append(tick_i)
                self.start_ms.append(acc_ms)
                last_tick = tick_i
                last_us_qn = us_qn_i
                last_mpt = (last_us_qn / 1000.0) / self.resolution
                self.ms_per_tick.append(last_mpt)

            self._start_ms_for_bisect = list(self.start_ms)

        def _idx_for_tick(self, tick: int) -> int:
            i = bisect.bisect_right(self.start_tick, tick) - 1
            return max(0, i)

        def _idx_for_ms(self, ms: float) -> int:
            i = bisect.bisect_right(self._start_ms_for_bisect, ms) - 1
            return max(0, i)

        def tick_to_ms(self, tick: int) -> float:
            i = self._idx_for_tick(tick)
            dticks = tick - self.start_tick[i]
            return self.start_ms[i] + dticks * self.ms_per_tick[i]

        def ms_to_tick(self, ms: float) -> float:
            i = self._idx_for_ms(ms)
            dms = ms - self.start_ms[i]
            return self.start_tick[i] + dms / self.ms_per_tick[i]

    def __init__(self, path: str):
        self.path = path
        self.metadata: Dict[str, object] = {}
        self.resolution: int = dataset_chart_res
        self.sync: List[SyncEvent] = []
        self.events: List[TrackEvent] = []
        self.tracks: Dict[str, Track] = {}
        self.tempo: Optional[Chart.TempoMap] = None

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            self._raw = f.read().lstrip("\ufeff")

        self._parse(self._raw)

        # Build internal tempo map once parsing is done
        self.tempo = self.TempoMap(self.resolution, self.sync)



    # --------------- parsing sections ---------------

    def _parse(self, raw: str) -> None:
        for match in self._section_re.finditer(raw):
            section = match.group("section").strip()
            body = match.group("body").strip()
            handler = getattr(self, f"_parse_{section.lower()}", self._parse_track)
            handler(section, body)


    def _parse_song(self, section: str, body: str) -> None:
        for line in body.splitlines():

            if "=" not in line:
                continue

            k, v = map(str.strip, line.split("=", 1))
            v = v.strip()

            if v.startswith('"') and v.endswith('"'): formatted_value: object = v[1:-1]
            else:

                try: formatted_value = int(v)
                except ValueError:

                    try: formatted_value = float(v)
                    except ValueError:
                        formatted_value = v

            self.metadata[k] = formatted_value

            if k.lower() == "resolution":
                self.resolution = int(formatted_value)


    def _parse_synctrack(self, section: str, body: str) -> None:

        for line in body.splitlines():
            parts = line.split()

            if len(parts) < 4 or parts[1] != "=":
                continue

            tick = int(parts[0])
            kind = parts[2]
            vals = [int(x) for x in parts[3:]]
            self.sync.append(SyncEvent(tick, kind, vals))

        self.sync.sort(key=lambda e: e.tick)


    def _parse_events(self, section: str, body: str) -> None:

        for line in body.splitlines():

            if "=" not in line:
                continue

            lhs, rhs = line.split("=", 1)
            tick = int(lhs.strip())
            parts = rhs.strip().split(None, 1)

            if not parts or parts[0] != "E" or len(parts) < 2:
                continue

            text = parts[1].strip().strip('"')
            self.events.append(TrackEvent(tick, text))

        self.events.sort(key=lambda e: e.tick)


    def _parse_track(self, section: str, body: str) -> None:

        track = Track()

        for line in body.splitlines():
            parts = line.split()

            if len(parts) < 4:
                continue

            if parts[1] != "=":
                continue

            tick = int(parts[0])
            key = parts[2]

            if key == "N" and len(parts) >= 5:
                lane = int(parts[3])
                length = int(parts[4])
                track.notes.append(Note(tick=tick, lane=lane, length=length))

            elif key == "S" and len(parts) >= 5:
                typ = int(parts[3])
                length = int(parts[4])
                track.specials.append(Special(tick=tick, type=typ, length=length))

            elif key == "E" and len(parts) >= 4:
                text = " ".join(parts[3:])
                text = text.strip().strip('"')
                track.events.append(TrackEvent(tick=tick, text=text))

        self.tracks[section] = track



    # --------------- helpers ---------------

    def get_track(self, name: str) -> Track:
        return self.tracks.setdefault(name, Track())


    def iter_notes(self, track: str) -> Iterable[Note]:
        return self.get_track(track).notes


    # Attach real time to notes/events using the internal TempoMap
    def attach_times(self) -> None:

        if not self.tempo:
            self.tempo = Chart.TempoMap(self.resolution, self.sync)

        for track in self.tracks.values():
            for note in track.notes:
                note.time_ms = int(round(self.tempo.tick_to_ms(note.tick)))


    # Convenience: expose tick to ms via the internal tempo
    def tick_to_ms(self, tick: int) -> float:

        if not self.tempo:
            self.tempo = Chart.TempoMap(self.resolution, self.sync)

        return self.tempo.tick_to_ms(tick)


    def ms_to_tick(self, ms: float) -> float:

        if not self.tempo:
            self.tempo = Chart.TempoMap(self.resolution, self.sync)

        return self.tempo.ms_to_tick(ms)


    # Quantize to a musical grid in ticks, subdivision is 48th notes by default (32nd note triplets)
    def build_grid_ticks(self, subdivision: int = 48, track_hint: str = "ExpertDrums") -> List[int]:
        notes_per_quarter = round(subdivision / 4)
        step = self.resolution // notes_per_quarter

        if step <= 0: raise ValueError(f"Subdivision must be a positive divisor of resolution ({self.resolution})")

        # Determine span: prefer hinted track if present
        if track_hint in self.tracks and self.tracks[track_hint].notes:
            notes = self.tracks[track_hint].notes
            min_tick = min(n.tick for n in notes)
            max_tick = max(n.tick for n in notes)

        else:
            min_tick, max_tick = 0, 0
            for track in self.tracks.values():
                if track.notes:
                    tmin = min(note.tick for note in track.notes)
                    tmax = max(note.tick for note in track.notes)
                    min_tick = tmin if min_tick == 0 else min(min_tick, tmin)
                    max_tick = max(max_tick, tmax)

        # Pad a quarter on both ends
        start = max(0, (min_tick // step) * step)
        end   = ((max_tick + self.resolution) // step) * step
        return list(range(start, end + 1, step))


    # Serialize back to '.chart' format
    def to_chart_text(self) -> str:
        out = io.StringIO()
        print("[Song]\n{", file=out)
        for k, v in self.metadata.items():
            if isinstance(v, str):
                print(f'  {k} = "{v}"', file=out)
            else:
                print(f"  {k} = {v}", file=out)
        print("}", file=out)

        print("[SyncTrack]\n{", file=out)
        for event in sorted(self.sync, key=lambda e: e.tick):
            vals = " ".join(str(x) for x in event.values)
            print(f"  {event.tick} = {event.kind} {vals}", file=out)
        print("}", file=out)

        if self.events:
            print("[Events]\n{", file=out)
            for event in sorted(self.events, key=lambda e: e.tick):
                print(f'  {event.tick} = E "{event.text}"', file=out)
            print("}", file=out)

        for name, track in self.tracks.items():
            print(f"[{name}]\n{{", file=out)
            for event in sorted(track.events, key=lambda e: e.tick):
                print(f'  {event.tick} = E "{event.text}"', file=out)
            for special in sorted(track.specials, key=lambda s: s.tick):
                print(f"  {special.tick} = S {special.type} {special.length}", file=out)
            for note in sorted(track.notes, key=lambda n: (n.tick, n.lane)):
                print(f"  {note.tick} = N {note.lane} {note.length}", file=out)
            print("}", file=out)

        return out.getvalue()


    # Write the chart back to a file
    def write(self, path: Optional[str] = None) -> None:
        path = path or self.path
        with open(path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(self.to_chart_text())



# ------------------------- Pro-Drums Helpers (cymbal flags, grouping) ---------------------------

cymbal_lane_ids = {2: 66, 3: 67, 4: 68}  # Y/B/G cymbal marker lanes

def group_notes_by_tick(notes: Iterable[Note]) -> Dict[int, List[Note]]:
    by_tick: Dict[int, List[Note]] = {}

    for note in notes:
        by_tick.setdefault(note.tick, []).append(note)

    for tick in by_tick:
        by_tick[tick].sort(key=lambda n: n.lane)

    return by_tick


# Derive a per-tick view directly from notes into a simple dict
# > keys: kick, red, yellow, blue, green, cymbal_y, cymbal_b, cymbal_g
def normalize_pro_drums_tick(notes_at_tick: List[Note]) -> Dict[str, bool]:
    lanes = set(n.lane for n in notes_at_tick)
    return {
        "kick":     0 in lanes,
        "red":      1 in lanes,
        "yellow":   2 in lanes,
        "blue":     3 in lanes,
        "green":    4 in lanes,
        "cymbal_y": 66 in lanes,
        "cymbal_b": 67 in lanes,
        "cymbal_g": 68 in lanes,
    }

def add_pro_drums_note(track: Track, tick: int, pad_lane: int, is_cymbal: bool = False) -> None:

    track.notes.append(Note(tick=tick, lane=pad_lane, length=0))

    if is_cymbal and pad_lane in cymbal_lane_ids:
        track.notes.append(Note(tick=tick, lane=cymbal_lane_ids[pad_lane], length=0))



# ----------------------------------------- Convenience API --------------------------------------

def load_chart(path: str) -> Chart:
    return Chart(path)

def attach_times_inplace(chart: Chart) -> None:
    chart.attach_times()

def write_chart(chart: Chart, path: Optional[str] = None) -> None:
    chart.write(path)
