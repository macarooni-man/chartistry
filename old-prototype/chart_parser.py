from dataclasses import dataclass, field
from typing import Dict, List, Union
import re



@dataclass
class Note:
    tick: int
    fret: int
    length: int

@dataclass
class Special:
    tick: int
    type: int
    length: int

@dataclass
class Track:
    notes: List[Note] = field(default_factory=list)
    specials: List[Special] = field(default_factory=list)
    events: List[str] = field(default_factory=list)

@dataclass
class SyncEvent:
    tick: int
    type: str   # 'B', 'TS', or 'A'
    values: List[int]

@dataclass
class TextEvent:
    tick: int
    text: str

class Chart:
    """
    Load a .chart file into:
      - metadata: Dict[str, Union[str,int]]
      - resolution: int
      - sync: List[SyncEvent]
      - events: List[TextEvent]
      - tracks: Dict[str, Track]
    """
    _SECTION_RE = re.compile(
        r'^\[(?P<section>[^\]]+)\]\s*\{\s*(?P<body>.*?)\s*\}',
        re.MULTILINE | re.DOTALL
    )

    def __init__(self, path: str):
        # core data
        self.metadata: Dict[str, Union[str,int]] = {}
        self.resolution: int = 0
        self.sync: List[SyncEvent] = []
        self.events: List[TextEvent] = []
        self.tracks: Dict[str, Track] = {}

        # read & parse
        self._file = path
        with open(self._file, 'r', encoding='utf-8', errors='ignore') as f:
            self._raw = f.read()
            self._raw = self._raw.lstrip('\ufeff')  # strip BOM
            self._parse(self._raw)

    def _parse(self, raw: str):
        for m in self._SECTION_RE.finditer(raw):
            sec = m.group('section').strip()
            body = m.group('body').strip()
            handler = getattr(self, f'_parse_{sec.lower()}', self._parse_track)
            handler(sec, body)

    def _parse_song(self, sec: str, body: str):
        for line in body.splitlines():
            if '=' not in line:
                continue
            key, val = map(str.strip, line.split('=', 1))
            val = val.strip().strip('"')
            # try int, then float, otherwise leave as string
            if val.isdigit():
                v: Union[int,str] = int(val)
            else:
                try:
                    v = float(val)
                except ValueError:
                    v = val
            self.metadata[key] = v
            if key.lower() == 'resolution':
                self.resolution = int(v)

    def _parse_synctrack(self, sec: str, body: str):
        for line in body.splitlines():
            parts = line.split()
            if len(parts) < 4 or parts[1] != '=':
                continue
            tick = int(parts[0])
            typ = parts[2]
            vals = [int(x) for x in parts[3:]]
            self.sync.append(SyncEvent(tick, typ, vals))

    def _parse_events(self, sec: str, body: str):
        for line in body.splitlines():
            if '=' not in line:
                continue
            lhs, rhs = line.split('=', 1)
            tick = int(lhs.strip())
            # expect something like: 0 = E "section Intro"
            parts = rhs.strip().split(None, 1)
            if parts[0] != 'E' or len(parts) < 2:
                continue
            txt = parts[1].strip().strip('"')
            self.events.append(TextEvent(tick, txt))

    def _parse_track(self, sec: str, body: str):
        tr = Track()
        for line in body.splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[1] != '=':
                continue
            tick = int(parts[0])
            key = parts[2]
            if key == 'N':
                fret = int(parts[3])
                length = int(parts[4])
                tr.notes.append(Note(tick, fret, length))
            elif key == 'S':
                typ = int(parts[3])
                length = int(parts[4])
                tr.specials.append(Special(tick, typ, length))
            elif key == 'E':
                ev = parts[3].strip().strip('"')
                tr.events.append(ev)
        self.tracks[sec] = tr

    def add_chart(self, ts: "Transcription"):

        # Currently, only works for 4-lane pro drums
        if ts.type == 'ExpertDrums':
            data = f"[{ts.type}]\n{{"

            def note_parse(n: "Note"):
                print(n)

                if n.category == "kick":
                    return f"  {n.tick} = N 0 0"

                elif n.category == "snare":
                    return f"  {n.tick} = N 1 0"

                elif n.category == "tom1":
                    return f"  {n.tick} = N 2 0"

                elif n.category == "tom2":
                    return f"  {n.tick} = N 3 0"

                elif n.category == "tom3":
                    return f"  {n.tick} = N 4 0"

                elif n.category == "cymbal1":
                    return f"  {n.tick} = N 2 0\n  {n.tick} = N 66 0"

                elif n.category == "cymbal2":
                    return f"  {n.tick} = N 3 0\n  {n.tick} = N 67 0"

                elif n.category == "cymbal3":
                    return f"  {n.tick} = N 4 0\n  {n.tick} = N 68 0"

            # Iterate over each note for each group
            for group in ts.notes:
                for note in group:
                    note_data = note_parse(note)

                    # Add to chart
                    if note_data:
                        data += f'\n{note_data}'

            data += '\n}'

            print(data)
            return

            # write & parse
            with open(self._file, 'w+', encoding='utf-8', errors='ignore') as f:
                self._raw += data
                f.write(self._raw)
                self._parse(self._raw)
