#!/usr/bin/env python
from pathlib import Path

from music21.note import Note
from music21.scale import ChromaticScale
from music21.stream import Stream

from helpers import midi_to_wav

# paths
project_root = Path(__file__).resolve().parents[1]
wav_dir = project_root / "stimgen" / "subvoc_music"


notes = ChromaticScale("C4")

for pitch in notes.pitches[:-1]:  # omit octave
    stream = Stream([Note(pitch=pitch)])
    # write stream to (temporary MIDI file, then to) WAV
    midi_path = stream.write(fmt="midi")  # fp=path/to/midi → save elsewhere than /tmp/
    wav_path = wav_dir / f"{pitch}.wav"
    midi_to_wav(midi_path, wav_path)
