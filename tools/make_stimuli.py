#!/usr/bin/env python
from pathlib import Path

# from midi2audio import FluidSynth
import music21 as mm
import numpy as np

outdir = Path("..").resolve() / "src" / "imagined_sound" / "stimuli" / "music21"
outdir.mkdir(exist_ok=True)

rng = np.random.default_rng(8675309)
# fs = FluidSynth(
#     # sound_font="/home/drmccloy/.fluidsynth/MuseScore_General.sf3",
#     # sample_rate=44100,
# )

# pitch numbers: integers (0 is C4; 1 integer is 1 half-step)
# note strings: [A-G][+-]*[1-9]? → [note][flat/sharp][octave]
# intervals: (m)inor (M)ajor (A)ugmented (d)ouble(d)iminished
# sequence of strings ←→ single space-separated string
#
# mm.duration.Duration(...) float = n_quarternotes; string = "half", "eighth", etc
# mm.chord.Chord(notes: None | str | Sequence[Pitch|Note|Chord|str|int], **keywords)
#

quarter_note = mm.duration.Duration(1.0)
pentatonic_scale = mm.scale.ConcreteScale(
    tonic="C4",
    pitches=("C4", "E-4", "F4", "G4", "B-4"),
    name="C Pentatonic",
)
pitches = pentatonic_scale.getPitches()

# set tempo
speed = mm.tempo.MetronomeMark(number=150, referent=quarter_note, playbackOnly=True)


# # simple ascending scale
# melody = mm.stream.Stream()
# melody.append(speed)
# for p in pitches:
#     note = mm.note.Note(pitch=p, duration=quarter_note)
#     note.volume.velocityScalar = 1.0
#     melody.append(note)
#
# melody.show("midi", options="--no-shell")


# sequence of 3 randomly chosen intervals
n_stims = 5
for ix in range(n_stims):
    melody = mm.stream.Stream()
    melody.append(speed)
    for _ in range(3):
        chord = mm.chord.Chord(
            rng.choice(pitches, size=2, replace=False), duration=quarter_note
        )
        chord.volume.velocityScalar = 1.0
        melody.append(chord)

    melody.show("midi")
    # tmp_midi = melody.write(fmt="midi")
    # fs.midi_to_audio(str(tmp_midi), audio_file=outdir / f"three-chords-{ix:03}.mid")
