#!/usr/bin/env python
import subprocess

from pathlib import Path

import music21 as mm
import numpy as np


def is_score(obj):
    return isinstance(obj, mm.lily.lilyObjects.LyScoreBlock)


end_on_tonic = True
pseudo_pentatonic = True
allow_rests = True
rest_prob = 0.2

rng = np.random.default_rng(8675309)

stim_outdir = Path("..").resolve() / "experiment" / "stimuli" / "music"
stim_outdir.mkdir(exist_ok=True)
score_outdir = Path(__file__).parent

keys = [
    *"ABCDEFG",  # major
    *"abcdefg",  # minor
    *[f"{k}-" for k in "BEADGb"],  # flatted major (plus B♭ minor)
    *[f"{k}#" for k in "dgcf"],  # sharped minor
]
keysigs = [mm.key.Key(key) for key in keys]

# pentatonic_scale = mm.scale.ConcreteScale(
#     tonic="C4",
#     pitches=("C4", "E-4", "F4", "G4", "B-4"),
#     name="C Pentatonic",
# )

# time signatures
_timesigs = ("2/4", "3/4", "4/4")
timesigs = [mm.meter.TimeSignature(ts) for ts in _timesigs]

# beat patterns
durs = {
    dur: mm.duration.Duration(type=dur) for dur in ("16th", "eighth", "quarter", "half")
}
durs.update(
    {
        f"dotted_{dur}": mm.duration.Duration(type=dur, dots=1)
        for dur in ("eighth", "quarter")
    }
)
phrases = (
    # two-beat
    [durs["half"]],
    [durs[d] for d in ("quarter", "quarter")],
    [durs[d] for d in ("quarter", "eighth", "eighth")],
    [durs[d] for d in ("eighth", "eighth", "quarter")],
    [durs[d] for d in ("dotted_quarter", "eighth")],
    [durs[d] for d in ("eighth", "eighth", "eighth", "eighth")],
    [durs[d] for d in ("eighth", "dotted_quarter")],
    # one-beat
    [durs["quarter"]],
    [durs[d] for d in ("eighth", "eighth")],
    [durs[d] for d in ("dotted_eighth", "16th")],
)

n_stims = 20
melodies = list()
scores = list()

for file_ix in range(n_stims):
    timesig = rng.choice(timesigs)  # timesigs[file_ix % len(timesigs)]
    # set tempo
    # speed = mm.tempo.MetronomeMark(number=150, referent=quarter_note, playbackOnly=True)
    bar_duration = timesig.barDuration.quarterLength
    n_bars = 3 if bar_duration == 2 else 2  # use 3 bars if 2/4 time
    n_beats = bar_duration * n_bars
    this_bar_beats = bar_duration
    beats = list()
    while n_beats > 0:
        candidates = [
            ph for ph in phrases if sum([p.quarterLength for p in ph]) <= this_bar_beats
        ]
        this_phrase = candidates[rng.choice(len(candidates))]
        phrase_dur = sum([b.quarterLength for b in this_phrase])
        n_notes = len(this_phrase)
        this_bar_beats -= phrase_dur
        n_beats -= phrase_dur
        beats.extend(this_phrase)
        # reset at end of bar
        this_bar_beats = this_bar_beats or bar_duration

    # choose the pitches
    selection = np.array([0, 2, 3, 4, 6, 7]) if pseudo_pentatonic else slice(None)
    keysig = rng.choice(keysigs)
    pitches = np.array(keysig.getPitches())[selection]
    this_pitches = rng.choice(pitches, len(beats), replace=True)
    if end_on_tonic:
        this_pitches[-1] = rng.choice([pitches[0], pitches[-1]])
    # initialize the stream
    s = mm.stream.Stream([keysig, timesig])  # TODO add speed too
    # add notes or (maybe) rests
    prev_note_was_rest = False
    for ix, (pitch, dur) in enumerate(zip(this_pitches, beats)):
        rest = (
            allow_rests  # global switch
            and not prev_note_was_rest  # avoid 2 rests in a row
            and (ix not in (0, len(beats) - 1))  # avoid rest on first or last note
            and rng.choice((False, True), p=(1 - rest_prob, rest_prob))
        )
        if rest:
            note = mm.note.Rest(dur.quarterLength)
            prev_note_was_rest = True
        else:
            note = mm.note.Note(pitch=pitch, duration=dur)
            # note.duration = dur
            prev_note_was_rest = False
        s.append(note)

    melodies.append(s)
    wav_path = stim_outdir / f"{file_ix:03}.wav"
    # write to (temporary MIDI file, then to) WAV
    midi_path = s.write(fmt="midi")  # fp=path/to/midi → save elsewhere than /tmp/
    subprocess.run(
        [
            "timidity",
            "--quiet=2",
            # append ↓ 8,1,2: 8,16,24-bits, u/s: (un)signed, l(inear), M(ono)/S(tereo)
            "--output-mode=w",
            f"--output-file={wav_path}",
            str(midi_path),
            "-EFreverb=0",
            "--sampling-freq=44100",
            # "--adjust-tempo=100",  # percent
        ],
        check=True,
        timeout=10,
    )
    # assemble all scores
    converter = mm.lily.translate.LilypondConverter()
    converter.loadFromMusic21Object(s)
    scores.extend(list(filter(is_score, converter.context.contents)))

# write scores to disk
header = (score_outdir / "scores-header.ly").read_text()
scores_path = score_outdir / "scores.ly"
scores_path.write_text("\n".join([header, *list(map(str, scores))]))
subprocess.run(
    ["lilypond", "--pdf", f"--output={score_outdir}/scores", str(scores_path)]
)  # .pdf extension automatically added
