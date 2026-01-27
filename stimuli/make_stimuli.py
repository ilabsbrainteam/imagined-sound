#!/usr/bin/env python
import subprocess

from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import music21 as mm
import numpy as np
import scipy.stats
import yaml

from fitter import Fitter, get_common_distributions


def is_lily_score(obj):
    return isinstance(obj, mm.lily.lilyObjects.LyScoreBlock)


def n_beats(seq):
    return sum([x.quarterLength for x in seq])


def get_distribution(data):
    distributions = get_common_distributions()
    f_obj = Fitter(data, distributions=distributions)
    f_obj.fit()
    best = f_obj.get_best()  # s(hape), loc(ation), scale
    dist_name = list(best)[0]
    func = getattr(scipy.stats, dist_name)
    return func(**best[dist_name])


# tweakable params
n_stims = 20
end_on_tonic = True
pentatonic = True
allow_rests = True
rest_prob = 0.3
seed = 8675309

# states
rng = np.random.default_rng(seed)
today = date.today().isoformat()

# paths
project_root = Path(__file__).resolve().parents[1]
stim_metadata_dir = project_root / "stimuli" / "metadata"
score_dir = project_root / "stimuli" / "scores"
wav_dir = project_root / "experiment" / "stimuli" / f"music_{today}"
# log_dir = project_root / "logs" / "stimgen"
for _dir in (wav_dir, score_dir):
    _dir.mkdir(exist_ok=True)

# get the sentence IDs for the speech stims actually used in the experiment
with open(project_root / "experiment" / "block_stims.yaml") as fid:
    speech_stims = yaml.safe_load(fid)
speech_stim_ids = [
    n.removeprefix("NWF04_").removesuffix(".wav")
    for n in (*speech_stims["click_speech"], *speech_stims["imagine_speech"])
]

# computed params: duration (to match distribution of speech stim durations)
with open(stim_metadata_dir / "ieee_durations.yaml") as fid:
    speech_durations = yaml.safe_load(fid)
speech_durs = list(speech_durations.values())
dur_distribution = get_distribution(speech_durs)
durations = sorted(map(float, dur_distribution.rvs(size=n_stims, random_state=rng)))
# write to file what the distribution was that we fit / used
output = dict(
    name=dur_distribution.dist.name,
    shape=float(dur_distribution.kwds["s"]),
    location=float(dur_distribution.kwds["loc"]),
    scale=float(dur_distribution.kwds["scale"]),
)
with open(stim_metadata_dir / "stim_duration_distribution.yaml", "w") as fid:
    yaml.safe_dump(output, fid)

# computed params: number of notes (to match distribution of speech stim syllable counts)
with open(stim_metadata_dir / "ieee_n_syllables.yaml") as fid:
    speech_syllables = yaml.safe_load(fid)
n_syls = list(speech_syllables.values())
syl_distribution = dict(sorted(Counter(n_syls).items()))
n_notes = rng.choice(n_syls, size=n_stims, replace=True)
# write to file the distribution was that we used
with open(stim_metadata_dir / "stim_nsyll_distribution.yaml", "w") as fid:
    yaml.safe_dump(dict(syl_distribution), fid)

# key signatures
keys = [
    *"ABCDEFG",  # major
    *"abcdefg",  # minor
    *[f"{k}-" for k in "BEADGb"],  # flatted major (plus B♭ minor)
    *[f"{k}#" for k in "dgcf"],  # sharped minor
]
keysigs = [mm.key.Key(key) for key in keys]

# time signatures
_timesigs = ("2/4", "3/4", "4/4")
timesigs = [mm.meter.TimeSignature(ts) for ts in _timesigs]

# rhythms
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
    [durs[d] for d in ("16th", "16th", "eighth")],
    [durs[d] for d in ("16th", "dotted_eighth")],
)
nonfinal_phrases = (
    # one-beat
    [durs[d] for d in ("16th", "16th", "16th", "16th")],
    [durs[d] for d in ("eighth", "16th", "16th")],
    [durs[d] for d in ("dotted_eighth", "16th")],
)

# containers
streams = list()
scores = list()

for file_ix in range(n_stims):
    # choose the pitches
    this_n_notes = n_notes[file_ix]
    keysig = rng.choice(keysigs)
    if not pentatonic:
        selection = slice(None)
    elif keysig.type == "major":
        selection = np.array([0, 1, 2, 4, 5, 7])
    else:
        selection = np.array([0, 2, 3, 4, 6, 7])
    pitches = np.array(keysig.getPitches())[selection]
    # TODO: CONSIDER CONSTRAINING INTERVALS TO BE NO LARGER THAN -3 OR +3
    # (w/r/t notes in the scale / indices in `pitches`)
    this_pitches = rng.choice(pitches, this_n_notes, replace=True)
    if end_on_tonic:
        this_pitches[-1] = rng.choice([pitches[0], pitches[-1]])

    # create rhythm
    rhythm = list()
    melody = list()
    prev_was_rest = False  # we're guaranteed to get at least one note before first rest
    while this_n_notes > 0:
        candidates = [ph for ph in phrases if len(ph) <= this_n_notes]
        candidate_nonfinals = list(
            filter(lambda ph: len(ph) < this_n_notes, nonfinal_phrases)
        )
        candidates.extend(candidate_nonfinals)
        this_phrase = candidates[rng.choice(len(candidates))]
        this_n_notes -= len(this_phrase)
        rhythm.extend(this_phrase)
        melody.extend(
            [
                mm.note.Note(pitch=pitch, duration=dur)
                for pitch, dur in zip(this_pitches, this_phrase)
            ]
        )
        this_pitches = this_pitches[len(this_phrase) :]
        if (
            allow_rests
            and not prev_was_rest  # avoid 2 rests in a row
            and len(this_pitches)  # not the end of the melody
            and rng.choice((False, True), p=(1 - rest_prob, rest_prob))
        ):
            rest_dur = rng.choice(("quarter", "eighth"))
            rhythm.append(mm.duration.Duration(type=rest_dur))
            melody.append(mm.note.Rest(type=rest_dur))
            prev_was_rest = True
        else:
            prev_was_rest = False
    assert len(rhythm) == len(melody)
    assert n_beats(rhythm) == n_beats(melody)

    # set tempo
    this_duration = durations[file_ix]
    beats_per_min = np.rint(n_beats(rhythm) / (this_duration / 60)).astype(int).item()
    tempo = mm.tempo.MetronomeMark(
        number=beats_per_min, referent=mm.note.Note(type="quarter")
    )  # playbackOnly=True
    tempo.placement = "above"

    # initialize the stream
    timesig = rng.choice(timesigs)
    stream = mm.stream.Stream([keysig, timesig, tempo, *melody])

    streams.append(stream)
    wav_path = wav_dir / f"{file_ix:03}.wav"
    # write to (temporary MIDI file, then to) WAV
    midi_path = stream.write(fmt="midi")  # fp=path/to/midi → save elsewhere than /tmp/
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
    # append rest to yield full measure, as needed (for score only) TODO THIS IS BUGGY I THINK?
    if missing_beats := (stream.quarterLength % timesig.barDuration.quarterLength):
        stream.append(mm.note.Rest(missing_beats))
    # assemble all scores
    converter = mm.lily.translate.LilypondConverter()
    converter.loadFromMusic21Object(stream)
    scores.extend(list(filter(is_lily_score, converter.context.contents)))

# write scores to disk
header = (score_dir / "scores-header.ly").read_text()
scores_path = score_dir / f"scores_{today}.ly"
scores_path.write_text("\n".join([header, *list(map(str, scores))]))
subprocess.run(
    ["lilypond", "--pdf", f"--output={score_dir}/scores_{today}", str(scores_path)]
)  # .pdf extension is automatically added


# plot stimulus properties
fig, axs = plt.subplots(1, 2, layout="constrained")
color_speech = "C0"
color_music = "C2"
color_distr = "C1"
distr_label = (
    f"{dur_distribution.dist.name}("
    f"shape={dur_distribution.kwds['s']:0.3}, "
    f"loc={dur_distribution.kwds['loc']:0.3}, "
    f"scale={dur_distribution.kwds['scale']:0.3})"
)
x = np.linspace(dur_distribution.ppf(0.001), dur_distribution.ppf(0.999), 999)
y_distr = dur_distribution.pdf(x)
y_music = dur_distribution.pdf(durations)
axs[0].plot(x, y_distr, color=color_distr, lw=1.5, label=distr_label)
axs[0].plot(durations, y_music, "o", color=color_music, ms=6)
axs[0].vlines(durations, 0, y_music, color=color_music, lw=1, label="music")
axs[0].vlines(speech_durs, 0, -0.05, color=color_speech, lw=1, label="speech")
axs[0].legend(loc="upper right")
axs[0].set(title="Durations", ylabel="density", xlabel="seconds")

hist_kwargs = dict(
    bins=np.arange(np.min(n_syls), np.max(n_syls) + 1, dtype=int),
    align="left",
    rwidth=0.9,
)
axs[1].hist(
    n_notes,
    color=color_music,
    label="music",
    weights=np.full(len(n_notes), 1 / len(n_notes)),
    **hist_kwargs,
)
axs[1].hist(
    n_syls,
    color=color_speech,
    label="speech",
    weights=np.full(len(n_syls), -1 / len(n_syls)),
    **hist_kwargs,
)
axs[1].legend()
axs[1].set(
    title="Syllable/note count", ylabel="density", xlabel="number of syllables/notes"
)

fig.suptitle("Distribution of speech and music stimulus properties")
fig.set_size_inches(12, 6)
fig.savefig(stim_metadata_dir / "stim-properties.pdf")
