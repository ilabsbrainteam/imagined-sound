#!/usr/bin/env python
import subprocess

from collections import Counter
from datetime import date
from pathlib import Path
# from pprint import pprint

import matplotlib.pyplot as plt
from music21.duration import Duration
from music21.instrument import Piano
from music21.key import Key
from music21.lily.lilyObjects import (
    LyScoreBlock,
    LyStenoDuration,
    LyTempoEvent,
    LyTempoRange,
)
from music21.lily.translate import LilypondConverter
from music21.meter import TimeSignature
from music21.note import Note, Rest
from music21.stream import Stream
from music21.tempo import MetronomeMark
from music21.tie import Tie
import numpy as np
import scipy.stats
import yaml

from fitter import Fitter, get_common_distributions


def is_lily_score(obj):
    return isinstance(obj, LyScoreBlock)


def n_beats(seq):
    return sum([x.quarterLength for x in seq])


def insert_metronome_mark_into_score(score, tempo):
    tempo_range = LyTempoRange(tempo)
    steno = LyStenoDuration("4")
    tempo_mark = LyTempoEvent(tempoRange=tempo_range, stenoDuration=steno)
    score[
        0
    ].scoreBody.music.compositeMusic.contents.music.compositeMusic.contents.sequentialMusic.musicList.contents[
        0
    ].music.compositeMusic.contents.sequentialMusic.musicList.contents.insert(
        2, tempo_mark
    )


def get_distribution(data):
    distributions = get_common_distributions()
    f_obj = Fitter(data, distributions=distributions)
    f_obj.fit()
    best = f_obj.get_best()  # s(hape), loc(ation), scale
    dist_name = list(best)[0]
    func = getattr(scipy.stats, dist_name)
    frozen_dist = func(**best[dist_name])
    # set a hard lower limit on duration (truncate the best-fit distribution)

    def trunc_dist(**kwargs):
        # uniformly-distributed RVs in [0,1]
        rvs = scipy.stats.uniform.rvs(**kwargs)
        # equation for truncating a distr: https://stats.stackexchange.com/a/534340
        return frozen_dist.ppf(
            rvs * (frozen_dist.cdf(max_duration) - frozen_dist.cdf(min_duration))
            + frozen_dist.cdf(min_duration)
        )

    frozen_dist.rvs = trunc_dist
    frozen_dist.dist.name += f" (truncated to [{min_duration}, {max_duration}])"
    return frozen_dist


# tweakable params
n_stims = 20
min_duration = 1.5  # seconds
max_duration = np.inf
max_n_notes = 10
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
# discard the highest number of syllables, else we get melodies that are too fast
note_distr = list(filter(lambda x: x <= max_n_notes, n_syls))
n_notes = rng.choice(note_distr, size=n_stims, replace=True)
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
keysigs = [Key(key) for key in keys]

# time signatures
_timesigs = ("2/4", "3/4", "4/4")
timesigs = [TimeSignature(ts) for ts in _timesigs]

# rhythms
durs = {dur: Duration(type=dur) for dur in ("16th", "eighth", "quarter", "half")}
durs.update(
    {f"dotted_{dur}": Duration(type=dur, dots=1) for dur in ("eighth", "quarter")}
)
phrases = (
    # two-beat
    [durs["half"]],
    [durs[d] for d in ("quarter", "quarter")],
    [durs[d] for d in ("eighth", "eighth", "quarter")],
    [durs[d] for d in ("eighth", "eighth", "eighth", "eighth")],
    [durs[d] for d in ("eighth", "dotted_quarter")],
    # one-beat
    [durs["quarter"]],
    [durs[d] for d in ("eighth", "eighth")],
    # [durs[d] for d in ("16th", "16th", "eighth")],
    # [durs[d] for d in ("16th", "dotted_eighth")],
)
nonfinal_phrases = (
    [durs[d] for d in ("quarter", "eighth", "eighth")],
    [durs[d] for d in ("dotted_quarter", "eighth")],
    # one-beat
    # [durs[d] for d in ("16th", "16th", "16th", "16th")],
    # [durs[d] for d in ("eighth", "16th", "16th")],
    # [durs[d] for d in ("dotted_eighth", "16th")],
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
    melody = list()
    prev_was_rest = False  # we're guaranteed to get at least one note before first rest
    timesig = rng.choice(timesigs)
    this_measure = timesig.barDuration.quarterLength
    # print("=" * 60)
    # pprint(this_pitches.tolist())
    while this_n_notes > 0:
        # print(f"  {this_measure=}")
        candidate_phrases = [ph for ph in phrases if len(ph) <= this_n_notes]
        # also include nonfinal phrases that are strictly shorter than remaining n_notes
        candidate_phrases.extend(
            [ph for ph in nonfinal_phrases if len(ph) < this_n_notes]
        )
        this_phrase = candidate_phrases[rng.choice(len(candidate_phrases))]
        # pprint(list(this_phrase), indent=4, width=50)
        if n_beats(this_phrase) > this_measure:
            for this_beat in this_phrase:
                this_pitch = this_pitches[0]
                this_pitches = this_pitches[1:]
                if this_beat.quarterLength > this_measure:
                    # print(f"      {this_beat.quarterLength=}")
                    pre = Note(pitch=this_pitch, duration=Duration(this_measure))
                    post = Note(
                        pitch=this_pitch,
                        duration=Duration(this_beat.quarterLength - this_measure),
                    )
                    pre.tie = Tie("start")
                    post.tie = Tie("stop")
                    melody.extend([pre, post])
                else:
                    melody.append(Note(pitch=this_pitch, duration=this_beat))
        else:
            melody.extend(
                [
                    Note(pitch=pitch, duration=dur)
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
            this_rest = Rest(type=rng.choice(("quarter", "eighth")))
            melody.append(this_rest)
            this_measure -= this_rest.quarterLength
            prev_was_rest = True
        else:
            prev_was_rest = False
        this_n_notes -= len(this_phrase)
        this_measure -= n_beats(this_phrase)
        while this_measure <= 0:
            this_measure += timesig.barDuration.quarterLength
    assert all([beat.quarterLength > 0 for beat in melody])

    # set tempo
    this_duration = durations[file_ix]
    beats_per_min = np.rint(n_beats(melody) / (this_duration / 60)).astype(int).item()
    tempo = MetronomeMark(number=beats_per_min, referent=Note(type="quarter"))

    # initialize the stream
    instrument = Piano()
    for attr in (
        "instrumentName",
        "instrumentAbbreviation",
        "partName",
        "partAbbreviation",
    ):
        setattr(instrument, attr, f"{file_ix:03}")
    stream = Stream([keysig, tempo, timesig, *melody])
    stream.partName = f"{file_ix:03}"

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
    if partial_measure := (stream.quarterLength % timesig.barDuration.quarterLength):
        stream.append(Rest(timesig.barDuration.quarterLength - partial_measure))
    # assemble all scores
    converter = LilypondConverter()
    converter.loadFromMusic21Object(stream)
    # fixup the bug that music21 doesn't write metronome marks into the score
    score = list(filter(is_lily_score, converter.context.contents))
    assert len(score) == 1
    insert_metronome_mark_into_score(score, beats_per_min)
    scores.extend(score)

# write scores to disk
header = (score_dir / "scores-header.ly").read_text()
scores_path = score_dir / f"scores_{today}.ly"
scores_path.write_text("\n".join([header, *list(map(str, scores))]))

# fixup: add stim names to score
lines_in = scores_path.read_text().split("\n")
lines_out = list()
ix = 0
for line in lines_in:
    if "\\new Voice" in line:
        line = line.replace(
            "\\new Voice { \\new Voice",
            f'\\new Staff \\with {{ instrumentName = "{ix:003}" }} {{ \\new Voice',
        )
        ix += 1
    lines_out.append(line)
assert ix == n_stims
scores_path.write_text("\n".join(lines_out))


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
