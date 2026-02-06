# analyze the corpus of sentences to get total duration and number of syllables
import re
import wave

from pathlib import Path

import cmudict
import syllables
import yaml

cmu = cmudict.dict()  # expensive, so use singleton (not inside function)


def count_syllables(sentence):
    cmu_syll = re.compile(r"[A-Z]{1,2}\d", flags=(re.A | re.I))
    n_syllables = 0
    words = re.findall(r"\w+", sentence.lower())
    for word in words:
        if word in cmu:
            first_entry = cmu[word][0]
            n_syllables += sum(map(bool, map(cmu_syll.match, first_entry)))
        else:
            n_syllables += syllables.estimate(word)
    return n_syllables


# paths
project_root = Path(__file__).parents[1]
stim_metadata_dir = project_root / "stimgen" / "metadata"
wav_dir = project_root / "experiment" / "stimuli" / "speech"

# get duration of audio files
wav_files = sorted(wav_dir.glob("NW[MF]0[0-9]_[0-9][0-9]-[0-9][0-9].wav"))
pattern = re.compile(r"\w+_(?P<sent_id>\d\d-\d\d)\.wav")
wav_durs = dict()
for wav_file in wav_files:
    sent_id = pattern.match(wav_file.name).group("sent_id")
    with wave.open(str(wav_file)) as fid:
        dur = fid.getnframes() / fid.getframerate()
    wav_durs[sent_id] = dur
# write to file
with open("ieee_durations.yaml", "w") as fid:
    yaml.safe_dump(wav_durs, fid)

# load the sentence texts (dict mapping sentence IDs to sentence texts)
fname_sentences = stim_metadata_dir / "ieee_sentences.yaml"
with open(fname_sentences) as fid:
    sentences = yaml.safe_load(fid)

# convert to dict mapping sentence IDs to syllable counts
syllable_counts = {key: count_syllables(val) for key, val in sentences.items()}

with open("ieee_n_syllables.yaml", "w") as fid:
    yaml.safe_dump(syllable_counts, fid)
