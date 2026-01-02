import re
import wave

from pathlib import Path

import cmudict
import syllables
import yaml

cmu = cmudict.dict()


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


# get duration of audio files
wav_dir = Path(__file__).parents[1] / "experiment" / "stimuli" / "NWF003"
wav_files = sorted(wav_dir.glob("NWF03_*.wav"))
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
fname_sentences = "all_ieee_sentences.yaml"
with open(fname_sentences) as fid:
    sentences = yaml.safe_load(fid)

# convert to dict mapping sentence IDs to syllable counts
syllable_counts = {key: count_syllables(val) for key, val in sentences.items()}

with open("ieee_n_syllables.yaml", "w") as fid:
    yaml.safe_dump(syllable_counts, fid)
