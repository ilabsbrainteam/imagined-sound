# find words from our list of possible fake words that don't occur in any of the
# sentences we use

import yaml

from pathlib import Path

with open("all_ieee_sentences.yaml") as fid:
    all_sents = yaml.safe_load(fid)

with open("possible_fake_keywords.yaml") as fid:
    fakes = yaml.safe_load(fid)

with open(Path("..") / "src" / "imagined_sound" / "block_stims.yaml") as fid:
    block_stims = yaml.safe_load(fid)

used_fnames = [x for _, y in block_stims.items() for x in y]
used_sent_nums = [x[6:-4] for x in used_fnames]
used_sents = {k: v for k, v in all_sents.items() if k in used_sent_nums}

used_words = set(
    [x.rstrip(".").lower() for y in used_sents.values() for x in y.split()]
)
available_fakes = sorted(set(fakes) - used_words)

with open("usable_fakes.yaml", "w") as fid:
    yaml.safe_dump(available_fakes, fid)
