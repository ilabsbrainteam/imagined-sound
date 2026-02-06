# find words from our list of possible fake words that don't occur in any of the
# sentences we use
from pathlib import Path

import yaml

# paths
project_root = Path(__file__).parents[1]
stim_metadata_dir = project_root / "stimgen" / "metadata"

with open(stim_metadata_dir / "ieee_sentences.yaml") as fid:
    all_sents = yaml.safe_load(fid)

with open(stim_metadata_dir / "possible_fake_keywords.yaml") as fid:
    fakes = yaml.safe_load(fid)

with open(project_root / "experiment" / "block_stims.yaml") as fid:
    block_stims = yaml.safe_load(fid)

used_fnames = [x for _, y in block_stims.items() for x in y]
used_sent_nums = [x[6:-4] for x in used_fnames]
used_sents = {k: v for k, v in all_sents.items() if k in used_sent_nums}

used_words = set(
    [x.rstrip(".").lower() for y in used_sents.values() for x in y.split()]
)
available_fakes = sorted(set(fakes) - used_words)

with open(stim_metadata_dir / "usable_fakes.yaml", "w") as fid:
    yaml.safe_dump(available_fakes, fid)
