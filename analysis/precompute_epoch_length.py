from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from yaml import safe_dump

from mne_bids import find_matching_paths, get_entities_from_fname


root = Path("/data/prism")
bids_root = root / "bids-data"
metadata = root / "metadata"

event_files = find_matching_paths(root=bids_root, suffixes="events", extensions=".tsv")

epoch_durs = dict()

for ev_file in event_files:
    subj = get_entities_from_fname(ev_file)["subject"]
    epoch_durs[subj] = defaultdict(list)
    # load the trial data
    trial_data = pd.read_csv(
        root / "experiment-logs" / f"{subj}_trial_info.csv", index_col=0
    )
    # load the events data from the BIDS tree and assign trial numbers
    # trial zero is assigned to all events prior to the first stimulus
    df = pd.read_csv(ev_file, sep="\t")
    df["trial_num"] = (df["trial_type"] == "stim_start").cumsum()
    assert df["trial_num"].max() == trial_data.shape[0], "mismatched number of trials"
    # determine the ideal epoch length for each trial
    for ix, row in df.iterrows():
        # start of epoch is at stimulus end
        if row["trial_type"] != "stim_end":
            continue
        sub_df = df.loc[df["trial_num"] == row["trial_num"]]
        # get block identity
        block = trial_data.iloc[row["trial_num"] - 1]["block"]
        # for "click" blocks we end epoch at START of response period
        if block.startswith("click"):
            end_row = sub_df.loc[sub_df["trial_type"] == "resp_start"]
        # for "imagine" blocks we end epoch when they click
        else:
            end_row = sub_df.loc[sub_df["trial_type"].str.startswith("button")]
        if not len(end_row):
            continue
        start = row["sample"]
        end = end_row.iloc[0]["sample"]
        epoch_durs[subj][block].append(end - start)
    # convert to array
    for block in epoch_durs[subj]:
        epoch_durs[subj][block] = np.array(epoch_durs[subj][block])

# aggregate
final_epoch_durs = dict()

for subj, blocks in epoch_durs.items():
    final_epoch_durs[subj] = dict()
    for block, durs in blocks.items():
        # pick an epoch duration that preserves 90% of trials
        cutoff = np.percentile(durs, 10)
        final_epoch_durs[subj][block] = int(durs[durs >= cutoff].min())

with open(metadata / "derived-epoch-durs.yaml", "w") as fid:
    safe_dump(final_epoch_durs, fid)
