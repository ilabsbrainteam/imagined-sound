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
    ents = get_entities_from_fname(ev_file)
    subj = ents["subject"]
    session = ents["session"]
    epoch_durs[subj] = defaultdict(list)
    # load the trial data
    trial_data = pd.read_csv(
        root / "experiment-logs" / f"{subj}_{session}_trial_info.csv", index_col=0
    )
    # load the events data from the BIDS tree and assign trial numbers
    # trial zero is assigned to all events prior to the first stimulus
    df = pd.read_csv(ev_file, sep="\t")
    # avoid doing this if already done
    if "trial_num" not in df.columns:
        # assign trial numbers
        df["trial_num"] = pd.array(
            (df["trial_type"].str.endswith(("click", "imagine"))).cumsum(),
            dtype=pd.Int64Dtype(),
        )
        # fixup for spurious triggers
        df["spurious_trial"] = df.groupby("trial_num")["trial_type"].transform(
            lambda x: "stim_end" not in x.array
        )
        df.loc[df["spurious_trial"], "trial_num"] = pd.NA
        # all the spurious triggers come mid-trial, so ffill (not bfill)
        df["trial_num"] = df["trial_num"].ffill()
        # handle the initial button-presses from start-of-experiment instructions
        df["trial_num"] = df["trial_num"].fillna(-1)
        # now fixup the trial numbers to be sequential
        df["trial_num"] = (
            df["trial_num"] != df["trial_num"].shift(periods=1, fill_value=-1)
        ).cumsum() - 1
        # cleanup and write to disk
        df.drop(columns=["spurious_trial"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(ev_file, sep="\t", index=False)

    # minus 1 here because of "trial# -1" ↓↓↓ (button presses preceding first trial)
    assert df["trial_num"].unique().size - 1 == trial_data.shape[0], (
        "mismatched number of trials"
    )
    # determine the ideal epoch length for each trial
    for ix, row in df.iterrows():
        # start of epoch is at stimulus end
        if not row["trial_type"].endswith("stim_end"):
            continue
        sub_df = df.loc[df["trial_num"] == row["trial_num"]]
        # get block identity
        block = trial_data.iloc[row["trial_num"] - 1]["block"]
        # for "click" blocks we end epoch at START of response period
        if block.startswith("click"):
            end_row = sub_df.loc[sub_df["trial_type"].str.endswith("resp_start")]
        # for "imagine" blocks we end epoch when they first click
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
