from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from mne_bids import find_matching_paths, get_entities_from_fname
from yaml import safe_dump

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
    # trial 0 is assigned to all events prior to the first stimulus, so the first real
    # trial is number 1
    df = pd.read_csv(ev_file, sep="\t")
    # assign trial numbers
    df["trial_num"] = pd.array(
        df["trial_type"].str.endswith("stim_start").cumsum(),
        dtype=pd.Int64Dtype(),
    )
    # cleanup and write to disk
    df.reset_index(drop=True, inplace=True)
    df.to_csv(ev_file, sep="\t", index=False)
    # minus 1 here because of "trial# 0" ↓↓↓ (button presses preceding first trial)
    assert df["trial_num"].unique().size - 1 == trial_data.shape[0], (
        f"mismatched number of trials {df['trial_num'].unique().size - 1} vs TAB {trial_data.shape[0]}"
    )
    # determine the ideal epoch length for each trial
    for ix, row in df.iterrows():
        # start of epoch is at stimulus end
        if not row["trial_type"].endswith("stim_end"):
            continue
        sub_df = df.loc[df["trial_num"] == row["trial_num"]]
        # get block identity
        block = trial_data.iloc[row["trial_num"] - 1]["block"]
        if block == "finale":
            continue
        # for "click" blocks we end epoch at START of response period
        if block.startswith("click"):
            end_row = sub_df.loc[sub_df["trial_type"].str.endswith("resp_start")]
        # for "imagine" blocks we end epoch when they first click
        else:
            in_resp_period = False
            for _ix, _row in sub_df.iterrows():
                if _row["trial_type"] == "resp_start":
                    in_resp_period = True
                if in_resp_period and _row["trial_type"].startswith("button"):
                    end_row = sub_df.loc[[_ix]]  # so it's a 1-row dataframe
        if not len(end_row):
            print(f"sub-{subj}: trial {row['trial_num']} ({block}) no end row found")
        else:
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
