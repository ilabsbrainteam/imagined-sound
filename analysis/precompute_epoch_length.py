import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne_bids import find_matching_paths, get_entities_from_fname
from yaml import safe_dump

root = Path("/data/prism")
bids_root = root / "bids-data"
metadata = root / "metadata"
figdir = Path(__file__).resolve().parent / "figures"

visualize = False  # set to False to skip the trial-duration visualization

event_files = find_matching_paths(root=bids_root, suffixes="events", extensions=".tsv")

epoch_durs = dict()
sfreqs = dict()

for ev_file in event_files:
    ents = get_entities_from_fname(ev_file)
    subj = ents["subject"]
    session = ents["session"]
    epoch_durs[subj] = defaultdict(list)
    # sampling frequency (for plotting durations in seconds)
    with open(ev_file.copy().update(suffix="meg", extension=".json")) as fid:
        sfreqs[subj] = json.load(fid)["SamplingFrequency"]
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
    # duration of each (non-practice, non-finale) trial, from stim_start to resp_end
    for trial_num, trial_df in df.groupby("trial_num"):
        if trial_num == 0:  # button presses preceding first trial
            continue
        trial_info = trial_data.iloc[trial_num - 1]
        block = trial_info["block"]
        if block == "finale" or trial_info["practice"]:
            continue
        start = trial_df.loc[
            trial_df["trial_type"].str.endswith("stim_start"), "sample"
        ].item()
        end = trial_df.loc[trial_df["trial_type"] == "resp_end", "sample"]
        if not len(end):
            print(f"sub-{subj}: trial {trial_num} ({block}) has no resp_end event")
            continue
        epoch_durs[subj][block].append(end.iloc[0] - start)
    # convert to array
    for block in epoch_durs[subj]:
        epoch_durs[subj][block] = np.array(epoch_durs[subj][block])

# aggregate
final_epoch_durs = dict()

if visualize:
    # visualize the distribution of trial durations (one panel per block, one strip of
    # tick marks per subject) to help inform a sensible truncation rule for outliers.
    # Candidate truncation thresholds (pooled percentiles across subjects) are drawn as
    # vertical lines; the per-subject max (what gets written to the YAML) is marked too.
    block_order = ("imagine_speech", "imagine_music", "click_speech", "click_music")
    subjects = sorted(epoch_durs)
    percentiles = (95, 99)
    # fixed color assignment by block: hue=stim type (speech=blue, music=orange) and
    # lightness=response type (imagine=dark, click=light); same scheme as unequal-epochs.py
    tab20 = plt.get_cmap("tab20").colors
    block_colors = {
        "imagine_speech": tab20[0],
        "click_speech": tab20[1],
        "imagine_music": tab20[2],
        "click_music": tab20[3],
    }
    pct_linestyles = dict(zip(percentiles, ("--", ":")))
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(10, 1.2 + 0.35 * len(subjects) * 2),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    print("Trial duration summary (seconds), pooled across subjects:")
    for ax, block in zip(axs.flat, block_order):
        color = block_colors[block]
        pooled = list()
        for ix, subj in enumerate(subjects):
            durs_sec = epoch_durs[subj].get(block, np.array([])) / sfreqs[subj]
            pooled.extend(durs_sec)
            ax.eventplot(
                durs_sec,
                lineoffsets=ix,
                linelengths=0.8,
                colors=[color],
                linewidths=0.75,
            )
            if len(durs_sec):
                # per-subject max: what gets used as the epoch duration for this block
                ax.plot(
                    durs_sec.max(), ix, marker="|", color="k", markersize=10, zorder=3
                )
        pooled = np.array(pooled)
        if not len(pooled):
            ax.set_title(f"{block} (no trials)")
            continue
        for pct in percentiles:
            value = np.percentile(pooled, pct)
            ax.axvline(
                value,
                color="0.3",
                linestyle=pct_linestyles[pct],
                linewidth=1,
                label=f"pooled {pct}th percentile",
                zorder=0,
            )
        ax.set_title(f"{block}  (n={len(pooled)} trials)")
        print(
            f"  {block:>15}: median={np.median(pooled):.2f}  "
            + "  ".join(
                f"p{pct}={np.percentile(pooled, pct):.2f}" for pct in percentiles
            )
            + f"  max={pooled.max():.2f}"
        )
    for ax in axs[-1, :]:
        ax.set_xlabel("Trial duration, stim_start to resp_end (s)")
    for ax in axs[:, 0]:
        ax.set_yticks(range(len(subjects)), [f"sub-{subj}" for subj in subjects])
        ax.set_ylabel("Subject")
    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.6, len(subjects) - 0.4)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    handles.append(
        plt.Line2D([], [], marker="|", color="k", markersize=10, linestyle="")
    )
    labels.append("per-subject max (used as epoch duration)")
    fig.legend(
        handles, labels, loc="outside lower center", ncol=len(labels), frameon=False
    )
    fig.suptitle("Distribution of trial durations by block and subject")
    fig.savefig(figdir / "trial-duration-distributions.pdf")
    plt.close(fig)

for subj, blocks in epoch_durs.items():
    final_epoch_durs[subj] = dict()
    for block, durs in blocks.items():
        # longest trial (in samples) for this subject & block
        final_epoch_durs[subj][block] = int(durs.max())

with open(metadata / "derived-epoch-durs.yaml", "w") as fid:
    safe_dump(final_epoch_durs, fid)
