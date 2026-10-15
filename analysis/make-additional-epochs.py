"""Create shorter, differently-aligned epochs from the pipeline's long epochs.

The mne-bids-pipeline creates one long epoch per trial, time-locked to `stim_start`
and extending to the longest trial's `resp_end`. This reads `*_proc-clean_epo.fif`
and writes three sets of shorter epochs alongside the input in the derivatives tree:

- `proc-cleanAlignedToStimStart`: 200 ms before `stim_start` to 500 ms after the
  latest `stim_end`; time zero is `stim_start`.
- `proc-cleanAlignedToButton`: a fixed `pre_button` seconds before `button`
  to `button`; time zero is the button press.
- `proc-cleanAlignedToStimEnd`: `stim_end` to as far as the source epochs allow for
   every trial, i.e., the source epochs' `tmax` minus the latest `stim_end`).

"Latest" is relative to the alignment event, and is computed across *all* subjects
so that every output file has the same time axis (which makes grand averaging
straightforward). Within-trial event latencies are read from the
annotations stored in the epochs; the `button` event is the first button press
between `resp_start` and `resp_end`. Trials missing a needed event, or whose
window would extend beyond the available data, are dropped (and reported).

Before extraction, the source epochs are baseline-corrected by regression (Alday,
2019, doi:10.1111/psyp.13451): per channel, the single-trial data are regressed on
condition indicators plus the trial's mean over the baseline period, and the fitted
baseline contribution is subtracted from each trial. This replaces the pipeline's
baseline correction, which is disabled (`baseline = None` in `pipeline-config.py`).

The new epochs' metadata are the pipeline's metadata plus `<event>_latency` columns
(seconds, relative to the new time zero). Annotations falling inside the new window
are preserved.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, get_entities_from_fname

mne.set_log_level("WARNING")

# paths
root = Path("/data/prism")
deriv_root = root / "bids-data" / "derivatives" / "mne-bids-pipeline"
# params
task = "SpeMusImaCli"

# epoching params (seconds)
baseline = (None, 0.0)  # period used for regression-based baseline correction
pre_stim = -0.2  # (matches `epochs_tmin` in pipeline config)
post_stim = 0.5
pre_button = -3.0
# NOTE: ↑↑↑↑ if the press occurs less than `pre_button + pre_stim` seconds after
# stim_start, the trial will be dropped
#
# NOTE: for the stim-end-aligned epochs, the window is automatically computed to be
# as long as possible without dropping any trial, so there is no `post_stim_end` param.


def regress_out_baseline(epochs, baseline):
    """Regression-based baseline correction of single-trial data, in place.

    For each channel, fits ``data ~ condition indicators + baseline``, where
    ``baseline`` is each trial's mean over the baseline period (scaled by its
    across-trial SD so the design matrix is well conditioned), then subtracts the
    fitted baseline contribution from each trial. The condition indicators keep
    condition differences in the baseline from being absorbed into the baseline beta.
    """
    assert epochs.preload
    data = epochs.get_data(copy=False)  # a view of the Epochs' data; modified in place
    tmin, tmax = baseline
    tmin = epochs.times[0] if tmin is None else tmin
    tmax = epochs.times[-1] if tmax is None else tmax
    bl_mask = (epochs.times >= tmin) & (epochs.times <= tmax)
    design = np.column_stack(
        [epochs.events[:, 2] == code for code in epochs.event_id.values()]
    ).astype(float)
    for ch in mne.pick_types(epochs.info, meg=True, eeg=True):
        bl = data[:, ch, bl_mask].mean(axis=-1)
        assert bl.std() > 0, f"{epochs.ch_names[ch]} has a flat baseline; is it bad?"
        bl /= bl.std()
        beta, *_ = np.linalg.lstsq(
            np.column_stack([design, bl]), data[:, ch], rcond=None
        )
        data[:, ch] -= np.outer(bl, beta[-1])


# input files
in_patterns = (
    f"sub-*/ses-*/meg/*_task-{task}_proc-clean_epo.fif",
    f"sub-*/ses-*/meg/*_task-{task}_proc-clean_split-01_epo.fif",
)
in_files = sorted(fname for pat in in_patterns for fname in deriv_root.glob(pat))

# For each file, build DataFrame (one row per epoch, one column per event) with the
# event's latency (relative to time zero). Missing events are NaN.
fname_to_latency_df_mapping = dict()
event_names = ("stim_start", "stim_end", "resp_start", "button", "resp_end")

for fname in in_files:
    epochs = mne.read_epochs(fname, preload=False, verbose=False)
    rows = list()
    for annots in epochs.get_annotations_per_epoch():
        row = dict.fromkeys(event_names, np.nan)
        for onset, _, desc in sorted(annots):
            # ignore anything before time zero (e.g., prior trial's events)
            if onset < 0:
                continue
            # `stim_start` is always time zero (determined in `pipeline-config.py`)
            if desc.endswith("stim_start"):
                if onset < 1 / epochs.info["sfreq"]:  # the time-locking event itself
                    row["stim_start"] = 0.0
                    continue
                # else: it's the next trial's t_zero; its event shows up in this trial's
                # epoch because this trial was short
                break
            # handle other events
            if desc.endswith("stim_end"):
                key = "stim_end"
            elif desc in ("resp_start", "resp_end"):
                key = desc
            elif desc.startswith("button") and onset > row["resp_start"]:
                # ↑↑↑ accept any button press after resp_start (not just button/b1). NaN
                # comparisons are False, so missing `resp_start` → button press ignored
                key = "button"
            else:
                continue
            if np.isnan(row[key]):  # prevents writing 2nd, 3rd, etc button presses
                row[key] = onset
            if key == "resp_end":
                break  # continue to next trial
        # this shouldn't be necessary; sorting the annots and breaking on resp_end
        # should prevent it.
        # if row["button"] > row["resp_end"]:
        #     row["button"] = np.nan
        rows.append(row)
    fname_to_latency_df_mapping[fname] = pd.DataFrame(rows)
pooled_df = pd.concat(fname_to_latency_df_mapping.values(), ignore_index=True)
latest_stim_end = pooled_df["stim_end"].max()
max_avail_post_stim_dur = epochs.tmax - latest_stim_end
assert np.isfinite([latest_stim_end, max_avail_post_stim_dur]).all(), (
    "non-finite result when computing stim_end window"
)

# now we know enough to specify (tmin, tmax) for the three new epoch sets.
epo_windows = dict(
    cleanAlignedToStimStart=dict(
        tzero_event_name="stim_start", tmin=pre_stim, tmax=latest_stim_end + post_stim
    ),
    cleanAlignedToStimEnd=dict(
        tzero_event_name="stim_end", tmin=0.0, tmax=max_avail_post_stim_dur
    ),
    cleanAlignedToButton=dict(tzero_event_name="button", tmin=pre_button, tmax=0.0),
)
print("Epoch windows (s, relative to alignment event):")
for name, win in epo_windows.items():
    print(f"  {name:>24}: {win['tmin']:7.3f} to {win['tmax']:6.3f}")
print()

# now create and save the re-aligned epochs
for fname, latency_df in fname_to_latency_df_mapping.items():
    # preloaded later, only if some output needs (re)computing
    epochs = mne.read_epochs(fname, preload=False, verbose=False)
    ents = get_entities_from_fname(fname)
    bidspath_out = BIDSPath(
        subject=ents["subject"],
        session=ents["session"],
        task=task,
        root=deriv_root,
        datatype="meg",
        suffix="epo",
        extension=".fif",
        check=False,
    )
    # check the mtime of the proc_clean input epochs
    in_mtime = max(
        f.stat().st_mtime
        for f in fname.parent.glob(
            f"*_task-{task}_proc-clean_*epo.fif"
        )  # may be split files        ↑
    )
    # loop over the three new epoch sets
    print(f"sub-{ents['subject']} ses-{ents['session']}")
    for proc_name, win in epo_windows.items():
        bidspath_out.update(processing=proc_name)
        out_fname = bidspath_out.fpath
        # window edges (relative to the new time zero) rounded to whole samples
        sfreq = epochs.info["sfreq"]
        n_pre_samp = int(np.rint(win["tmin"] * sfreq))
        n_post_samp = int(np.rint(win["tmax"] * sfreq))
        # skip if the output exists, is newer than its input, & has the right window
        if not out_fname.exists():  # maybe it was written as split files
            out_fname = bidspath_out.copy().update(split="01").fpath
        if out_fname.exists() and out_fname.stat().st_mtime >= in_mtime:
            existing = mne.read_epochs(out_fname, preload=False, verbose=False)
            tol = 0.5 / sfreq
            if (
                abs(existing.tmin - win["tmin"]) < tol
                and abs(existing.tmax - win["tmax"]) < tol
            ):
                print(f"  {proc_name}: up to date, skipping")
                continue
        # something needs (re)computing so load the epochs and baseline-correct them
        # (once per file; the first time any of its outputs needs computing)
        if not epochs.preload:
            epochs.load_data()
            regress_out_baseline(epochs, baseline)
        # index (within the source epoch) of the new time zero, per epoch
        tzero_event_name = win["tzero_event_name"]
        tzero_latencies_sec = latency_df[tzero_event_name].to_numpy()
        tzero_inbounds = ~np.isnan(tzero_latencies_sec)
        tzero_ixs = np.full(len(epochs), -1)
        tzero_ixs[tzero_inbounds] = epochs.time_as_index(
            tzero_latencies_sec[tzero_inbounds], use_rounding=True
        )
        # start/stop indices
        start_ix = tzero_ixs - n_pre_samp
        stop_ix = tzero_ixs + n_post_samp + 1
        start_stop_inbounds = (
            tzero_inbounds & (start_ix >= 0) & (stop_ix <= len(epochs.times))
        )
        # report drops. `dropped.index` is positional (within the surviving source
        # epochs); `epochs.selection` maps that to the original trial number.
        dropped = latency_df.loc[~start_stop_inbounds].copy()
        dropped["reason"] = np.where(
            tzero_inbounds[~start_stop_inbounds],
            "WINDOW_EXCEEDS_SOURCE_EPOCH",
            f"NO_{tzero_event_name.upper()}_EVENT",
        )
        dropped["orig_ix"] = epochs.selection[dropped.index]
        keep = np.nonzero(start_stop_inbounds)[0]
        print(f"  {proc_name}: {len(keep)} epochs, {len(dropped)} dropped")
        for reason, grp in dropped.groupby("reason"):
            print(f"    {reason}: original epochs {grp['orig_ix'].to_list()}")
        # drop log: start from the source's (which already records the pipeline's
        # drops, e.g., autoreject) and add entries for the trials we're dropping here.
        drop_log = list(epochs.drop_log)
        for orig_ix, reason in zip(dropped["orig_ix"], dropped["reason"]):
            assert drop_log[orig_ix] == (), f"epoch {orig_ix} already dropped"
            drop_log[orig_ix] = (reason,)
        drop_log = tuple(drop_log)
        if not len(keep):
            print("    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("      WARNING: no epochs to write; skipping")
            print("    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue
        # extract the data (one epoch at a time, to avoid a second full copy)
        data = np.empty((len(keep), len(epochs.ch_names), n_pre_samp + n_post_samp + 1))
        for out_ix, in_ix in enumerate(keep):
            data[out_ix] = epochs.get_data(item=in_ix, copy=False)[
                0, :, start_ix[in_ix] : stop_ix[in_ix]
            ]
        # events: shift sample numbers to the new time zero (preserves ordering, since
        # the alignment event always precedes the next trial's stim_start)
        events = epochs.events[keep].copy()
        source_epochs_tzero_ixs = epochs.time_as_index(0, use_rounding=True)[0]
        events[:, 0] += tzero_ixs[keep] - source_epochs_tzero_ixs
        event_id = {k: v for k, v in epochs.event_id.items() if v in events[:, -1]}
        # metadata: pipeline's metadata plus event latencies relative to the new zero
        metadata = epochs.metadata.iloc[keep].reset_index(drop=True)
        for ev_name in event_names:
            metadata[f"{ev_name}_latency"] = (
                latency_df[ev_name].to_numpy()[keep] - tzero_latencies_sec[keep]
            )
        new_epochs = mne.EpochsArray(
            data,
            epochs.info,
            events=events,
            event_id=event_id,
            tmin=win["tmin"],
            baseline=None,
            metadata=metadata,
            selection=epochs.selection[keep],
            drop_log=drop_log,
            raw_sfreq=sfreq,
        )
        # carry over annotations that fall within the new window
        onsets, durations, descriptions = list(), list(), list()
        per_epoch = epochs.get_annotations_per_epoch()
        for out_ix, in_ix in enumerate(keep):
            for onset, duration, desc in per_epoch[in_ix]:
                new_onset = onset - tzero_latencies_sec[in_ix]
                if new_epochs.tmin <= new_onset <= new_epochs.tmax:
                    onsets.append(events[out_ix, 0] / sfreq + new_onset)
                    durations.append(duration)
                    descriptions.append(desc)
        if onsets:
            new_epochs.set_annotations(mne.Annotations(onsets, durations, descriptions))
        new_epochs.save(bidspath_out.fpath, overwrite=True, split_naming="bids")
        del new_epochs
    del epochs
