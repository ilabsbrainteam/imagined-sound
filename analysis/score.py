import mne
import numpy as np
import pandas as pd
from expyfun import binary_to_decimals
from expyfun.io import read_tab

# trigger dict
STIM_CHANNELS = dict(
    stim_start="STI001",
    _zeros="STI003",  # used for
    _ones="STI004",  # binary-encoded trial info
    button_1="STI005",
    button_2="STI006",
    button_3="STI007",
    button_4="STI008",
)
_EVENT_DICT = {
    "stim_start": 1,
    "attn_check_start": 3,
    "speech/click": 4,
    "speech/imagine": 8,
    "music/click": 6,
    "music/imagine": 10,
    "practice/speech/click": 5,
    "practice/speech/imagine": 9,
    "practice/music/click": 7,
    "practice/music/imagine": 11,
    "stim_end": 12,
    "resp_start": 13,
    "resp_end": 14,
    "finale": 15,
    "button_1": 16,
    "button_2": 32,
    "button_3": 64,
    "button_4": 128,
}
# differentiate all the "stim_end" events & event_ids by condition
trial_id_names = [
    "finale",
    *[x for x in list(_EVENT_DICT) if x.endswith(("click", "imagine"))],
]
_EVENT_DICT.update({f"{x}/stim_end": 200 + _EVENT_DICT[x] for x in trial_id_names})
for condition in list(_EVENT_DICT):
    if condition.endswith(("click", "imagine", "finale")):
        _EVENT_DICT[f"{condition}/stim_start"] = _EVENT_DICT.pop(condition)

EVENT_DICT = _EVENT_DICT.copy()  # for import to other files, after updating
REV_EVENT_DICT = {v: k for k, v in EVENT_DICT.items()}


def _stack_and_sort_arrays(*arrays):
    """Vstack event arrays and sort by sample number."""
    arr = np.vstack(arrays)
    return arr[np.argsort(arr[:, 0])]


def parse_expyfun_log(tabpath):
    tab = read_tab(tabpath)
    df_list = []
    for trial in tab:
        _row = dict(
            block=trial["block"][0][0],
            practice=trial["practice"][0][0],
            stim=trial["stimulus"][0][0],
            stim_onset=trial["play"][0][1],
        )
        if trial["response"]:
            _row["reaction_time"] = trial["response"][2][1]
        else:
            _row["reaction_time"] = pd.NA
        for key in ("attn_keyword", "attn_correct", "attn_is_fake"):
            if trial.get(key):
                val = trial[key][0][0]
                if key in ("attn_correct", "attn_is_fake"):
                    val = val == "True"
                _row.update({key: val})
            else:
                _row.update({key: pd.NA})
        attn_reaxtime = (
            trial["attn_correct"][0][1] if trial.get("attn_correct") else pd.NA
        )
        _row.update({"attn_reaction_time": attn_reaxtime})
        df_list.append(_row)
    return pd.DataFrame(df_list)


def score_func(raw):
    # extract button presses and stim-start events separately
    button_1_events = mne.find_events(raw, stim_channel=STIM_CHANNELS["button_1"])
    button_2_events = mne.find_events(raw, stim_channel=STIM_CHANNELS["button_2"])
    button_3_events = mne.find_events(raw, stim_channel=STIM_CHANNELS["button_3"])
    button_4_events = mne.find_events(raw, stim_channel=STIM_CHANNELS["button_4"])
    button_1_events[:, -1] = EVENT_DICT["button_1"]
    button_2_events[:, -1] = EVENT_DICT["button_2"]
    button_3_events[:, -1] = EVENT_DICT["button_3"]
    button_4_events[:, -1] = EVENT_DICT["button_4"]
    # 1024 is video sync pulses on STI011
    mask = sum(EVENT_DICT[f"button_{n}"] for n in (1, 2, 3, 4)) + 1024
    trial_events = mne.find_events(
        raw,
        min_duration=0.002,
        mask=mask,
        mask_type="not_and",
    )
    # parse trial ID events (sequence of four 4s or 8s)
    new_trial_events = list()
    # iterate over rows with a 1-trigger (stim start events)
    for row_ix in np.nonzero(trial_events[:, -1] == 1)[0]:
        trial_id = trial_events[(row_ix - 4) : row_ix, -1]
        # ↓ there are stim start triggers for "check" stims which don't have a preceding
        # 4-bit trial ID sequence, so don't keep those (yet)
        if all(x in (4, 8) for x in trial_id):
            bits = trial_id // 4 - 1
            new_trial_id = binary_to_decimals(bits, n_bits=4).item()
            new_trial_events.append(
                np.concat((trial_events[row_ix][:2], [new_trial_id]))
            )
            # mutate stim_end events to reflect trial type (for condition epoching)
            # (e.g., stim_end → practice/music/click/stim_end)
            found = False
            next_ix = row_ix
            while not found:
                next_ix += 1
                if (this_event := trial_events[next_ix, -1]) == EVENT_DICT["stim_end"]:
                    new_stim_end = EVENT_DICT[
                        f"{REV_EVENT_DICT[new_trial_id].removesuffix('start')}end"
                    ]
                    new_trial_events.append(
                        np.concat((trial_events[next_ix, :2], [new_stim_end]))
                    )
                    found = True
                elif this_event == 1:
                    raise RuntimeError(
                        "No stim_end event found for subject "
                        f"{raw._filenames[0].parts[-3]}, event ID {new_trial_id}, "
                        f"row {row_ix} (checked rows {row_ix + 1}:{next_ix} inclusive)"
                    )
        else:
            # assume it's an attention-check stim (verified later)
            new_trial_events.append(
                np.concat((trial_events[row_ix, :2], [EVENT_DICT["attn_check_start"]]))
            )
            # NOTE: the experiment runner script didn't send stim_end triggers for the
            # attention-check stimuli, so no need to mutate those. If needed, they can
            # be reconstructed from the attn_check_start events and the stimulus
            # durations.

    new_trial_events = np.array(new_trial_events)
    # now get all the non-stim-start rows. Go backwards to simplify skipping 4-bit
    # sequences that precede stim starts
    rev_row_ixs = list(range(trial_events.shape[0]))[::-1]
    more_trial_events = list()
    unexpected_events = list()
    skip_next_row = 0
    for row_ix in rev_row_ixs:
        if skip_next_row:
            skip_next_row -= 1
            continue
        this_sample = trial_events[row_ix, 0]
        this_event = trial_events[row_ix, -1]
        # skip stim_start rows and their associated 4-bit trial IDs
        if (
            this_sample in new_trial_events[:, 0]
            and this_event == EVENT_DICT["stim_start"]
            and all(x in (4, 8) for x in trial_events[(row_ix - 4) : row_ix, -1])
        ):
            skip_next_row += 4
            continue
        # skip stim_end events, they're handled above in the first loop:
        elif this_event == EVENT_DICT["stim_end"]:  # noqa: SIM114
            continue
        # skip attention-check events, they're added above in the first loop.
        elif (
            this_sample in new_trial_events[:, 0]
            and this_event == EVENT_DICT["stim_start"]
            and new_trial_events[
                np.nonzero(new_trial_events[:, 0] == this_sample)[0].item(), -1
            ]
            == EVENT_DICT["attn_check_start"]
        ):
            continue
        # handle response_start, response_end events
        elif this_event in (
            EVENT_DICT["resp_start"],
            EVENT_DICT["resp_end"],
        ):
            more_trial_events.append(trial_events[row_ix])
        # anything else is unexpected
        else:
            unexpected_events.append(trial_events[row_ix])
    if unexpected_events:
        unexpected_events = pd.DataFrame(
            unexpected_events,
            columns=["sample_number", "prior_sample_value", "event_id"],
        )
        print("WARNING: unexpected event IDs:")
        print(unexpected_events)

    clean_events = _stack_and_sort_arrays(
        new_trial_events,
        more_trial_events,
        button_1_events,
        button_2_events,
        button_3_events,
        button_4_events,
    )
    # make sure we didn't miss anything
    stim_start_events = [ev for ev in EVENT_DICT if ev.endswith("stim_start")]
    stim_start_codes = [
        EVENT_DICT[ev] for ev in ("attn_check_start", *stim_start_events)
    ]
    n_stims = np.isin(clean_events[:, -1], stim_start_codes).sum()
    n_stims_orig = (trial_events[:, -1] == 1).sum()
    assert n_stims == n_stims_orig, (
        f"{n_stims_orig} original {EVENT_DICT['stim_start']}-triggers vs "
        f"{n_stims} retained `stim_start` events"
    )
    # stim end
    stim_end_codes = [
        EVENT_DICT[f"{ev.removesuffix('start')}end"] for ev in stim_start_events
    ]
    n_stim_ends = np.isin(clean_events[:, -1], stim_end_codes).sum()
    n_stim_ends_orig = (trial_events[:, -1] == EVENT_DICT["stim_end"]).sum()
    assert n_stim_ends == n_stim_ends_orig, (
        f"{n_stim_ends_orig} original {EVENT_DICT['stim_end']}-triggers vs "
        f"{n_stim_ends} retained `stim_end` events"
    )
    # make sure every non-attention-check stim has an associated stim_end event
    n_attn_checks = (clean_events[:, -1] == EVENT_DICT["attn_check_start"]).sum()
    assert n_stim_ends + n_attn_checks == n_stims
    # resp start/end
    for ev in ("resp_start", "resp_end"):
        n_orig = (trial_events[:, -1] == EVENT_DICT[ev]).sum()
        n_new = (clean_events[:, -1] == EVENT_DICT[ev]).sum()
        assert n_orig == n_new, f"mismatch in event {ev}: {n_orig} orig vs {n_new} new"
    return clean_events, unexpected_events
