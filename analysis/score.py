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
EVENT_DICT = {
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
trial_id_names = [x for x in list(EVENT_DICT) if x.endswith(("click", "imagine"))]
EVENT_DICT.update({f"{x}/stim_end": 200 + EVENT_DICT[x] for x in trial_id_names})
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


def score_func_new_triggers(raw, stim_type=None):
    # extract button presses and stim-start events separately
    button_1_events = mne.find_events(raw, stim_channel=STIM_CHANNELS["button_1"])
    button_2_events = mne.find_events(raw, stim_channel=STIM_CHANNELS["button_2"])
    button_3_events = mne.find_events(raw, stim_channel=STIM_CHANNELS["button_3"])
    button_4_events = mne.find_events(raw, stim_channel=STIM_CHANNELS["button_4"])
    button_1_events[:, -1] = EVENT_DICT["button_1"]
    button_2_events[:, -1] = EVENT_DICT["button_2"]
    button_3_events[:, -1] = EVENT_DICT["button_3"]
    button_4_events[:, -1] = EVENT_DICT["button_4"]
    mask = sum(EVENT_DICT[f"button_{n}"] for n in (1, 2, 3, 4))
    trial_events = mne.find_events(
        raw,
        shortest_event=1,
        mask=mask,
        mask_type="not_and",
    )
    # parse trial ID events (sequence of four 4s or 8s)
    new_trial_events = list()
    ixs = list(range(trial_events.shape[0]))
    # iterate over rows with a 1-trigger (stim start events)
    for row_ix in np.nonzero(trial_events[:, -1] == 1)[0]:
        trial_id = trial_events[(row_ix - 4) : row_ix, -1]
        # ↓ there are stim start triggers for "check" stims which don't have a preceding
        # 4-bit trial ID sequence, so don't keep those (yet)
        if all(x in (4, 8) for x in trial_id):
            bits = trial_id // 4 - 1
            new_trial_events.append(
                np.concat(
                    (
                        trial_events[row_ix][:2],
                        [binary_to_decimals(bits, n_bits=4).item()],
                    ),
                )
            )
    new_trial_events = np.array(new_trial_events)
    # now get all the non-stim-start rows. Go backwards to simplify skipping 4-bit
    # sequences that precede stim starts
    more_trial_events = list()
    unexpected_events = list()
    skip_next_row = 0
    for row_ix in ixs[::-1]:
        if skip_next_row:
            skip_next_row -= 1
            continue
        # skip stim_start rows and their associated 4-bit trial IDs
        if (
            trial_events[row_ix, 0] in new_trial_events[:, 0]
            and trial_events[row_ix, -1] == 1
            and all(x in (4, 8) for x in trial_events[(row_ix - 4) : row_ix, -1])
        ):
            skip_next_row += 4
            continue
        # handle stim_start for attention-check stims
        if trial_events[row_ix, -1] == 1:
            more_trial_events.append(np.concat((trial_events[row_ix][:2], [3])))
        # handle stim_stop, response_start, response_end
        elif trial_events[row_ix, -1] in (12, 13, 14):
            more_trial_events.append(trial_events[row_ix])
        else:
            unexpected_events.append(trial_events[row_ix])
    if unexpected_events:
        print(f"BADNESS unexpected event IDs: {unexpected_events}")

    # mutate stim_end events to reflect trial type (for condition epoching)
    # (e.g., stim_end → practice/music/click/stim_end)
    for row in new_trial_events:
        if REV_EVENT_DICT.get(row[-1]) == "stim_end":
            row[-1] = EVENT_DICT[f"{trial_id}/stim_end"]

    clean_events = _stack_and_sort_arrays(
        new_trial_events,
        more_trial_events,
        button_1_events,
        button_2_events,
        button_3_events,
        button_4_events,
    )
    return clean_events
