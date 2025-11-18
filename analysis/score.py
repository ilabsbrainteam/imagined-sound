import mne
import numpy as np
import pandas as pd

from expyfun.io import read_tab

# trigger dict
TRIGGERS = {
    (4, 4): 2,  # "id_trial",
    (4, 8): 3,  # "stim_end",
    (8, 4): 5,  # "resp_start",
    (8, 8): 7,  # "resp_end",
}
STIM_CHANNELS = dict(
    stim_start="STI001",
    _zeros="STI003",  # used for
    _ones="STI004",  # binary-encoded trial info
    button_1="STI005",
    button_2="STI006",
    button_3="STI007",
    button_4="STI008",
)
EVENT_DICT = dict(
    stim_start=1,
    id_trial=2,
    stim_end=3,
    resp_start=5,
    resp_end=7,
    button_1=101,
    button_2=102,
    button_3=103,
    button_4=104,
)


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
            stim=trial["stimulus"][0][0],
            stim_onset=trial["play"][0][1],
            reaction_time=trial["response"][2][1],
        )
        for key in ("attn_keyword", "attn_correct", "attn_is_fake"):
            if trial[key]:
                val = trial[key][0][0]
                if key in ("attn_correct", "attn_is_fake"):
                    val = val == "True"
                _row.update({key: val})
            else:
                _row.update({key: pd.NA})
        attn_reaxtime = trial["attn_correct"][0][1] if trial["attn_correct"] else pd.NA
        _row.update({"attn_reaction_time": attn_reaxtime})
        df_list.append(_row)
    return pd.DataFrame(df_list)


def score_func(raw):
    # extract button presses and stim-start events separately
    event_arrays = {
        event_kind: mne.find_events(raw, shortest_event=1, stim_channel=stim_ch)
        for event_kind, stim_ch in STIM_CHANNELS.items()
    }
    # give each event type its unique integer value
    for key, val in (EVENT_DICT | dict(_zeros=4, _ones=8)).items():
        if key in event_arrays:
            event_arrays[key][:, -1] = val

    # interpret the 4's and 8's as binary codes
    binary_codes = _stack_and_sort_arrays(event_arrays["_zeros"], event_arrays["_ones"])
    # assemble into recoded events array
    other_trial_events = list()
    for first, second in zip(binary_codes[::2], binary_codes[1::2]):
        couplet = np.vstack((first, second))
        _event = tuple(couplet[:, -1])
        new_event_code = TRIGGERS[_event]
        other_trial_events.append(np.array([*first[:2], new_event_code]))

    # assemble the final events array
    clean_events = _stack_and_sort_arrays(
        *[
            val
            for key, val in event_arrays.items()
            if key == "stim_start" or key.startswith("button_")
        ],
        other_trial_events,
    )
    return clean_events
