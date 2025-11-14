"""Create BIDS folder structure for "prism" data."""

import re
import yaml

from pathlib import Path
from warnings import filterwarnings

import mne
import numpy as np
import pandas as pd

from expyfun.io import read_tab
from mne_bids import BIDSPath, mark_channels, write_raw_bids

# path stuff
root = Path("/data/prism").resolve()
orig_data = root / "orig-data"
bids_root = root / "bids-data"
metadata = root / "metadata"

mne.set_log_level("WARNING")
# suppress messages about IAS / MaxShield
filterwarnings(
    action="ignore",
    message="This file contains raw Internal Active Shielding data",
    category=RuntimeWarning,
    module="mne",
)
# escalate MNE-BIDS warning (we don't want to miss these)
filterwarnings(
    action="error",
    message="No events found or provided",
    category=RuntimeWarning,
    module="mne_bids",
)

# trigger dict
triggers = {
    (4, 4): 2,  # "id_trial",
    (4, 8): 3,  # "stim_end",
    (8, 4): 5,  # "resp_start",
    (8, 8): 7,  # "resp_end",
}
event_dict = dict(
    stim_start=1,
    id_trial=2,
    stim_end=3,
    resp_start=5,
    resp_end=7,
)

# metadata
with open(metadata / "daysback.yaml") as fid:
    DAYSBACK = yaml.safe_load(fid)

with open(metadata / "bad-channels.yaml") as fid:
    prebads = yaml.safe_load(fid)

read_raw_kw = dict(preload=False, allow_maxshield=True)
bids_path = BIDSPath(
    root=bids_root, datatype="meg", suffix="meg", extension=".fif", task="speech"
)

#
data_folder_pattern = re.compile(r"\w+/\d+")
rec_pattern = re.compile(r"prism_(?P<subj>\w+)_(?P<run>\d+)_raw.fif")
erm_pattern = re.compile(r"prism_(?P<subj>\w+)_erm_raw.fif")
tab_pattern = re.compile(
    r"prism_[a-zA-Z]{2}_\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2}\.\d{6}.tab"
)

for data_folder in orig_data.rglob("*/*/"):
    _dirpath = data_folder.relative_to(orig_data)
    if not data_folder_pattern.match(str(_dirpath)):
        print(f"skipping folder {_dirpath}")
        continue
    ermpaths = list()
    tabpaths = list()
    fpaths = list()
    for _fpath in data_folder.iterdir():
        if rec_match := rec_pattern.match(_fpath.name):
            fpaths.append(_fpath)
            subj = rec_match.group("subj")
            bids_path.update(subject=subj)
        elif erm_pattern.match(_fpath.name):
            ermpaths.append(_fpath)
        elif tab_pattern.match(_fpath.name):
            tabpaths.append(_fpath)
    assert len(fpaths), f"no data files found in {_dirpath}"
    assert len(ermpaths), f"no ERM files found in {_dirpath}"
    assert len(tabpaths), f"no experiment TAB files found in {_dirpath}"
    # TODO handle multiple sessions?
    assert len(fpaths) == 1, f"multiple data files found in {_dirpath}"
    assert len(ermpaths) == 1, f"multiple ERM files found in {_dirpath}"
    assert len(tabpaths) == 1, f"multiple experiment TAB files found in {_dirpath}"
    fpath = fpaths[0]
    ermpath = ermpaths[0]
    tabpath = tabpaths[0]
    # load the raw
    raw = mne.io.read_raw_fif(fpath, **read_raw_kw)
    erm = mne.io.read_raw_fif(ermpath, **read_raw_kw)
    # mark bads
    if subj in prebads:
        mark_channels(
            bids_path=bids_path,
            ch_names=prebads[subj],
            status="bad",
            descriptions="prebad",
        )
    # TODO extract below into separate score func
    # extract events
    tab = read_tab(tabpath)
    df_list = []
    for trial in tab:
        foo = dict(
            stim=trial["stimulus"][0][0],
            stim_onset=trial["play"][0][1],
            reaction_time=trial["response"][2][1],
        )
        for key in ("attn_keyword", "attn_correct", "attn_is_fake"):
            if trial[key]:
                val = trial[key][0][0]
                if key in ("attn_correct", "attn_is_fake"):
                    val = val == "True"
                foo.update({key: val})
            else:
                foo.update({key: pd.NA})
        attn_reaxtime = trial["attn_correct"][0][1] if trial["attn_correct"] else pd.NA
        foo.update({"attn_reaction_time": attn_reaxtime})
        df_list.append(foo)
    df = pd.DataFrame(df_list)
    events = mne.find_events(raw, shortest_event=1)
    clean_events = list()
    skip_next = False
    for ix, row in enumerate(events):
        if skip_next:
            skip_next = False
            continue
        # ignore STI005 - STI008 (buttons)  # TODO don't ignore buttons
        if row[-1] in (16, 32, 64, 128):
            continue
        # trigger was "1" → stim start
        if row[-1] == 1:
            clean_events.append(row)
            continue
        # trigger was odd → must have overlapped with a 1-trigger → stim start
        if row[-1] % 2 == 1:
            print(f"unexpected odd-valued event: {row[-1]}")
            clean_events.append(row)
            continue
        seq = tuple(events[ix : (ix + 2), -1] % 16)
        # sequence is in our dict of expected trigger sequences:
        if new_ev := triggers.get(seq):
            clean_events.append(np.array([*row[:2], new_ev]))
            skip_next = True
            continue
        else:
            raise RuntimeError("unexpected trigger sequence")
    clean_events = np.vstack(clean_events)
    stim_starts = clean_events[clean_events[:, -1] == 1]
    # TODO extract above into separate score func
    write_raw_bids(
        raw=raw,
        events=clean_events,
        event_id=event_dict,
        bids_path=bids_path,
        empty_room=erm,
        anonymize=dict(daysback=DAYSBACK),
        overwrite=True,
    )
