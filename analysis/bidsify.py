"""Create BIDS folder structure for "prism" data."""

import re
import yaml

from pathlib import Path
from warnings import filterwarnings

import mne
import numpy as np
from mne_bids import (
    BIDSPath,
    get_anat_landmarks,
    mark_channels,
    write_anat,
    write_meg_calibration,
    write_meg_crosstalk,
    write_raw_bids,
)

from score import parse_expyfun_log, score_func


EVENT_DICT = {
    "id_trial": 2,
    "speech/nothing/stim_start": 111,
    "speech/nothing/stim_end": 113,
    "speech/nothing/resp_start": 115,
    "speech/nothing/resp_end": 117,
    "speech/imagine/stim_start": 121,
    "speech/imagine/stim_end": 123,
    "speech/imagine/resp_start": 125,
    "speech/imagine/resp_end": 127,
    "music/nothing/stim_start": 211,
    "music/nothing/stim_end": 213,
    "music/nothing/resp_start": 215,
    "music/nothing/resp_end": 217,
    "music/imagine/stim_start": 221,
    "music/imagine/stim_end": 223,
    "music/imagine/resp_start": 225,
    "music/imagine/resp_end": 227,
    "button_1": 301,
    "button_2": 302,
    "button_3": 303,
    "button_4": 304,
}
stim_start_events = [
    EVENT_DICT[key] for key in EVENT_DICT if key.endswith("stim_start")
]

# path stuff
root = Path("/data/prism").resolve()
orig_data = root / "orig-data"
bids_root = root / "bids-data"
metadata = root / "metadata"
cal_dir = root / "calibration"
mri_dir = root / "anat"
trial_info = root / "experiment-logs"
trial_info.mkdir(exist_ok=True)

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

# metadata
with open(metadata / "daysback.yaml") as fid:
    DAYSBACK = yaml.safe_load(fid)

with open(metadata / "bad-channels.yaml") as fid:
    prebads = yaml.safe_load(fid)

read_raw_kw = dict(preload=False, allow_maxshield=True)
bids_path = BIDSPath(
    root=bids_root, datatype="meg", suffix="meg", extension=".fif", task="speech"
)

# filename patterns
data_folder_pattern = re.compile(r"\w+/\d+")
rec_pattern = re.compile(r"prism_(?P<subj>\w+)_(?P<run>\d+)_raw.fif")
erm_pattern = re.compile(r"prism_(?P<subj>\w+)_erm_raw.fif")
tab_pattern = re.compile(
    r"prism_(?P<subj>\w+)_\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2}(?:\.\d{6})?.tab"
)

for data_folder in orig_data.rglob("*/*/"):
    _dirpath = data_folder.relative_to(orig_data)
    if not data_folder_pattern.match(str(_dirpath)):
        print(f"skipping folder {_dirpath}")
        continue
    session = _dirpath.parts[-1]
    bids_path.update(session=session)
    rawpaths = list()
    ermpaths = list()
    tabpaths = list()
    for _fpath in data_folder.iterdir():
        if rec_match := rec_pattern.match(_fpath.name):
            rawpaths.append(_fpath)
            subj = rec_match.group("subj")
            bids_path.update(subject=subj)
        elif erm_pattern.match(_fpath.name):
            ermpaths.append(_fpath)
        elif tab_pattern.match(_fpath.name):
            tabpaths.append(_fpath)
    assert len(rawpaths), f"no data files found in {_dirpath}"
    assert len(ermpaths), f"no ERM files found in {_dirpath}"
    assert len(tabpaths), f"no experiment TAB files found in {_dirpath}"
    # TODO handle multiple sessions?
    assert len(rawpaths) == 1, f"multiple data files found in {_dirpath}"
    assert len(ermpaths) == 1, f"multiple ERM files found in {_dirpath}"
    assert len(tabpaths) == 1, f"multiple experiment TAB files found in {_dirpath}"
    rawpath = rawpaths[0]
    ermpath = ermpaths[0]
    tabpath = tabpaths[0]
    # load the raw
    raw = mne.io.read_raw_fif(rawpath, **read_raw_kw)
    erm = mne.io.read_raw_fif(ermpath, **read_raw_kw)
    # extract events
    if subj == "cz":  # hack for pilot data
        stim_type_dict = {"251021": "speech", "251118": "music"}
        events = score_func(raw=raw, stim_type=stim_type_dict[session])
    else:
        events = score_func(raw=raw)
    write_raw_bids(
        raw=raw,
        events=events,
        event_id=EVENT_DICT,
        bids_path=bids_path,
        empty_room=erm,
        anonymize=dict(daysback=DAYSBACK),
        overwrite=True,
    )
    # mark bads
    if subj in prebads:
        mark_channels(
            bids_path=bids_path,
            ch_names=prebads[subj][int(session)],
            status="bad",
            descriptions="prebad",
        )
    # use MNE-BIDS to (re)write the T1, so we can get the side
    # effect of converting the trans file to a JSON sidecar
    t1_fname = mri_dir / subj / "mri" / "T1.mgz"
    trans = mne.read_trans(rawpath.parent / f"prism_{subj}_01-trans.fif")
    landmarks = get_anat_landmarks(
        image=t1_fname,
        info=raw.info,
        trans=trans,
        fs_subject=subj,
        fs_subjects_dir=mri_dir,
    )
    mri_path = BIDSPath(root=bids_root, subject=subj, session=session)
    nii_file = write_anat(
        image=t1_fname, bids_path=mri_path, landmarks=landmarks, overwrite=True
    )
    # write the fine-cal and crosstalk files (once per subject/session)
    cal_path = BIDSPath(root=bids_root, subject=subj, session=session)
    write_meg_calibration(cal_dir / "sss_cal_triux.dat", bids_path=cal_path)
    write_meg_crosstalk(cal_dir / "ct_sparse_triux2.fif", bids_path=cal_path)

    # parse experiment logs and write to disk
    df = parse_expyfun_log(tabpath=tabpath)
    assert events[np.isin(events[:, -1], stim_start_events)].shape[0] == df.shape[0]
    # TODO hack for pilot data, these cols will already be there in future
    if session == "251021":
        df["block"] = ["click_speech"] * 98 + ["imagine_speech"] * 99
    elif session == "251118":
        df["block"] = ["click_music"] * 60 + ["imagine_music"] * 60
    else:
        assert "block" in df.columns
    df.to_csv(trial_info / f"{subj}_{session}_trial_info.csv")
