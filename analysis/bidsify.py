"""Create BIDS folder structure for "prism" data."""

import re
from pathlib import Path
from warnings import filterwarnings

import mne
import numpy as np
import yaml
from mne_bids import (
    BIDSPath,
    get_anat_landmarks,
    mark_channels,
    write_anat,
    write_meg_calibration,
    write_meg_crosstalk,
    write_raw_bids,
)
from score import (
    EVENT_DICT,
    parse_expyfun_log,
    score_func,
)

EVENT_DICT_DEFAULT = {
    "BAD boundary": 999,
    "EDGE boundary": 998,
}

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

read_raw_kw = dict(preload=False, allow_maxshield="yes")
bids_path = BIDSPath(
    root=bids_root, datatype="meg", suffix="meg", extension=".fif", task="SpeMusImaCli"
)

# filename patterns
data_folder_pattern = re.compile(r"prism_\d+/\d+")
rec_pattern = re.compile(r"prism_(?P<subj>\w+)_(?P<run>\d+)_raw.fif")
# rec_pattern = re.compile(r"prism_(?P<subj>\w+)_(?P<task>\w+)_(?P<run>\d+)_raw.fif")
erm_pattern = re.compile(r"prism_(?P<subj>\w+)_erm_raw.fif")
tab_pattern = re.compile(
    r"(?P<subj>\w+)_\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2}(?:\.\d{6})?.tab"
)
# tab_pattern = re.compile(
#     r"prism_(?P<subj>\w+)_(?P<task>\w+)_\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2}(?:\.\d{6})?.tab"
# )

# task dict
task_dict = dict(
    speechcontrol="SpeechControl",
    speechimagine="SpeechImagine",
    musiccontrol="MusicControl",
    musicimagine="MusicImagine",
)

for data_folder in orig_data.rglob("*/*/"):
    _dirpath = data_folder.relative_to(orig_data)
    if not data_folder_pattern.match(str(_dirpath)):
        print(f"skipping folder {_dirpath}")
        continue
    session = _dirpath.parts[-1]
    # final pilot
    EVENT_DICT |= EVENT_DICT_DEFAULT
    stim_start_events = [
        val for key, val in EVENT_DICT.items() if key.endswith(("click", "imagine"))
    ]
    bids_path.update(session=session)
    ermpaths = list()
    rawpaths = list()
    tabpaths = list()
    for _fpath in data_folder.iterdir():
        if rec_match := rec_pattern.match(_fpath.name):
            subj = rec_match.group("subj")
            rawpaths.append(_fpath)
            bids_path.update(subject=subj)
        elif erm_pattern.match(_fpath.name):
            ermpaths.append(_fpath)
        elif tab_match := tab_pattern.match(_fpath.name):
            subj = tab_match.group("subj")
            tabpaths.append(_fpath)
    assert len(rawpaths), f"no data files found in {_dirpath}"
    assert len(ermpaths), f"no ERM files found in {_dirpath}"
    assert len(tabpaths), f"no experiment TAB files found in {_dirpath}"
    assert len(ermpaths) == 1, f"multiple ERM files found in {_dirpath}"
    assert len(rawpaths) == 1, f"multiple data files found in {_dirpath}"
    assert len(tabpaths) == 1, f"multiple experiment TAB files found in {_dirpath}"
    ermpath = ermpaths[0]
    rawpath = rawpaths[0]
    tabpath = tabpaths[0]
    raws = list()
    dfs = list()
    evs = list()
    # load the raw
    raw = mne.io.read_raw_fif(rawpath, **read_raw_kw)
    erm = mne.io.read_raw_fif(ermpath, **read_raw_kw)
    # extract events
    events = score_func(raw=raw)
    # parse logfile
    df = parse_expyfun_log(tabpath=tabpath)
    # write raw
    write_raw_bids(
        raw=raw,
        events=events,
        event_id=EVENT_DICT,
        bids_path=bids_path,
        empty_room=erm,
        anonymize=dict(daysback=DAYSBACK),
        overwrite=True,
        # ↓ because we concatenate
        format="FIF",
        allow_preload=True,
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
    if t1_fname.exists():
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

    # write experiment logs to disk
    mask = np.isin(events[:, -1], stim_start_events)
    non_finale_trials = df.loc[~df["block"].isin(["finale"])]
    if events[mask].shape[0] != non_finale_trials.shape[0]:
        print(
            f"BADNESS: N events {mask.sum()} doesn't match N trials from TAB {non_finale_trials.shape[0]}"
        )
    df.to_csv(trial_info / f"{subj}_{session}_trial_info.csv")

# expect:
# 5+94 click-speech
# 5+95 click-music
# 5+95 imagine-speech
# 5+96 imagine-music
# 24 finale
