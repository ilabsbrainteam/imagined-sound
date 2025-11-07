"""Create BIDS folder structure for "prism" data."""

import re
import yaml

from pathlib import Path
from warnings import filterwarnings

import mne

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

# metadata
with open(metadata / "daysback.yaml") as fid:
    DAYSBACK = yaml.safe_load(fid)

with open(metadata / "bad-channels.yaml") as fid:
    prebads = yaml.safe_load(fid)

read_raw_kw = dict(preload=False, allow_maxshield=True)
bids_path = BIDSPath(root=bids_root, datatype="eeg", suffix="eeg", extension=".eeg")

#
data_folder_pattern = re.compile(r"\w+/\d+")
rec_pattern = re.compile(r"prism_(?P<subj>\w+)_(?P<run>\d+)_raw.fif")
erm_pattern = re.compile(r"prism_(?P<subj>\w+)_erm_raw.fif")

for data_folder in orig_data.rglob("*/*/"):
    _dirpath = data_folder.relative_to(orig_data)
    if not data_folder_pattern.match(str(_dirpath)):
        print(f"skipping folder {_dirpath}")
        continue
    ermpaths = list()
    fpaths = list()
    for _fpath in data_folder.iterdir():
        if recmatch := rec_pattern.match(_fpath.name):
            fpaths.append(_fpath)
        elif ermmatch := erm_pattern.match(_fpath.name):
            ermpaths.append(_fpath)
    assert len(fpaths), f"no data files found in {_dirpath}"
    assert len(ermpaths), f"no ERM files found in {_dirpath}"
    # TODO handle multiple sessions?
    assert len(fpaths) == 1, f"multiple data files found in {_dirpath}"
    assert len(ermpaths) == 1, f"multiple ERM files found in {_dirpath}"
    fpath = fpaths[0]
    ermpath = ermpaths[0]
    # load the raw
    raw = mne.io.read_raw_fif(fpath, **read_raw_kw)
    erm = mne.io.read_raw_fif(ermpath, **read_raw_kw)
