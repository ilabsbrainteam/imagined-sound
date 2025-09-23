# Author: Daniel McCloy <dan@mccloy.info>
#
# License: BSD (3-clause)
import yaml

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from expyfun import ExperimentController
from expyfun.stimuli import read_wav, rms
from expyfun.visual import FixationDot

# set timing parameters
block_start_delay = 0.4
max_response_dur = 4.0
# pre_response_delay = 0.8  # defined differently per block
inter_trial_interval = 1.2

# random number generator
rng = np.random.default_rng(seed=8675309)

# load textual prompts for the participant
with open("prompts.yaml") as fid:
    prompts = yaml.safe_load(fid)
prompts = SimpleNamespace(
    **{k: " ".join(v.strip().split("\n")) for k, v in prompts.items()}
)
paktc = " Press any key to continue."

# load stimulus lists
with open("blocks.yaml") as fid:
    blocks = yaml.safe_load(fid)
blocks = SimpleNamespace(**blocks)

# colors
colors = dict(
    pink=(187, 85, 102, 255),
    yellow=(221, 170, 51, 255),
)
white = (255, 255, 255, 255)
color_text = {
    name: f"{{color {col}}}{name}{{color {white}}}" for name, col in colors.items()
}
for col, text in color_text.items():
    prompts.only_click = prompts.only_click.replace(col, text)
    prompts.imagine_click_some = prompts.imagine_click_some.replace(col, text)
    prompts.imagine_click_all = prompts.imagine_click_all.replace(col, text)

with ExperimentController(
    "imagined-sound",
    participant="foo",
    session="999",
    stim_fs=44100,
    stim_rms=0.01,
    output_dir="logs",
    version="dev",
) as ec:
    # we'll need this later
    dot = FixationDot(ec)
    radius = dot._circles[0]._radius

    # welcome instructions
    ec.screen_prompt(prompts.welcome)

    # first block: only click (no imagining)
    ec.screen_prompt(prompts.only_click + paktc)
    ec.screen_prompt("Here we go!", max_wait=block_start_delay, live_keys=[])
    pre_response_delay = 0.8  # seconds

    for ix, stim_fname in enumerate(blocks.only_click, start=1):
        # is this a button-press trial or not?
        expect_press = ix % 3 == 0
        # load the audio file
        data, fs = read_wav(Path("stimuli") / "NWF003" / stim_fname)
        assert fs == 44100, "bad stimulus sampling frequency"
        rms_data = 0.01 * data / rms(data)
        ec.load_buffer(rms_data)
        duration = rms_data.shape[-1] / fs
        # light gray fixation dot during stimulus (TODO: might be confusing?)
        dot.set_colors(["0.9", "k"])
        dot.draw()
        # start the trial
        ec.identify_trial(ec_id=f"{stim_fname}", ttl_id=[0, 0])
        t_stim_start = ec.start_stimulus()  # sends a 1-trigger; "sentence start"
        ec.wait_secs(duration)
        ec.stop()
        ec.stamp_triggers([4, 8], wait_for_last=False)  # 4, 8 = audio over
        # white fixation dot during prepare-for-response period
        dot.set_colors(["w", "k"])
        dot.draw()
        _ = ec.flip()
        # larger, colored fixation dot during response period
        # convert pyglet RGBA (ints in [0 255]) to matplotlib RGBA (floats in [0 1])
        color = colors["pink"] if expect_press else colors["yellow"]
        dot_col = tuple(map(lambda x: x / 255, color))
        dot.set_colors([dot_col, "k"])
        dot.set_radius(2 * radius, idx=0, units="pix")
        dot.draw()  # won't actually change until next flip
        pre_response_jitter = rng.uniform(low=0.0, high=0.4)
        ec.wait_secs(pre_response_delay + pre_response_jitter)
        t_response_start = ec.flip()
        # response period
        ec.stamp_triggers([8, 4], wait_for_last=False)  # 8, 4 = begin response
        press_kwargs = dict(max_wait=max_response_dur, relative_to=t_response_start)
        if expect_press:
            pressed, t_press = ec.wait_one_press(**press_kwargs)
            t_response_end = t_press or ec.get_time()
        else:
            presses_and_timestamps = ec.wait_for_presses(**press_kwargs)
            if presses_and_timestamps:
                pressed, t_press = list(zip(*presses_and_timestamps))
            else:
                pressed, t_press = (), ()
            t_response_end = ec.get_time()
        ec.stamp_triggers([8, 8], wait_for_last=True)  # 8, 8 = end response
        # restore dot color and radius
        dot.set_radius(radius, idx=0, units="pix")
        dot.set_colors(["w", "k"])
        dot.draw()
        _ = ec.flip()
        # logging and end trial
        msg = (
            f"stimulus began at {t_stim_start}. "
            f"Response period began at {t_response_start}, ended at {t_response_end} "
            f"({t_response_end - t_response_start} s duration). "
            f"Button press at {t_press or '<NONE>'}"
        )
        ec.write_data_line(msg)
        ec.trial_ok()
        ec.wait_secs(inter_trial_interval)
