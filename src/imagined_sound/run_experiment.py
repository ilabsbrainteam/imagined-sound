# Author: Daniel McCloy <dan@mccloy.info>
#
# License: BSD (3-clause)
import yaml

from pathlib import Path

import numpy as np

from expyfun import ExperimentController
from expyfun.stimuli import read_wav, rms
from expyfun.visual import FixationDot

# set timing parameters
block_start_delay = 0.5
feedback_dur = 0.3
inter_trial_interval = 1.0
n_practice = 4
resp_duration_multiplier = 1.5  # multiplied by stimulus duration to get max timeout

# random number generator
rng = np.random.default_rng(seed=8675309)

# load textual prompts for the participant
with open("prompts.yaml") as fid:
    prompts = yaml.safe_load(fid)
prompts = {k: " ".join(v.strip().split("\n")) for k, v in prompts.items()}
paktc = " Press any key to continue."

# load stimulus lists
with open("block_stims.yaml") as fid:
    block_stims = yaml.safe_load(fid)

# colors
color = (187, 85, 102, 255)  # pink

# gather up all the bits that differ between blocks
blocks = {
    k: dict(prompt=v)
    for k, v in prompts.items()
    if k != "welcome" and not k.endswith("practice")
}
_ = [
    blocks[k.removesuffix("_practice")].update(practice=v)
    for k, v in prompts.items()
    if k.endswith("practice")
]
_ = [blocks[k].update(stims=v) for k, v in block_stims.items() if k in blocks]

with ExperimentController(
    "imagined-sound",
    participant="foo",
    session="999",
    stim_fs=44100,
    stim_rms=0.01,
    check_rms=None,
    output_dir="logs",
    version="dev",
) as ec:
    # we'll need this later
    dot = FixationDot(ec)
    radius = dot._circles[0]._radius

    # welcome instructions
    ec.screen_prompt(prompts["welcome"])

    # loop over blocks
    for block_name, block in blocks.items():
        ec.screen_prompt(block["prompt"] + paktc)
        ec.screen_prompt(
            f"First let's do {n_practice} practice trials (with feedback). "
            f"Remember, {block['practice']}{paktc}"
        )
        practice = True
        ec.screen_prompt("Here we go!", max_wait=block_start_delay, live_keys=[])

        for ix, stim_fname in enumerate(block["stims"], start=1):
            # load the audio file
            data, fs = read_wav(Path("stimuli") / "NWF003" / stim_fname)
            assert fs == 44100, "bad stimulus sampling frequency"
            rms_data = 0.01 * data / rms(data)
            ec.load_buffer(rms_data)
            stim_duration = rms_data.shape[-1] / fs
            dot.draw()

            # start the trial
            ec.identify_trial(ec_id=f"{stim_fname}", ttl_id=[0, 0])
            t_stim_start = ec.start_stimulus()  # sends a 1-trigger; "sentence start"
            ec.wait_secs(stim_duration)
            ec.stop()
            ec.stamp_triggers([4, 8], wait_for_last=False)  # 4, 8 = audio over

            # larger, colored fixation dot during response period
            # convert pyglet RGBA (ints in [0 255]) to matplotlib RGBA (floats in [0 1])
            dot_col = tuple(map(lambda x: x / 255, color))
            dot.set_colors([dot_col, "k"])
            dot.set_radius(2 * radius, idx=0, units="pix")
            dot.draw()  # won't actually change until next flip

            # triage trial timing depending on which block we're in
            if block_name.startswith("imagine"):
                pre_response_delay = 0.4  # short delay, then long time to imagine
                max_wait = stim_duration * resp_duration_multiplier
            else:
                pre_response_delay = stim_duration  # long time before buttonpress
                max_wait = 1.5
            pre_response_delay += rng.uniform(low=0.0, high=0.4)  # 400ms jitter
            ec.wait_secs(pre_response_delay)
            # TODO get presses during pre-response delay, to make sure the "rest" period
            # isn't contaminated by motor activity
            t_response_start = ec.flip()

            # response period
            ec.stamp_triggers([8, 4], wait_for_last=False)  # 8, 4 = begin response
            pressed, t_press = ec.wait_one_press(
                min_wait=0.2,  # shouldn't be anyone faster than that...
                max_wait=max_wait,
                relative_to=t_response_start,
            )
            t_response_end = t_press or ec.get_time()
            ec.stamp_triggers([8, 8], wait_for_last=True)  # 8, 8 = end response

            # feedback
            if practice:
                if pressed:
                    feedback = "✔"
                    feedback_color = "w"
                    dur = feedback_dur
                else:
                    feedback = "✘"
                    feedback_color = "k"
                    dur = feedback_dur + 0.5
                    ec.screen_text(
                        "too slow", pos=(0, -0.075), wrap=False, font_size=18
                    )
                dot.draw()
                ec.screen_text(
                    feedback, color=feedback_color, font_name="DejaVu Sans", wrap=False
                )
                _ = ec.flip()
                ec.wait_secs(feedback_dur)
            # transition from "practice" to "real"
            if ix == n_practice:
                ec.screen_prompt(
                    "OK, done with practice, so no more feedback. "
                    f"Remember, {block['practice']}{paktc}"
                )
                practice = False

            # restore dot color and radius
            dot.set_radius(radius, idx=0, units="pix")
            dot.set_colors(["w", "k"])
            dot.draw()
            _ = ec.flip()

            # logging and end trial
            ec.write_data_line("stimulus", value=stim_fname, timestamp=t_stim_start)
            ec.write_data_line("stimulus", value="duration", timestamp=stim_duration)
            ec.write_data_line("response", value="start", timestamp=t_response_start)
            ec.write_data_line("response", value="end", timestamp=t_response_end)
            ec.write_data_line("response", value="press", timestamp=t_press or np.nan)
            ec.trial_ok()
            ec.wait_secs(inter_trial_interval)
