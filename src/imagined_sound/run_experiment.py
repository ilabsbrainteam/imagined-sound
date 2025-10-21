# Author: Daniel McCloy <dan@mccloy.info>
#
# License: BSD (3-clause)
import re
import yaml

from pathlib import Path

import numpy as np

from expyfun import ExperimentController
from expyfun.stimuli import read_wav, rms
from expyfun.visual import FixationDot

# are we running in the MSR at the MEG center, or piloting elsewhere?
msr = True
yes = 1 if msr else "y"
no = 4 if msr else "n"
live_keys = [yes, no]

# set timing parameters
block_start_delay = 0.5
feedback_dur = 0.5
inter_trial_interval = 1.0
n_practice = 5
resp_duration_multiplier = 2.0  # multiplied by stimulus duration to get max timeout
post_response_delay = 0.1  # 100 ms

# random number generator
rng = np.random.default_rng(seed=8675309)

# load textual prompts for the participant
with open("prompts.yaml") as fid:
    prompts = yaml.safe_load(fid)
prompts = {k: " ".join(v.strip().split("\n")) for k, v in prompts.items()}
resp = "button" if msr else "key"
paktc = f" Press any {resp} to continue."

# load stimulus lists
with open("block_stims.yaml") as fid:
    block_stims = yaml.safe_load(fid)

# load keywords
with open("keywords.yaml") as fid:
    keywords = yaml.safe_load(fid)

with open(Path("..") / ".." / "tools" / "usable_fakes.yaml") as fid:
    fake_keywords = yaml.safe_load(fid)

# colors
colors = dict(pink=(187, 85, 102, 255), green=(78, 178, 101, 255))
# convert pyglet RGBA (ints in [0 255]) to matplotlib RGBA (floats in [0 1])
colors = {k: tuple(map(lambda x: x / 255, v)) for k, v in colors.items()}

# feedback
always = dict(font_name="DejaVu Sans", wrap=False)
correct = dict(text="✔", color="w", **always)
incorrect = dict(text="✘", color="k", **always)

# needed for behavioral check (to look up keywords)
attn_pattern = re.compile(r"NW[FM]0\d_(?P<sent_id>\d\d-\d\d).wav")

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
    stim_db=65,  # TODO edit as needed for MEG Center
    check_rms=None,
    output_dir="logs",
    version="dev",
) as ec:
    # we'll need this later
    dot = FixationDot(ec)
    radius = dot._circles[0]._radius

    # welcome instructions
    ec.screen_prompt(prompts["welcome"].format(resp=resp))

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
            # (won't actually appear until `dot.draw()` and `ec.flip()`)
            color = "pink" if block_name.startswith("imagine") else "green"
            dot.set_colors([colors[color], "k"])
            dot.set_radius(2 * radius, idx=0, units="pix")

            # triage trial timing depending on which block we're in
            if block_name.startswith("imagine"):
                pre_response_delay = 0.1  # short delay, then long time to imagine
                max_wait = stim_duration * resp_duration_multiplier
            else:
                pre_response_delay = stim_duration  # long time before buttonpress
                pre_response_delay += rng.uniform(low=0.0, high=0.4)  # 400ms jitter
                max_wait = 1.5

            # check for buttonpress during pre-response delay (so we know the
            # "do-nothing" period isn't contaminated by motor activity)
            pressed, t_press = ec.wait_one_press(
                max_wait=pre_response_delay, timestamp=True
            )
            if pressed:
                # they responded too quickly; maybe give feedback
                t_response_start = t_response_end = np.nan
                if practice:
                    feedback_kwargs = incorrect | dict(color=colors["pink"])
                    ec.screen_text(**feedback_kwargs, font_size=48)
                    ec.screen_text(
                        "too fast", pos=(0, -0.075), wrap=False, font_size=18
                    )
                    _ = ec.flip()
                    ec.wait_secs(feedback_dur)
                else:
                    # show the dot briefly, so we don't give away the fact that the
                    # press was early (by not showing the dot)
                    dot.draw()
                    _ = ec.flip()
                    ec.wait_secs(post_response_delay)
            else:  # they waited for the response cue
                dot.draw()
                # response period
                t_response_start = ec.flip()
                ec.stamp_triggers([8, 4], wait_for_last=False)  # 8, 4 = begin response
                pressed, t_press = ec.wait_one_press(max_wait=max_wait)
                t_response_end = t_press or ec.get_time()
                ec.stamp_triggers([8, 8], wait_for_last=True)  # 8, 8 = end response

                # feedback
                if practice:
                    if pressed:
                        feedback_kwargs = correct
                    else:
                        feedback_kwargs = incorrect
                        ec.screen_text(
                            "too slow", pos=(0, -0.075), wrap=False, font_size=18
                        )
                    dot.draw()
                    ec.screen_text(**feedback_kwargs)
                    _ = ec.flip(when=t_response_end + post_response_delay)
                    ec.wait_secs(feedback_dur)

            # logging
            ec.write_data_line("stimulus", value=stim_fname, timestamp=t_stim_start)
            ec.write_data_line("stimulus", value="duration", timestamp=stim_duration)
            ec.write_data_line("response", value="start", timestamp=t_response_start)
            ec.write_data_line("response", value="end", timestamp=t_response_end)
            ec.write_data_line("response", value="press", timestamp=t_press or np.nan)

            # attention check
            if block_name.startswith("imagine") and (
                ix % 3 == 0 or (practice and ix % 2 == 0)
            ):
                fake = bool(rng.choice(2))
                if fake:
                    keyword = fake_keywords.pop()
                else:
                    stim_id = attn_pattern.match(stim_fname).group("sent_id")
                    keyword = keywords[stim_id]
                attn_press, attn_time = ec.screen_prompt(
                    f'{{.align "center"}}Did you hear the word "{keyword}"?\n\nPress Y or N.',
                    live_keys=live_keys,
                    timestamp=True,
                )
                # True if pressed Y & it was real, or if pressed N & it was fake
                correct_response = (attn_press.lower() == yes) != fake
                if practice:
                    if correct_response:
                        feedback_kwargs = correct | dict(color=colors["green"])
                    else:
                        feedback_kwargs = incorrect | dict(color=colors["pink"])
                    ec.screen_text(**feedback_kwargs, font_size=48)
                    _ = ec.flip()
                    ec.wait_secs(feedback_dur)

                ec.write_data_line("attn_is_fake", value=fake, timestamp=None)
                ec.write_data_line("attn_keyword", value=keyword, timestamp=None)
                ec.write_data_line(
                    "attn_correct", value=correct_response, timestamp=attn_time
                )

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

            # end trial
            ec.trial_ok()
            ec.wait_secs(inter_trial_interval)

            # periodic rest breaks
            if ix > n_practice and (ix - n_practice) % 10 == 0:
                ec.screen_prompt(
                    f"Rest break! When you're ready to go on,{paktc.lower()}"
                )
