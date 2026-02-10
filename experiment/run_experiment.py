# Author: Daniel McCloy <dan@mccloy.info>
#
# License: BSD (3-clause)
import re
import yaml

from pathlib import Path

import numpy as np

from expyfun import ExperimentController, decimals_to_binary
from expyfun.stimuli import read_wav, rms
from expyfun.visual import FixationDot

# are we running in the MSR at the MEG center, or piloting elsewhere?
msr = True
yes = "1" if msr else "y"
no = "2" if msr else "n"  # yes,no should be adjacent buttons. could also be 3,4
live_keys = [yes, no]

# paths
project_root = Path(__file__).parents[1]
stim_metadata_dir = project_root / "stimgen" / "metadata"
stim_file_dir = Path(__file__).parent / "stimuli"

# set timing parameters
block_start_delay = 0.5
feedback_dur = 0.6
inter_trial_interval = 1.0
n_practice = 5
resp_duration_multiplier = 2.0  # multiplied by stimulus duration to get max timeout
pre_response_delay = 1.0  # after "did you hear..." and before test stim starts
post_response_delay = 0.1  # 100 ms

# offset to compensate for MSR's projector (Epson)
center_offset = np.array([0.14, -0.05])

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

with open(stim_metadata_dir / "usable_fakes.yaml") as fid:
    fake_keywords = yaml.safe_load(fid)

# colors
colors = dict(pink=(187, 85, 102, 255), green=(78, 178, 101, 255))
# convert pyglet RGBA (ints in [0 255]) to matplotlib RGBA (floats in [0 1])
colors = {k: tuple(map(lambda x: x / 255, v)) for k, v in colors.items()}

# feedback
always = dict(font_name="DejaVu Sans", wrap=False)
correct = dict(text="✔", color="w", **always)
incorrect = dict(text="✘", color="k", **always)
click_cue = dict(text="🖱️", color="w", **always)
imagine_cue = dict(text="💭", color="w", **always)

# trigger map
trial_ids = dict(
    real=0,
    practice=1,
    speech=0,
    music=2,
    click=4,
    imagine=8,
    stim_stop=12,
    response_start=13,
    response_end=14,
)
# real speech click → 4
# real speech imag  → 8
# real music click  → 6
# real music imag   → 10
# prac speech click → 5
# prac speech imag  → 9
# prac music click  → 7
# prac music imag   → 11
#
# 2, 3, 15 available for other uses

# needed for behavioral check (to look up keywords)
attn_pattern = re.compile(r"NW[FM]0\d_(?P<sent_id>\d\d-\d\d).wav")

# gather up all the bits that differ between blocks
blocks = {
    k: dict(prompt=v)
    for k, v in prompts.items()
    if k != "welcome" and not k.endswith("practice")
}
# add in the practice instructions
_ = [
    blocks[k.removesuffix("_practice")].update(practice=v)
    for k, v in prompts.items()
    if k.endswith("practice")
]
# add the stimulus lists to each block
_ = [blocks[k].update(stims=v) for k, v in block_stims.items() if k in blocks]

# operator instructions
print("Enter session=1 for speech first, session=2 for music first")

# edit stim_db as needed for MEG Center
sub_ses = dict(stim_db=70) if msr else dict(participant="foo", session="1", stim_db=65)
with ExperimentController(
    "prism",
    stim_fs=44100,
    stim_rms=0.01,
    check_rms=None,
    output_dir="logs",
    version="dev",
    **sub_ses,
) as ec:
    if ec.session == "1":
        block_order = ["click_speech", "click_music", "imagine_speech", "imagine_music"]
    elif ec.session == "2":
        block_order = ["click_music", "click_speech", "imagine_music", "imagine_speech"]
    else:
        raise ValueError(f"bad session, expected 1 or 2, got {ec.session}")

    # we'll need this later
    dot = FixationDot(ec)
    radius = dot._circles[0]._radius

    # welcome instructions
    ec.screen_prompt(prompts["welcome"].format(resp=resp))

    # loop over blocks
    for block_name in block_order:
        block = blocks[block_name]
        stim_folder = block_name.partition("_")[-1]

        # pre-select attention-check trials
        test_trial_indices = np.concatenate(
            (
                np.arange(1, n_practice, 2),  # every 2nd trial during practice
                np.arange(n_practice, len(block["stims"]), 4),  # every 4th trial
            )
        )
        test_trial_jitter = rng.choice([-1, 0, 1], size=len(test_trial_indices))
        test_trial_jitter[::2] = 0  # only (maybe) jitter every-other trial
        test_trial_jitter[:n_practice] = 0  # don't jitter during practice
        test_trial_indices += test_trial_jitter
        test_trials = np.array(block["stims"])[test_trial_indices]
        non_test_trials = sorted(set(block["stims"]) - set(test_trials.tolist()))

        # initial instructions
        ec.screen_prompt(block["prompt"] + paktc)
        ec.screen_prompt(
            f"First let's do {n_practice} practice trials (with feedback). "
            f"Remember, {block['practice']}{paktc}"
        )
        practice = True
        ec.screen_prompt("Here we go!", max_wait=block_start_delay, live_keys=[])

        for ix, stim_fname in enumerate(block["stims"], start=1):
            # load the audio file
            data, fs = read_wav(stim_file_dir / stim_folder / stim_fname)
            assert fs == 44100, "bad stimulus sampling frequency"
            rms_data = 0.01 * data / rms(data)

            # identify the trial
            is_real = "practice" if practice else "real"
            is_click = "click" if block_name.startswith("click") else "imagine"
            is_music = "music" if block_name.endswith("music") else "speech"
            trial_id = decimals_to_binary(
                [sum([trial_ids[n] for n in (is_real, is_music, is_click)])], [4]
            )
            ec.identify_trial(ec_id=f"{stim_fname}", ttl_id=trial_id)

            # bugfix: stimuli are twice as long as they should be
            if is_music == "music":
                rms_data = rms_data[..., : rms_data.shape[-1] // 2]

            ec.load_buffer(rms_data)
            stim_duration = rms_data.shape[-1] / fs
            dot.draw()

            # start the trial
            t_stim_start = ec.start_stimulus()  # sends a 1-trigger; "sentence start"
            ec.wait_secs(stim_duration)
            ec.stop()
            ec.stamp_triggers(trial_ids["stim_stop"], check="int4", wait_for_last=False)

            # larger, colored fixation dot during response period
            # (won't actually appear until `dot.draw()` and `ec.flip()`)
            # TODO need to add icon/emoji as reminder of the task
            #      (imagine+click or just click)
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
                    ec.screen_text(**feedback_kwargs, font_size=48, pos=center_offset)
                    ec.screen_text(
                        "too fast",
                        pos=(0, -0.075) + center_offset,
                        wrap=False,
                        font_size=18,
                    )
                    _ = ec.flip()
                    ec.wait_secs(feedback_dur)
                    # TODO: force a buttonpress to advance, when response was too quick?
                    #       or just make `feedback_dur` longer?
                else:
                    # show the dot briefly, so we don't give away the fact that the
                    # press was early (by not showing the dot)
                    dot.draw()
                    _ = ec.flip()
                    ec.wait_secs(post_response_delay)
                # TODO maybe stamp triggers here anyway, even though they already
                #      pressed (and thus contaminated the trial) and we don't really
                #      care about the actual post-cue reaction time?
            else:  # they waited for the response cue
                dot.draw()
                # response period
                t_response_start = ec.flip()
                ec.stamp_triggers(
                    trial_ids["response_start"], check="int4", wait_for_last=False
                )
                pressed, t_press = ec.wait_one_press(max_wait=max_wait)
                t_response_end = t_press or ec.get_time()
                ec.stamp_triggers(
                    trial_ids["response_end"], check="int4", wait_for_last=True
                )

                # feedback
                # TODO: debug feedback appearing early during practice of imagine block
                #       (could it have been an early button press?)
                if practice:
                    if pressed:
                        feedback_kwargs = correct
                        extra_feedback_dur = 0.0
                    else:
                        feedback_kwargs = incorrect
                        ec.screen_text(
                            "too slow",
                            pos=(0, -0.075) + center_offset,
                            wrap=False,
                            font_size=18,
                        )
                        extra_feedback_dur = 0.25
                    dot.draw()
                    ec.screen_text(**feedback_kwargs, pos=center_offset)
                    _ = ec.flip(when=t_response_end + post_response_delay)
                    ec.wait_secs(feedback_dur + extra_feedback_dur)

            # logging
            ec.write_data_line("block", value=block_name)
            ec.write_data_line("practice", value=practice)
            ec.write_data_line("stimulus", value=stim_fname, timestamp=t_stim_start)
            ec.write_data_line("stimulus", value="duration", timestamp=stim_duration)
            ec.write_data_line("response", value="start", timestamp=t_response_start)
            ec.write_data_line("response", value="end", timestamp=t_response_end)
            ec.write_data_line("response", value="press", timestamp=t_press or np.nan)

            # attention check
            if stim_fname in test_trials and is_click == "imagine":
                fake = bool(rng.choice(2))
                if is_music == "music":
                    keyword = non_test_trials.pop(-1) if fake else stim_fname
                    # load the audio file
                    data, fs = read_wav(stim_file_dir / "test_music" / keyword)
                    assert fs == 44100, "bad stimulus sampling frequency"
                    rms_data = 0.01 * data / rms(data)
                    # bugfix: stimuli are twice as long as they should be
                    rms_data = rms_data[..., : rms_data.shape[-1] // 2]
                    ec.load_buffer(rms_data)
                    stim_duration = rms_data.shape[-1] / fs
                    ec.screen_text(
                        '{.align "center"}Did you hear these notes?\n\nPress Y or N.',
                    )
                    ec.flip()
                    ec.wait_secs(pre_response_delay)
                    test_start = ec.start_stimulus(start_of_trial=False, flip=False)
                    attn_press, attn_time = ec.wait_one_press(
                        live_keys=live_keys,
                        timestamp=True,
                        min_wait=stim_duration,
                    )
                    ec.stop()
                else:  # speech
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
                    ec.screen_text(**feedback_kwargs, font_size=48, pos=center_offset)
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
