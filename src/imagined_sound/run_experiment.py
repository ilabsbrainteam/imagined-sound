# Author: Daniel McCloy <dan@mccloy.info>
#
# License: BSD (3-clause)
from pathlib import Path
from types import SimpleNamespace

from expyfun import ExperimentController
from expyfun.visual import FixationDot


# set configuration
fs = 48000.0
response_dur = 5.0
ready_dur = 0.4
response_wait_dur = 0.5

blocks = dict(
    one=SimpleNamespace(
        videos=["test.mpeg", "test.mpeg"], noun="speech", verb="repeating"
    ),
    two=SimpleNamespace(
        videos=["test.mpeg", "test.mpeg"], noun="melody", verb="playing"
    ),
)

paktc = " Press any key to continue."

with ExperimentController(
    "MSA",
    participant="foo",
    session="999",
    output_dir="logs",
    version="1c3e802",
    audio_controller=dict(TYPE="sound_card", SOUND_CARD_BACKEND="pyglet"),
) as ec:
    # we'll need this later
    dot = FixationDot(ec)
    radius = dot._circles[0]._radius

    # welcome instructions
    ec.screen_prompt(f"TODO insert welcome-screen instructions here.{paktc}")

    # loop over blocks
    for block_name, block in blocks.items():
        # block instructions
        ec.screen_prompt(
            f"In this block, listen to the {block.noun}, and when the fixation dot "
            f"changes to green, imagine you're {block.verb} the {block.noun}. When "
            f"you're done imagining the {block.noun}, press any key.{paktc}",
        )
        ec.screen_prompt("Here we go!", max_wait=ready_dur, live_keys=[])

        # loop over stimuli
        for vix, fname in enumerate(block.videos):
            # load video (TODO if audio, just draw dot)
            ec.load_video(str(Path(__file__).parent / "stimuli" / fname))
            ec.video.set_scale("fit")
            ec.identify_trial(ec_id=f"{block_name}_{vix}", ttl_id=[0, 0])
            # play video
            t_start = ec.start_stimulus()  # sends a 1-trigger; "video start"
            t_zero = ec.video.play(audio=True)
            while not ec.video.finished:
                if ec.video.playing:
                    fliptime = ec.flip()
                ec.check_force_quit()
            # log the video playback
            ec.stamp_triggers([4, 8], wait_for_last=False)  # 4, 8 = video over
            elapsed = ec.get_time() - t_zero
            t_diff = elapsed - ec.video.duration
            speed = "faster" if t_diff < 0 else "slower"
            ec.write_data_line(
                f"video {fname} (duration {ec.video.duration}) played in "
                f"{elapsed} seconds ({t_diff} seconds {speed} than expected)"
            )
            # clean up
            ec.stop()
            ec.delete_video()
            # prepare for response
            dot.draw()
            ec.flip()
            ec.wait_secs(response_wait_dur)
            # change dot color and radius at start of response period
            dot.set_colors(["g", "k"])
            dot.set_radius(2 * radius, idx=0, units="pix")
            dot.draw()
            # response period
            r_start = ec.flip()
            ec.stamp_triggers([8, 4], wait_for_last=False)  # 8, 4 = begin response
            pressed, t_press = ec.wait_one_press(
                max_wait=response_dur, relative_to=r_start
            )
            r_end = ec.get_time()
            ec.stamp_triggers([8, 8], wait_for_last=True)  # 8, 8 = end response
            # log the response period
            extra = "" if pressed is None else f" {pressed} pressed after {t_press}."
            msg = (
                f"response period began at {r_start}, ended at {r_end} "
                f"({r_end - r_start} s duration).{extra}"
            )
            ec.write_data_line(extra)
            # restore dot color and radius
            dot.set_radius(radius, idx=0, units="pix")
            dot.set_colors(["w", "k"])
            dot.draw()
            ec.trial_ok()
            ec.wait_secs(0.2)
            ec.flip()
        # block end
        ec.screen_prompt(f"End of block.{paktc}")
