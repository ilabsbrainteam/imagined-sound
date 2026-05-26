#!/usr/bin/env python
import subprocess


def midi_to_wav(midi_path, wav_path):
    subprocess.run(
        [
            "timidity",
            "--quiet=2",
            # append ↓ 8,1,2: 8,16,24-bits, u/s: (un)signed, l(inear), M(ono)/S(tereo)
            "--output-mode=wM",
            f"--output-file={wav_path}",
            str(midi_path),
            "-EFreverb=0",
            "--sampling-freq=44100",
        ],
        check=True,
        timeout=10,
    )
