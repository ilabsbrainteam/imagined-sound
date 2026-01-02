# Prism

Experiment for imagined repetition of speech and music.

## Requirements

Managed with `pixi`; most dependencies will be automatically installed when tasks are invoked. Exceptions:

- music stimulus generation (`stimuli/make_stimuli.py`) requires `timidity` and `lilypond`, which are not available through conda-forge. For interactive listening via `music21`'s `Stream.show("midi")` syntax, I also needed `fluidsynth`.
- `rsync` cannot be managed via `pixi` on windows. It's helpful to have it when loading the stimuli onto the experiment runner computer (via `pixi run rsyncstims`). The task `pixi run getstims` is an alternative that uses `scp` instead.

# Setup

`music21` has a somewhat complicated setup process. Running `python -m music21.configure` is supposed to detect various accessory programs, but doesn't always succeed. Here's a partial `~/.music21rc` file that was working for me during development:

```xml
<settings encoding="utf-8">
  <preference name="graphicsPath" value="xdg-open" />
  <preference name="lilypondBackend" value="ps" />
  <preference name="lilypondFormat" value="pdf" />
  <preference name="lilypondPath" value="lilypond" />
  <preference name="lilypondVersion" value="2.24.3"/>
  <preference name="midiPath" value="synth" />
  <preference name="pdfPath" value="xdg-open" />
  </settings>
```

The program `synth` used as the value for `midiPath` is a small shell script somewhere on your `PATH` (e.g., at `~/.local/bin/synth`):

```sh
#!/usr/bin/bash
fluidsynth -q --no-shell $@
```
