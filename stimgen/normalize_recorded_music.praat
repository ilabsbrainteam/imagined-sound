stimdir$ = "~/Documents/academics/research/ilabs/prism/stimgen/recorded_music/"
outdir$ = "~/Documents/academics/research/ilabs/prism/experiment/stimuli/music/"

stimuli = Create Strings as file list: "stims", stimdir$ + "*.wav"
n_files = Get number of strings

for ix to n_files
    selectObject: stimuli
    fname$ = Get string: ix

    # load file
    stimulus = Read from file: stimdir$ + fname$
    monowav = Convert to mono

    # RMS normalize
    rms = Get root-mean-square: 0, 0
    Multiply: 0.01 / rms

    Save as WAV file: outdir$ + fname$

    # clean up
    removeObject: stimulus
    removeObject: monowav
endfor
removeObject: stimuli
