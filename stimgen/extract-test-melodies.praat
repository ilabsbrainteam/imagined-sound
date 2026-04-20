stimdir$ = "~/Documents/academics/research/ilabs/prism/experiment/stimuli/music/"
outdir$ = "~/Documents/academics/research/ilabs/prism/stimgen/test_music/"
mel_file$ = "~/Documents/academics/research/ilabs/prism/stimgen/metadata/test-melody-indices.yaml"

stimuli = Create Strings as file list: "stims", stimdir$ + "*.wav"
n_files = Get number of strings

mel_map = Read Strings from raw text file: mel_file$

for ix to n_files
    # get the stim number and chosen note span
    selectObject: mel_map
    mel_map$ = Get string: ix
    mel_vec$# = splitByWhitespace$# (mel_map$)
    stim_num$ = mid$ (mel_vec$# [1], 2, 3) ; remove quotes and colon
    mel_slice$ = mid$ (mel_vec$# [2], 2, 3) ; remove quotes

    selectObject: stimuli
    fname$ = Get string: ix
    # make sure the audio and stim number match
    assert left$ (fname$, 3) = stim_num$
    outfile$ = outdir$ + fname$

    if fileReadable (outfile$) = 0

        # load file
        stimulus = Read from file: stimdir$ + fname$
        dur = Get total duration

        # open editor
        View & Edit
        editor: stimulus
            # show wideband spectrogram
            Show analyses: "yes", "no", "no", "no", "no", 10
            Spectrogram settings: 0, 3000, 0.02, 70
            Advanced spectrogram settings: 1000, 250, "Fourier", "Gaussian", "yes", 100, 6, 0
            Play: 0.0, dur
        endeditor

        # show pause window (wait for selection)
        repeat
            beginPause: mel_slice$
            clicked = endPause: "Zoom", "Accept", 1
            # zoom, align to zero crossings, and play
            if clicked = 1
                editor: stimulus
                    # zoom to 0.5 s window around word
                    start = Get start of selection
                    end = Get end of selection
                    len = Get length of selection
                    if len > 0.5
                        Zoom to selection
                    else
                        pad = (0.5 - len) / 2
                        Zoom: start - pad, end + pad
                    endif
                endeditor
                zc_start = Get nearest zero crossing: 1, start
                zc_end = Get nearest zero crossing: 1, end
                editor: stimulus
                    Select: zc_start, zc_end
                    Play: zc_start, zc_end
                endeditor
            elif clicked = 2
                # save selection to file
                editor: stimulus
                    Save selected sound as WAV file: outfile$
                endeditor
            endif
        until clicked = 2
        # clean up
        removeObject: stimulus
    endif
endfor
# clean up
removeObject: mel_map
removeObject: stimuli