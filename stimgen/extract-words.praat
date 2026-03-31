stimdir$ = "~/Documents/academics/research/ilabs/prism/stimgen/speech/NWF004/"
outdir$ = "~/Documents/academics/research/ilabs/prism/stimgen/test_speech/NWF004/"
kwds$ = "~/Documents/academics/research/ilabs/prism/experiment/keywords.yaml"

stimuli = Create Strings as file list: "stims", stimdir$ + "*.wav"
n_files = Get number of strings

keyword_map = Read Strings from raw text file: kwds$
; keywords = Replace all: "\d\d-\d\d: ", "", 0, "regular expressions"

for ix to n_files
    # get the sentence number and chosen keyword
    selectObject: keyword_map
    kw_map$ = Get string: ix
    kw_vec$# = splitByWhitespace$# (kw_map$)
    sent$ = truncateRight$ (kw_vec$# [1], 5) ; remove colon
    kw$ = kw_vec$# [2]
    ; selectObject: keywords
    ; kw$ = Get string: ix

    selectObject: stimuli
    fname$ = Get string: ix
    outfile$ = outdir$ + truncateRight$ (fname$, 11) + "_" + kw$ + ".wav"

    # make sure the audio and sentence number match
    assert right$ (truncateRight$ (fname$, 11), 5) = sent$

    if fileReadable (outfile$) = 0

        # load file
        stimulus = Read from file: stimdir$ + fname$
        dur = Get total duration

        # open editor
        View & Edit
        editor: stimulus
            Show analyses: "no", "no", "no", "no", "no", 10 ; hide all except waveform
            Play: 0.0, dur
        endeditor

        # show pause window (wait for selection)
        repeat
            beginPause: kw$
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
removeObject: keyword_map
removeObject: stimuli