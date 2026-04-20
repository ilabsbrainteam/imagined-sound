# make sure stims are mono and have RMS energy of 0.01
input_dir$ = "/data/corpora/pnnc/2011/NWF004/"
output_dir$ = "~/Documents/academics/research/ilabs/prism/stimgen/speech/NWF004/"
raw_stims$# = fileNames$#: input_dir$ + "*.wav"

for ix to size(raw_stims$#)
    path$ = input_dir$ + raw_stims$# [ix]
    orig = Read from file: path$
    mono = Extract one channel: 1
    rms = Get root-mean-square: 0, 0
    Multiply: 0.01 / rms
    outpath$ = output_dir$ + raw_stims$# [ix]
    Save as WAV file: outpath$
    # clean up
    selectObject: orig
    plusObject: mono
    Remove
endfor
