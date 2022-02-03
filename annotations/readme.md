# Speech corpus

This directory contains all the manual annotations to align each words. It is the gold standard annotations used to evaluate the pipeline.

## How to create the speech corpus

- Firstly, parse every TextGrid and wav files with this format : [0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]\.(TextGrid|wav) from NCCFr into a directory `X`.
- Secondly, run `script/create_speech_corpus.py`. It takes three arguments : Path to the directory `X`, path to the annotations (`../textgrid` in this case), path to the desired output directory `Y`

The directory `Y` will contain multiple pairs of TextGrid containing the full sentence with its associated wav file.