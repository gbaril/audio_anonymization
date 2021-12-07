#!/bin/bash
# Convert every .wav to stereo in current directory

for file in *.wav; do
    NEW_FILE="${file%.*}.stereo.wav"
    mv $file $NEW_FILE
    ffmpeg -i $NEW_FILE -ac 1 $file
    rm $NEW_FILE
done
exit 0
