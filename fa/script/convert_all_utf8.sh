#!/bin/bash
# Convert every .TextGrid in directory to UTF-8

rm *.utf8.TextGrid

for  file  in  *.TextGrid; do
	ENCODING=$(uchardet $file)
	CONVERT=" iconv -f $ENCODING -t UTF-8"
	$CONVERT "$file" -o "${file%.*}.utf8.TextGrid"
done
exit 0
