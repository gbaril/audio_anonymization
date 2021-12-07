path = "/media/gbaril/Linux/Data/Nijmegen_Corpus_of_Casual_French/"

from=(
#"${path}textgrid/03-12-07_1.wav"
"${path}textgrid/14-11-07_1.wav"
"${path}textgrid/14-11-07_2.wav"
"${path}textgrid/16-11-07_2.wav"
"${path}textgrid/20-11-07_1.wav"
"${path}textgrid/26-11-07_1.wav"
"${path}textgrid/26-11-07_3.wav"
"${path}textgrid/27-11-07_1.wav"
"${path}textgrid/27-11-07_2.wav"
"${path}textgrid/28-11-07_2.wav"
"${path}textgrid/29-11-07_2.wav"
)

files=(
#"${path}textgrid_cleaned_soft/03-12-07_1.wav"
"${path}textgrid_cleaned_soft/14-11-07_1.wav"
"${path}textgrid_cleaned_soft/14-11-07_2.wav"
"${path}textgrid_cleaned_soft/16-11-07_2.wav"
"${path}textgrid_cleaned_soft/20-11-07_1.wav"
"${path}textgrid_cleaned_soft/26-11-07_1.wav"
"${path}textgrid_cleaned_soft/26-11-07_3.wav"
"${path}textgrid_cleaned_soft/27-11-07_1.wav"
"${path}textgrid_cleaned_soft/27-11-07_2.wav"
"${path}textgrid_cleaned_soft/28-11-07_2.wav"
"${path}textgrid_cleaned_soft/29-11-07_2.wav"
)

boost=(
#"20dB"
"10dB"
"10dB"
"15dB"
"5dB"
"15dB"
"5dB"
"5dB"
"10dB"
"10dB"
"5dB"
)

for i in {0..10}
do
	ffmpeg -y -i ${from[$i]} -filter:a "volume=${boost[$i]}" ${files[$i]}
done
