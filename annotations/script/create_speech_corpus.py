from pathlib import Path
from argparse import ArgumentParser
from typing import Text
import spacy
from pympi.Praat import TextGrid, Tier
from numpy.random import randint
import sys
from re import sub
from pydub import AudioSegment
from string import punctuation
from fuzzywuzzy import fuzz

TIERS_TO_SKIP = ['background']
REPLACE_BY_SPACE = punctuation.replace('[', '').replace(']', '').replace('{', '').replace('}', '')

audios = {}

def clean_phrase(phrase: str):
    if len(phrase) == 0:
        return None
    # replace punctuation with space
    new = sub('[' + REPLACE_BY_SPACE + ']', ' ', phrase)
    # remove stuff between brackets
    new = sub('[\[\{].*?[\}\]]', ' ', new)
    # remove multiple spaces
    while '  ' in new:
        new = new.replace('  ', ' ')
    new = new.strip()

    return new.lower() if len(new) > 0 else None

def get_intervals(textgrid: TextGrid):
    intervals = {}
    for tier in textgrid.get_tiers():
        if tier.name in TIERS_TO_SKIP: # skip unwanted tiers
            continue
        for start, end, phrase in tier.get_intervals():
            phrase = clean_phrase(phrase)
            if phrase is not None:
                intervals[phrase] = (start, end)
    return intervals

def get_phrase(tier: Tier):
    phrase = ''
    for _, _, word in list(tier.get_intervals())[1:-1]:
        phrase += word + ' '
    return clean_phrase(phrase)

def in_here(phrase: str, intervals: dict):
    for key, value in intervals.items():
        if fuzz.ratio(key, phrase) >= 80:
            return value
    return (-1, -1)

def save_example(full_audio: Path, output_audio: Path, begin, end, output_textgrid: Path, phrase):
    if full_audio not in audios:
        audios[full_audio] = AudioSegment.from_wav(full_audio)
    audio = audios[full_audio]
    
    # create audio file
    begin = (begin - 0.5) * 1000
    end = (end + 0.5) * 1000
    small_audio = audio[begin:end]
    small_audio.export(output_audio, format='wav')

     # create textgrid file
    begin = 0.5
    end = small_audio.duration_seconds - 0.5
    textgrid = TextGrid(xmax=end + 0.5)
    textgrid.add_tier('spkr_1_1-trans') # phrase transcript
    tier = textgrid.get_tier('spkr_1_1-trans')
    tier.add_interval(begin, end, phrase)
    textgrid.to_file(output_textgrid)

if __name__ == '__main__':
    this_file = Path(__file__)

    parser = ArgumentParser(this_file.name)
    parser.add_argument('textgrid', metavar='textgrid_dir', help='Path to directory containing the original textgrids')
    parser.add_argument('annotation', metavar='annotation_dir', help='Path to directory containing the annotations')
    parser.add_argument('output', metavar='output_dir', help='Path to output directory')

    args = parser.parse_args()

    textgrid_dir = this_file.parent.joinpath(args.textgrid)
    annotations_dir = this_file.parent.joinpath(args.annotation)
    output_dir = this_file.parent.joinpath(args.output)
    output_dir.mkdir(exist_ok=False)

    intervals = {}

    print("Fetching original textgrids...", end=' ')
    for filepath in textgrid_dir.glob('*.TextGrid'):
        name = filepath.name.split('.')[0]
        textgrid = TextGrid(filepath)
        intervals[name] = {
            'pwd': filepath.parent,
            'intervals': get_intervals(textgrid),
            'textgrid': textgrid
        }
    print('done')

    for filepath in annotations_dir.glob('*.TextGrid'):
        name = filepath.name.split('.')[0]
        ori_name = '_'.join(name.split('_')[0:-1])
        
        textgrid = TextGrid(filepath)
        phrase = get_phrase(textgrid.get_tier('spkr_1_1-words'))

        print(f'Matching {name}...', end=' ')
        begin, end = in_here(phrase, intervals[ori_name]['intervals'])
        
        if begin == end and begin == -1:
            raise Exception(f"Error, {name} not found")
        else:
            print('found...', end=' ')
            full_audio = intervals[ori_name]['pwd'].joinpath(ori_name + '.wav').absolute()
            output_audio = output_dir.joinpath(name + '.wav').absolute()
            output_textgrid = output_dir.joinpath(name + '.TextGrid').absolute()
            save_example(full_audio, output_audio, begin, end, output_textgrid, phrase)
            print('saved!')