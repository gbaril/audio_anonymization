from argparse import ArgumentParser
from pympi.Praat import TextGrid
from pathlib import Path
from re import sub
from string import punctuation

REPLACE_BY_SPACE = punctuation.replace('[', '').replace(']', '').replace('{', '').replace('}', '')

def is_noisy_or_empty(phrase):
    return any(noise in phrase for noise in punctuation) or len(phrase) == 0

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

    return new if len(new) > 0 else None

def clean(textgrid: TextGrid, filepath):
    for tier in textgrid.get_tiers():
        to_clean = []

        for interval in tier.get_intervals():
            if is_noisy_or_empty(interval[2]):
                to_clean.append(interval)

        for begin, end, phrase in to_clean:
            tier.remove_interval((begin + end) / 2)
            cleaned_phrase = clean_phrase(phrase)
            if cleaned_phrase is not None:
                tier.add_interval(begin, end, cleaned_phrase)
    
    textgrid.to_file(filepath.absolute())

if __name__ == "__main__":
    this_file = Path(__file__)
    
    parser = ArgumentParser(this_file.name)
    parser.add_argument('textgrid', metavar='textgrid_dir', help='Path to textgrid directory')
    parser.add_argument('-s', '--soft', help='Time in minutes to sample', action='store_true')
    args = parser.parse_args()

    if args.soft:
        REPLACE_BY_SPACE = '#&*<>_|~()'

    data_dir = this_file.parent.joinpath(args.textgrid)
    for filepath in data_dir.glob('*.TextGrid'):
        clean(TextGrid(filepath.absolute()), filepath)