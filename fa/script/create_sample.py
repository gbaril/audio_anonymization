from pathlib import Path
from argparse import ArgumentParser
import spacy
from pympi.Praat import TextGrid, Tier
from numpy.random import randint
import sys
from pydub import AudioSegment

TIERS_TO_SKIP = ['background']

class Interval:
    def __init__(self, textgrid, tier, filepath, start, end, phrase, doc):
        self.textgrid = textgrid
        self.tier = tier
        self.filepath = filepath
        self.start = start
        self.end = end
        self.phrase = phrase
        self.doc = doc

    def time_in_minute(self):
        return (self.end - self.start) / 60
    
def intervals_with_named_entities(textgrid: TextGrid, filepath: Path):
    intervals = []
    for tier in textgrid.get_tiers():
        if tier.name in TIERS_TO_SKIP: # skip unwanted tiers
            continue
        for start, end, phrase in tier.get_intervals():
            if len(phrase) < 10 or start < 0.5: # skip empty (or almost empty) phrases
                continue
            doc = nlp(phrase)
            if doc.ents:
                intervals.append(Interval(textgrid, tier, filepath, start, end, phrase, doc))
    return intervals

def uniform_sampling(intervals: list, time_in_minute: int):
    current_time = 0
    picked_intervals = []
    while current_time < time_in_minute and intervals:
        random = randint(len(intervals)) # uniform pick between 0 and number of intervals remaining
        picked_intervals.append(intervals[random])
        current_time += intervals[random].time_in_minute()
        del intervals[random]
    return picked_intervals, current_time

def save_sampling(intervals: list, output_dir: Path):
    counters = {}
    audios = {}
    for interval in intervals:
        name = interval.filepath.stem

        if name not in counters:
            counters[name] = 0
        textgrid_path = output_dir.joinpath(name + '_' + str(counters[name]) + '.TextGrid').absolute()
        wav_path = textgrid_path.with_suffix('.wav')
        counters[name] += 1

        if name not in audios:
            source_wav = interval.filepath.with_suffix('.wav')
            audios[name] = AudioSegment.from_wav(source_wav)
        audio = audios[name]

        # create audio file
        start = (interval.start - 0.5) * 1000
        end = (interval.end + 0.5) * 1000
        small_audio = audio[start:end]
        small_audio.export(wav_path, format='wav')
        
        # create textgrid file
        start = 0.5
        end = small_audio.duration_seconds - 0.5
        textgrid = TextGrid(xmax=end + 0.5)
        textgrid.add_tier('spkr_1_1-trans') # phrase transcript
        tier = textgrid.get_tier('spkr_1_1-trans')
        tier.add_interval(start, end, interval.phrase)
        textgrid.add_tier('spkr_1_1-words') # empty word transcript
        tier = textgrid.get_tier('spkr_1_1-words')
        tier.add_interval(start, end, '')
        textgrid.add_tier('spkr_1_1-entities') # empty entities tier
        tier = textgrid.get_tier('spkr_1_1-entities')
        tier.add_interval(start, end, '')
        textgrid.to_file(textgrid_path)

if __name__ == '__main__':
    this_file = Path(__file__)

    parser = ArgumentParser(this_file.name)
    parser.add_argument('output', metavar='output_dir', help='Path to desired sample directory')
    parser.add_argument('textgrid', metavar='textgrid_dir', help='Path to textgrid directory')
    parser.add_argument('-t', '--time', type=int, help='Time in minutes to sample', default=30)
    args = parser.parse_args()

    output_dir = this_file.parent.joinpath(args.output)
    output_dir.mkdir(exist_ok=False)

    nlp = spacy.load('fr_core_news_sm')

    intervals = []
    textgrid_dir = this_file.parent.joinpath(args.textgrid)

    for filepath in textgrid_dir.glob('*.TextGrid'):
        textgrid = TextGrid(filepath)
        intervals.extend(intervals_with_named_entities(textgrid, filepath))

    if len(intervals) == 0:
        print('No interval found in textgrid directory. Exiting.')
        sys.exit(0)
    
    picked_intervals, time_picked = uniform_sampling(intervals, args.time)

    print('{} intervals totalling {:.2f} minutes were picked.'.format(len(picked_intervals), time_picked))

    # need to do this to reduce memory usage
    intervals_per_textgrid = {}
    for interval in picked_intervals:
        if interval.filepath not in intervals_per_textgrid:
            intervals_per_textgrid[interval.filepath] = []
        intervals_per_textgrid[interval.filepath].append(interval)
    
    for filepath, intervals in intervals_per_textgrid.items():
        save_sampling(intervals, output_dir)
