from argparse import ArgumentParser
from pathlib import Path
from pympi.Praat import TextGrid, Tier
from shutil import copy2, rmtree

def splitTextGrid(filepath: Path, test_dir: Path, gold_dir: Path):
    textGrid = TextGrid(filepath.absolute())
    testTextGrid = TextGrid(xmax=textGrid.xmax)
    goldTextGrid = TextGrid(xmax=textGrid.xmax)
    
    for tier in textGrid.get_tiers():
        if 'trans' in tier.name:
            testTextGrid.tiers.append(tier)
        else:
            newTier: Tier = goldTextGrid.add_tier(tier.name)
            for start, end, word in tier.get_all_intervals():
                if len(word) > 0: # Remove unnecessary empty intervals
                    newTier.add_interval(start, end, word)
    
    testTextGrid.to_file(new_path(test_dir, filepath))
    goldTextGrid.to_file(new_path(gold_dir, filepath))

def new_path(dirpath: Path, filepath: Path):
    filename = str(filepath.name)[:-13] + str(filepath.name)[-8:]
    return dirpath.joinpath(filename).absolute()

def copy_wav(filepath: Path, dirpath: Path):
    wavpath = str(filepath.absolute())[:-14] + '.wav'
    copy2(wavpath, dirpath)

if __name__ == "__main__":
    this_file = Path(__file__)
    
    parser = ArgumentParser(this_file.name)
    parser.add_argument('annotated', metavar='annotated_dir', help='Path to annotated directory')
    parser.add_argument('output', metavar='output_dir', help='Path to output directory')
    args = parser.parse_args()

    output_dir = this_file.parent.joinpath(args.output)
    
    test_dir = output_dir.joinpath('test')
    rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(exist_ok=True)
    
    gold_dir = output_dir.joinpath('gold')
    rmtree(gold_dir, ignore_errors=True)
    gold_dir.mkdir(exist_ok=True)

    data_dir = this_file.parent.joinpath(args.annotated)
    counter = 0
    for filepath in data_dir.glob('*.utf8.TextGrid'):
        textGrid = TextGrid(filepath.absolute())
        splitTextGrid(filepath, test_dir, gold_dir)
        copy_wav(filepath, test_dir)
        counter += 1
    print(f'Parsed {counter} files.')