from pympi.Elan import Eaf
from argparse import ArgumentParser
from pathlib import Path
from re import compile
from shutil import copy2

TIER_REGEX = compile('^spkr_.*_.*$')

def eaf_to_textgrid_path(eaf_file: Path, textgrid_dir: Path):
    return textgrid_dir.joinpath(eaf_file.name).with_suffix('.TextGrid')

def cp_wav_file(eaf_file: Path, textgrid_dir: str):
    eaf = eaf_file.with_suffix('.wav')
    copy2(eaf, textgrid_dir)

def to_textgrid(eaf: Eaf, textgrid_file: str):
    # Filter tiers containing only phrases
    tiers_to_keep = []
    for tier in eaf.get_tier_names():
        if TIER_REGEX.match(tier):
            tiers_to_keep.append(tier)
    textgrid = eaf.to_textgrid(filtin=tiers_to_keep)

    for num, name in textgrid.get_tier_name_num():
        textgrid.change_tier_name(name, name + '-trans')

    textgrid.to_file(textgrid_file)

if __name__ == "__main__":
    this_file = Path(__file__)
    
    parser = ArgumentParser(this_file.name)
    parser.add_argument('textgrid', metavar='textgrid_dir', help='Path to textgrid directory')
    parser.add_argument('eaf', metavar='eaf_dir', help='Path to eaf directory')
    args = parser.parse_args()

    eaf_dir = this_file.parent.joinpath(args.eaf)
    textgrid_dir = this_file.parent.joinpath(args.textgrid)
    textgrid_dir.mkdir(exist_ok=True)
    
    for eaf_file in eaf_dir.glob('*.eaf'):
        textgrid_file = eaf_to_textgrid_path(eaf_file, textgrid_dir)
        to_textgrid(Eaf(eaf_file.absolute()), textgrid_file)
        cp_wav_file(eaf_file, textgrid_dir)