from argparse import ArgumentParser
from pathlib import Path
from pympi.Praat import TextGrid

# take two directory containing textgrid

if __name__ == '__main__':
    this_file = Path(__file__)
    
    parser = ArgumentParser(this_file.name)
    parser.add_argument('output', metavar='output_dir', help='Path to desired output directory')
    parser.add_argument('textgrids', metavar='textgrid_dir', nargs='+', help='Path to textgrid directory')
    args = parser.parse_args()

    textgrid_map = {}

    for textgrid in args.textgrids:
        textgrid_dir = this_file.parent.joinpath(textgrid)
        for textgrid_file in textgrid_dir.glob('*.TextGrid'):
            name = textgrid_file.name
            if name not in textgrid_map:
                textgrid_map[name] = TextGrid(textgrid_file)
            else:
                with open(textgrid_file.absolute(), 'rb') as f:
                    textgrid_map[name].from_file(f, textgrid_map[name].codec)

    output_dir = this_file.parent.joinpath(args.output)
    output_dir.mkdir(exist_ok=True)

    for file_name, textgrid in textgrid_map.items():
        output_file = output_dir.joinpath(file_name).absolute()
        textgrid.to_file(output_file)