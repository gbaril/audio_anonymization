import sys
import os
from config import SPPAS_DIR
import shutil
from pathlib import Path
from pympi.Praat import TextGrid

def split_speakers_sppas(input_dir: str, tmp_dir: str):
    for f in Path(input_dir).glob('*.TextGrid'):
        textgrid = TextGrid(f)
        for tier in textgrid.get_tiers():
            new_textgrid = TextGrid(xmax=textgrid.xmax)
            newTier: Tier = new_textgrid.add_tier(tier.name)
            for start, end, word in tier.get_all_intervals():
                newTier.add_interval(start, end, word)
            
            new_name = tier.name + '__' + f.name
            new_textgrid.to_file(os.path.join(tmp_dir, new_name))
            
            wavpath = str(f.absolute())[:-9] + '.wav'
            new_wavpath = new_name[:-9] + '.wav'
            os.symlink(wavpath, os.path.join(tmp_dir, new_wavpath))

def merge_speakers_sppas(tmp_dir: str, output_dir: str):
    from sppas.src.anndata import sppasRW, sppasTranscription

    speakers_per_file = {}
    parser = sppasRW('')

    for f in Path(tmp_dir).glob('*.TextGrid'):
        path = str(f)
        if 'palign' in path:
            parser.set_filename(path)
            textgrid = parser.read()
            speaker, filename = f.name.split('__')
            filename = filename[:-16] + filename[-9:]
            
            if filename not in speakers_per_file:
                speakers_per_file[filename] = []
            tier = textgrid.find('TokensAlign')
            tier.set_name(speaker + ' - words')
            speakers_per_file[filename].append(tier)

    for f, tiers in speakers_per_file.items():
        textgrid = sppasTranscription()
        for tier in tiers:
            textgrid.append(tier)
        parser.set_filename(os.path.join(output_dir, f))
        parser.write(textgrid)


def align_sppas(input_dir: str, output_dir: str):
    sys.path.append(SPPAS_DIR)
    from sppas import sppasLogSetup
    from sppas.src.annotations import sppasParam
    from sppas.src.annotations import sppasAnnotationsManager
    from sppas.src.ui.term.textprogress import ProcessProgressTerminal

    # Get directories
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    tmp_dir = os.path.join(input_dir, 'tmp')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)

    split_speakers_sppas(input_dir, tmp_dir)

    # Create objects
    parameters = sppasParam()
    manager = sppasAnnotationsManager()
   
    lgs = sppasLogSetup(50)
    lgs.null_handler()

    # Set parameters
    for i in [4, 5, 6]: # 4 textnorm, phonetize, alignment
        parameters.activate_step(i)
    parameters.add_to_workspace(tmp_dir)
    parameters.set_lang('fra')
    parameters.set_output_extension('.TextGrid', 'ANNOT')
    parameters.set_report_filename('sppas.log')

    # Align
    p = ProcessProgressTerminal()
    # Set beam sizes, etc. in sppas/src/annotations/Align/aligners/juliusalign.py. For now, beam size = 1000
    manager.annotate(parameters, p)

    # Move to desired output directory
    merge_speakers_sppas(tmp_dir, output_dir)
    p.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)