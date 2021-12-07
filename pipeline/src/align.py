import multiprocessing as mp
import os
import shutil

def align_mfa(input_dir: str, output_dir: str, dict_dir: str, acoustic_dir: str):
    from montreal_forced_aligner.aligner import PretrainedAligner
    from montreal_forced_aligner.corpus import AlignableCorpus
    from montreal_forced_aligner.dictionary import Dictionary
    from montreal_forced_aligner.helper import setup_logger
    from montreal_forced_aligner.models import AcousticModel
    from montreal_forced_aligner.command_line.mfa import fix_path, unfix_path
    from montreal_forced_aligner.config import TEMP_DIR, load_basic_align, AlignConfig

    mp.freeze_support()
    fix_path()
    
    # Init temporary directory
    data_dir = os.path.join(TEMP_DIR, 'textgrid')
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir)

    # Create objects
    logger = setup_logger('align', data_dir)
    config: AlignConfig = load_basic_align()
    config.beam = 100
    config.retry_beam = 400
    corpus = AlignableCorpus(input_dir, data_dir, num_jobs=4, logger=logger, use_mp=config.use_mp)
    dictionary = Dictionary(dict_dir, data_dir, word_set=corpus.word_set, logger=logger)
    
    acoustic = AcousticModel(acoustic_dir)
    #acoustic.feature_config.generate_features.__func__.__defaults__ = (None, False, None, False) # Remove CMVN calculation
    #acoustic.validate(dictionary)

    aligner = PretrainedAligner(corpus, dictionary, acoustic, config, temp_directory=data_dir, logger=logger, verbose=False)

    # Align
    aligner.align()

    # Move to desired output directory
    aligner.export_textgrids(output_dir)

    unfix_path()