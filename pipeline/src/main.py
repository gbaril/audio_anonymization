from align import align_mfa
from ner import redact_all
from evaluation import evaluate
import os
import glob
from config import INPUT, ALIGN, NER_MODELS, REDACT, GOLD, DICT_DIR, ACOUSTIC_DIR

def delete_folder_content(folder: str):
    files = glob.glob(folder + '/*')
    for f in files:
        os.remove(f)

if __name__ == '__main__':
    delete_folder_content(ALIGN)
    align_mfa(INPUT, ALIGN, DICT_DIR, ACOUSTIC_DIR)
    delete_folder_content(REDACT)
    redact_segments = redact_all(INPUT, ALIGN, NER_MODELS, REDACT)
    #evaluate(redact_segments, GOLD)