from pathlib import Path
from pympi.Praat import TextGrid
from model import EnsembleModel, create_dataloader
from pydub import AudioSegment

def get_textgrids(input_dir: Path, align_dir: Path):
    textgrids = {}

    for input in input_dir.glob('*.TextGrid'):
        textgrid = TextGrid(input.absolute())
        textgrids[input.stem] = [textgrid]
        sentence = textgrid.get_tier('spkr_1_1-trans').get_all_intervals()[1][2]
        textgrids[input.stem] = [textgrid, sentence.lower()]

    for align in align_dir.glob('*.TextGrid'):
        textgrid = TextGrid(align.absolute())
        align_token = [t for t in textgrid.get_tier('spkr_1_1-trans - words').get_all_intervals() if t[2] != '']
        textgrids[align.stem].extend([textgrid, align_token])

    return [(n, i, s, a, t) for n, (i, s, a, t) in textgrids.items()]

def replace_unk(sentence: str, align_token: list):
    s = sentence.replace("' ", "'").replace('-', ' ')
    unks = []
    
    for i, t in enumerate(align_token):
        word = t[2].replace('-', ' ')
        if '<unk>' in word:
            unks.append(i)
            continue
        if i > 0:
            word = ' ' + word
        if i < len(align_token) - 1:
            word += ' '
        if word in s:
            s = s.replace(word, ' ', 1)
    
    while '  ' in s:
        s = s.replace('  ', ' ')
    
    s = s.strip()
    
    unk_words = s.split(' ')
    
    if not (s == '' and len(unks) == 0 or len(unk_words) == len(unks)):
        print('Non-matching unks and s')
    
    for i, word in zip(unks, unk_words):
        start, end, _ = align_token[i]
        align_token[i] = (start, end, word)

def redact(segments_to_remove: list, input_dir: Path, output_dir: Path, name: str):
    wav_path = input_dir.joinpath(name + '.wav').resolve()
    redact_path = output_dir.joinpath(name + '.wav').resolve()

    audio = AudioSegment.from_wav(wav_path)

    redacted_audio = AudioSegment.empty()

    last_end = 0.0
    for start, end in segments_to_remove:
        redacted_audio += audio[last_end:start]
        redacted_audio += AudioSegment.silent(duration = end - start)
        last_end = end
    
    if len(audio) > last_end:
        redacted_audio += audio[last_end:]

    redacted_audio.export(redact_path, format='wav')

def redact_all(input_dir: str, align_dir: str, models_dir: str, output_dir: str, confidence=None):
    textgrids = get_textgrids(Path(input_dir), Path(align_dir))
    model = EnsembleModel(Path(models_dir))

    sentences = []
    align_tokens = []

    for name, input, sentence, align, align_token in textgrids:
        sentences.append(sentence)
        replace_unk(sentence, align_token)
        align_tokens.append([t[2] for t in align_token])
    
    loader, tokens = create_dataloader(align_tokens, tokenize=False)

    preds = model.predict(loader, confidence=confidence)

    redact_segments = {}

    for (name, input, sentence, align, align_token), pred, token in zip(textgrids, preds, tokens):
        segments_to_remove = []

        last_end = 0.0
        for (start, end, word), entity in zip(align_token, pred):
            if entity != 'O':
                data = (start * 1000, end * 1000)
                if data[0] == last_end: # Merge segment with the previous if last.end == this.start
                    last = segments_to_remove.pop()
                    data = (last[0], data[1])
                segments_to_remove.append(data)
                last_end = data[1]
        
        redact(segments_to_remove, Path(input_dir), Path(output_dir), name)
        
        redact_segments[name] = segments_to_remove

    return redact_segments