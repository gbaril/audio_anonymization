from pathlib import Path
from pympi.Praat import TextGrid
from unidecode import unidecode
from fuzzywuzzy import fuzz
import string
from random import sample
import numpy as np

THRESHOLDS = [0, 100, 100, 100, 75, 85]
TABLE = str.maketrans(dict.fromkeys(string.punctuation))

def is_match(out, gol, partial=False, strip=False, threshold=None):
    if out == '<unk>':
        return True

    if not threshold:
        threshold = THRESHOLDS[min(max(len(out), len(gol)), len(THRESHOLDS) - 1)]

    if strip:
        gol = gol.translate(TABLE)
        out = out.translate(TABLE)
    
    if partial:
        return fuzz.partial_ratio(out, gol) >= threshold

    return fuzz.ratio(out, gol) >= threshold

def sanitize_interval(interval: str):
    return unidecode(interval.lower().strip())

def is_correct(out_start, out_end, gol_start, gol_end, tolerance, method):
    if method == 'std': # is start and end between the boundary of time +- tolerance
        return abs(out_start - gol_start) <= tolerance and abs(out_end - gol_end) <= tolerance
    elif method == 'outer':
        return out_start <= gol_start + tolerance and out_end >= gol_end - tolerance
    return False

def evaluate(textgrids: list, tolerance: float, method: str):
    correct_count = 0
    total_count = 0

    for num, (name, output_txt, gold_txt) in list(enumerate(textgrids)):
        # Get intervals
        output = output_txt.get_tier('spkr_1_1-trans - words').get_all_intervals()
        gold = gold_txt.get_tier('spkr_1_1-words').get_all_intervals()
        output = [t for t in output if len(t[2]) > 0]
        gold = [t for t in gold if len(t[2]) > 0]

        # print("--"*20)
        # print(num, name)
        # for a in output:
        #     print(a)
        # print("*" * 10)
        # for a in gold:
        #     print(a)
        
        i = 0
        j = 0
        skipped = []
        while i < len(output):
            start, end, out = output[i]
            # Sanitize words (accents, whitespace, lowercase)
            out = sanitize_interval(out)
            gol = sanitize_interval(gold[j][2])
            while not gol:
                j += 1
                gol = sanitize_interval(gold[j][2])

            # If we have an exact match
            if is_match(out, gol):
                # print('match', out, gol)
                if is_correct(start, end, gold[j][0], gold[j][1], tolerance, method):
                    correct_count += 1
                j += 1

            elif any(c in out and all(len(a) > 0 for a in out.split(c)) for c in string.punctuation):
                gold_start = gold[j][0]
                gold_end = gold[j][1]
                while j < len(gold) - 1 and is_match(out, sanitize_interval(gold[j+1][2]), True, True) and not is_match(out, gol):
                    j += 1
                    gold_end = gold[j][1]
                    gol += sanitize_interval(gold[j][2])
                # print('partial on gold', start, end, out, gol)
                if is_correct(start, end, gold_start, gold_end, tolerance, method):
                    correct_count += 1
                j += 1

            # If gold annotation is made of multiple words
            elif any(c in gol for c in string.punctuation) and is_match(out, gol, True):
                # While the next word get partial match or the whole new word is exactly the gold annotation
                while i < len(output) - 1 and is_match(sanitize_interval(output[i+1][2]), gol, True) and not is_match(out, gol):
                    i += 1
                    end = output[i][1]
                    out += sanitize_interval(output[i][2])
                # print('partial on out', start, end, out)
                if is_correct(start, end, gold[j][0], gold[j][1], tolerance, method):
                    correct_count += 1
                j += 1

            # If we find previous words not associated yet
            else:
                # Remove older annotations without an association
                tmp = []
                for previous in skipped:
                    if start - previous[1] <= 2:
                        tmp.append(previous)
                skipped = tmp

                # Check if we have a match
                found = None
                for num, previous in enumerate(skipped):
                    if is_match(previous[2], gol):
                        found = num
                        break
                if found is not None:
                    match = skipped.pop(num)
                    # print('previous', match)
                    if is_correct(match[0], match[1], gold[j][0], gold[j][1], tolerance, method):
                        correct_count += 1
                    j += 1

                # Add current word
                skipped.append((start, end, out))
            
            i += 1

        # print(skipped)
        total_count += len(gold)

    return correct_count, total_count

def get_textgrids(gold_standard_dir: Path, output_dir: Path):
    textgrids = {}

    for output in output_dir.glob('*.TextGrid'):
        output_textgrid = TextGrid(output.absolute())
        textgrids[output.stem] = [output_textgrid]

    for gold in gold_standard_dir.glob('*.TextGrid'):
        gold_textgrid = TextGrid(gold.absolute())
        textgrids[gold.stem].append(gold_textgrid)

    return [(n, o, g) for n, (o, g) in textgrids.items()]

def single_eval(gold_standard_dir: Path, output_dir: Path, tolerance: float, method: str):
    return evaluate(get_textgrids(gold_standard_dir, output_dir), tolerance, method)

def multiple_pop_eval(gold_standard_dir: Path, output_dir: Path, tolerance: float, method: str):
    textgrids = get_textgrids(gold_standard_dir, output_dir)
    subset_len = len(textgrids) // 5
    pop = []
    total_acc = []
    mean = []
    std = []

    for population_size in range(4, len(textgrids), 5):
        population = textgrids[:population_size]
        accuracies = []

        for i in range(10):
            subset = sample(textgrids, subset_len)
            corr, tot = evaluate(subset, tolerance, method)
            accuracies.append(corr / tot)

        pop.append(population_size + 1)
        correct_count, total_count = evaluate(population, tolerance, method)
        total_acc.append(correct_count / total_count)
        mean.append(np.mean(accuracies))
        std.append(np.std(accuracies))

    return pop, total_acc, mean, std