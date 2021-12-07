from pathlib import Path
from pympi.Praat import TextGrid

TOLERANCE = 0.25 * 1000
METHOD = 'outer'

def get_textgrids(gold_path: Path):
    return [(g.stem, TextGrid(g.absolute())) for g in gold_path.glob('*.TextGrid')]

def get_gold_entities(textgrid: TextGrid):
    return [(t[0] * 1000, t[1] * 1000) for t in textgrid.get_tier('spkr_1_1-entities').get_all_intervals() if t[2] not in ['']]
    
def get_gold_words(textgrid: TextGrid):
    a = get_gold_entities(textgrid)
    ans = []
    for t in textgrid.get_tier('spkr_1_1-words').get_all_intervals():
        for b in a:
            if abs(t[0] * 1000 - b[0]) <= 10 and t[1] and 1000 <= b[1] or t[0] * 1000 >= b[0] and abs(t[1] * 1000 - b[1]) <= 10:
                ans.append(t[2])
    return ans

def is_correct(gold, pred, tolerance, method):
    if method == 'std': # is start and end between the boundary of time +- tolerance
        return abs(pred[0] - gold[0]) <= tolerance and abs(pred[1] - gold[1]) <= tolerance
    elif method == 'outer':
        if pred[0] <= gold[0] + tolerance and pred[1] >= gold[1] - tolerance == False:
            print("HEY LA LA")
        return pred[0] <= gold[0] + tolerance and pred[1] >= gold[1] - tolerance
    return False

def calculate_metrics(tp, fp, fn): # https://towardsdatascience.com/entity-level-evaluation-for-ner-task-c21fb3a8edf
    precision = tp / (tp + fp + 1e-100)
    recall = tp / (tp + fn + 1e-100)
    fscore = 2 * (precision * recall) / (precision + recall + 1e-100)
    return fscore, precision, recall

def printt(gold, pred):
    for a in [gold, pred]:
        print(','.join([str(b[0]) + '-' + str(b[1]) for b in a]))

def evaluate(redact_segments: list, gold_dir: str):
    gold_path = Path(gold_dir)

    if not gold_path.is_dir():
        print('Skipping evaluation, no gold annotations directory given')
        return
    
    tp = 0
    fp = 0
    fn = 0
    a = 0
    b = 0

    for name, textgrid in get_textgrids(gold_path):
        gold = get_gold_entities(textgrid)
        pred = redact_segments[name]

        printt(gold, pred)
        #print(','.join(get_gold_words(textgrid)))

        for g, p in zip(gold, pred):
            if is_correct(g, p, TOLERANCE, METHOD):
                tp += 1
            else:
                fn += 1
        fn += max(0, len(gold) - len(pred))
        fp += max(0, len(pred) - len(gold))
        a += len(gold)
        b += len(pred)

    print('tp:', tp, 'fp:', fp,'fn:', fn, 'n_gold:', a, 'n_pred:', b)

    print('f1 score: {}, precision: {}, recall: {}'.format(*calculate_metrics(tp, fp, fn)))
