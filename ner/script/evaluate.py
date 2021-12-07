#TP If prediction is good
#FP If there is a prediction, but no gold
#FN If there is a gold, but no prediction

# Dans notre cas, recall plus important que precision

def _default_label_transform(label):
    return label

def info_label_transform(label):
    return 'O' if label == 'O' else 'INFO'

def evaluate(preds, golds, labels, label_transform=_default_label_transform):
    preds = _get_clean_biluo([p for p in preds])
    golds = _get_clean_biluo([g for g in golds])

    counts = {'total': {'tp': 0, 'fp': 0, 'fn': 0, 'sum': 0}}
    for label in labels:
        counts[label_transform(label)] = {'tp': 0, 'fp': 0, 'fn': 0, 'sum': 0}

    for gold, pred in zip(golds, preds):
        if gold == 'O' and pred == 'O':
            continue

        pred = label_transform(pred)
        gold = label_transform(gold)

        cats = ['tp']
        label = pred
        
        # https://github.com/explosion/spaCy/blob/master/spacy/scorer.py
        if gold == 'O' and pred != 'O':
            cats = ['fp']
        elif gold != 'O' and pred == 'O':
            cats = ['fn']
        elif pred != gold:
            cats = ['fp', 'fn'] # https://towardsdatascience.com/entity-level-evaluation-for-ner-task-c21fb3a8edf

        if 'fn' in cats:
            label = gold

        for cat in cats:
            counts[label][cat] += 1
            counts['total'][cat] += 1
        counts[label]['sum'] += 1
        counts['total']['sum'] += 1

    metrics = {}

    for key, values in counts.items():
        fscore, precision, recall = _calculate_metrics(values['tp'], values['fp'], values['fn'])
        metrics[key] = {'fscore': fscore, 'precision': precision, 'recall': recall, 'count': values['sum']}

    return metrics

def _get_clean_biluo(tags):
    return [tag.split('-')[-1] for tag in tags]

def _calculate_metrics(tp, fp, fn): # https://towardsdatascience.com/entity-level-evaluation-for-ner-task-c21fb3a8edf
    precision = tp / (tp + fp + 1e-100)
    recall = tp / (tp + fn + 1e-100)
    fscore = 2 * (precision * recall) / (precision + recall + 1e-100)
    return fscore, precision, recall

    