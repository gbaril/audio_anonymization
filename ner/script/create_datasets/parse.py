import spacy
from spacy.training.iob_utils import doc_to_biluo_tags, biluo_to_iob
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split

RANDOM = 4

def docbinToList(docbin):
    nlp = spacy.load("fr_dep_news_trf")
    dataset = {'tokens': [], 'labels': []}
    for doc in docbin.get_docs(nlp.vocab):
        dataset['tokens'].append([str(s) for s in doc])
        dataset['labels'].append(biluo_to_iob(doc_to_biluo_tags(doc)))
    return dataset

def docbinToConll(docbin):
    nlp = spacy.load("fr_dep_news_trf")
    string = ''
    for doc in docbin.get_docs(nlp.vocab):
        tokens = [str(s) for s in doc]
        labels = biluo_to_iob(doc_to_biluo_tags(doc))
        for token, label in zip(tokens, labels):
            string += '{} -X- _ {}\n'.format(token, label)
        string += '\n'
    return string

def convertToDocbins(examples: list, verbose: int, split: bool):
    labels_num = {}
    labels = {}
    counts = {'total': 0}
    nlp = spacy.load("fr_core_news_lg")
    nlp.add_pipe('sentencizer')

    if split:
        docbins = [DocBin(), DocBin(), DocBin()] # 0 : train, 1: dev, 2: test
        train, test = train_test_split(examples, test_size=0.1, random_state=RANDOM)
        train, dev = train_test_split(train, test_size=0.1, random_state=RANDOM)
        loop = [train, dev, test]
    else:
        docbins = [DocBin()]
        loop = [examples]

    for dataset_id, dataset in enumerate(loop):
        counts[dataset_id] = 0
        for text, ents, ann in dataset:
            i = 0
            lastEnd = 0
            sentences = []
            # Split long example into smaller sentences and find its entities
            for span in nlp(text).sents:
                sentence = str(span)
                length = len(sentence)
                start = text.index(sentence, lastEnd)
                end = start + length

                phrase_ents = []
                done = False
                while not done and i < len(ents):
                    estart, eend, elabel, eline = ents[i]
                    # If end of entity in this sentence, adds it
                    if eend <= end:
                        # If negative, then we concatenate previous sentence with this one
                        while estart - start < 0:
                            lstart, _, lphrase_ents = sentences.pop()
                            start = lstart
                            sentence = text[start:end]
                            phrase_ents.extend(lphrase_ents)

                        phrase_ents.append((estart - start, eend - start, elabel, eline))
                        i += 1
                    else:
                        done = True

                # print(lastEnd, start, end, sentence, phrase_ents)
                sentences.append((start, end, phrase_ents))
                lastEnd = end

            # Create doc with ents from each sentence
            for sstart, send, phrase_ents in sentences:
                sentence = text[sstart:send]
                # Skip short sentences
                if len(sentence) <= 5:
                    if verbose >= 1:
                        print('====[SKIPPING SHORT SENTENCE]====', sentence, phrase_ents)
                    continue

                doc = nlp.make_doc(sentence)
                spans = []
                for start, end, label, line in phrase_ents:
                    if label not in labels_num:
                        labels_num[label] = 0
                        labels[label] = []
                    labels_num[label] += 1
                    # labels[label].append(sentence[start:end] + " --- " + ann + " " + line)
                    labels[label].append(sentence[start:end])
                    
                    span = doc.char_span(start, end, label=label, alignment_mode="contract")
                    if span is None:
                        if verbose >= 1:
                            if verbose == 1:
                                print(ann)
                            print('====[SPAN CREATION FAILED]', start, end, sentence[start:end], label, '|', line, '====')
                    else:
                        spans.append(span)
                
                filtered = spacy.util.filter_spans(spans) # Filter overlapping spans

                if verbose >= 2 and len(filtered) != len(spans):
                    print("-"* 50)
                    print('[Overlapping filters]', ann.split("/")[-1], len(filtered), len(spans))
            
                doc.set_ents(filtered)
                # print("-------------------")
                # print(doc.text)
                # print('---')
                # for a in doc.ents:
                #     print(a)

                counts[dataset_id] += len(filtered)
                counts['total'] += len(filtered)
                docbins[dataset_id].add(doc)

    if verbose >= 1:
        print(labels_num)
        # print(set(labels['Activity']))

    print('Number of entities', counts)

    return docbins