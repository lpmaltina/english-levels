import re
import numpy as np
import pandas as pd
import readability
from wordfreq import zipf_frequency
import spacy
from flask import Flask, request, render_template
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

app = Flask(__name__)


class PassthroughTransformer(BaseEstimator, TransformerMixin):

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            self.X = X
            return X

        def get_feature_names(self):
            return self.X.columns.tolist()

clfs = []
for clf_name in {'lgbm', 'lr1', 'lr2', 'rand_for', 'extra_trees'}:
    clfs.append(joblib.load('Dataset with 3 levels-all features-{}.tmp'.format(clf_name)))

nlp = spacy.load("en_core_web_sm")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    input_text = [str(x) for x in request.form.values()][0]
    tokens = nlp(input_text)
    preprocessed_text = '\n'.join([' '.join([token.text for token in sent]) for sent in tokens.sents])

    def lemmas(x):
        lemmas = []
        for token in x:
            if token.is_alpha:
                if token.lemma_ != '-PRON-':
                    lemmas.append(token.lemma_.lower())
                else:
                    lemmas.append(token.text.lower())
        return lemmas

    def preprocess_word_list(file_name):
        with open(file_name, encoding='utf-8') as f:
            words_from_list = f.read().split('\n')

        words_from_list = ' '.join([w for w in words_from_list if ' ' not in w])
        words_from_list = nlp(words_from_list)
        words_from_list = set(lemmas(words_from_list))
        return words_from_list

    readability_indices = ['Kincaid', 'ARI', 'Coleman-Liau',
                          'FleschReadingEase', 'GunningFogIndex', 'LIX',
                          'SMOGIndex', 'RIX', 'DaleChallIndex']
    indices_values = readability.getmeasures(preprocessed_text, lang='en')['readability grades']
    features = {ind: indices_values[ind] for ind in readability_indices}

    features['lemmas'] = lemmas(tokens)
    features['unique_lemmas'] = len(set(features['lemmas']))/len(features['lemmas'])
    features['named_entities'] = len(tokens.ents)/len([token for token in tokens if not token.is_space])
    features['unique_named_entities'] = len(set([entity.text for entity in tokens.ents])
                                            ) / len([token for token in tokens if not token.is_space])

    zipf_freqs = []
    for token in features['lemmas']:
        zipf = zipf_frequency(token, 'en')
        if zipf > 0:
            zipf_freqs.append(round(zipf))

    features['zipf_freqs<=4'] = len([freq for freq in zipf_freqs if freq <= 4]) / len(zipf_freqs)

    for i in range(5, 8):
        features['zipf_freqs={}'.format(str(i))] = zipf_freqs.count(i) / len(zipf_freqs)

    a1 = preprocess_word_list('A1' + '.txt')
    a2 = preprocess_word_list('A2' + '.txt') - a1
    b1 = preprocess_word_list('B1' + '.txt') - a2 - a1
    b2 = preprocess_word_list('B2' + '.txt') - b1 - a2 - a1
    concrete = preprocess_word_list('concrete nouns' + '.txt')
    abstract = preprocess_word_list('abstract nouns' + '.txt')

    word_lists = {'A1': a1, 'A2': a2, 'B1': b1, 'B2': b2,
                  'concrete_nouns': concrete, 'abstract_nouns': abstract}

    for list_name, word_list in word_lists.items():
        features[list_name] = len([w for w in features['lemmas'] if w in word_list]) / len(features['lemmas'])

    features['pos'] = ' '.join([token.pos_ for token in tokens if not token.is_space])
    features['tag'] = ' '.join([token.tag_ for token in tokens if not token.is_space])
    features['dep'] = ' '.join([token.dep_ for token in tokens if not token.is_space])
    features['pos_pos'] = ' '.join(['_'.join([token.head.pos_, token.pos_]) for token in tokens if not token.is_space])
    features['pos_dep_pos'] = ' '.join(['_'.join([token.head.pos_, token.dep_, token.pos_])
                                        for token in tokens if not token.is_space])

    num_dependencies = [len(list(token.children)) for token in tokens if not token.is_space]
    features['mean_num_dependencies'] = sum(num_dependencies) / len([token for token in tokens if not token.is_space])

    features['num_noun_chunks'] = len(list(tokens.noun_chunks))/len([token for token in tokens if not token.is_space])

    noun_chunks = list(tokens.noun_chunks)
    len_noun_chunks = [len(noun_chunk) for noun_chunk in noun_chunks]
    features['length_noun_chunks'] = sum(len_noun_chunks) / len(noun_chunks)

    def activeness(x):
        x = x.split()
        active = len([dep for dep in x if dep in ('aux', 'csubj', 'nsubj')])
        passive = len([dep for dep in x if dep in ('aux_pass', 'csubjpass', 'nsubjpass')])

        if passive + active > 0:
            result = active / (active + passive)
        else:
            result = 0
        return result

    features['activeness'] = activeness(features['dep'])

    reg_exp = re.compile('(?:.+)(?:ion|ment|ism|ity|ship|ness|nce|th|hood|cy|ry)')

    nouns = [token.lemma_.lower() for token in tokens if token.pos_ == 'NOUN']
    abstract_nouns = [noun for noun in nouns if re.fullmatch(reg_exp, noun)]
    features['noun_abstract_suffixes'] = len(abstract_nouns) / len(nouns)
    features['lemmas'] = ' '.join(features['lemmas'])

    weights = None
    example = pd.DataFrame(features, index=[0])
    pred = np.asarray([clf.predict_proba(example) for clf in clfs])
    pred = np.average(pred, axis=0, weights=weights)
    pred = np.argmax(pred, axis=1)

    return render_template('index.html', prediction_text='Level: {}'.format(pred[0]))


if __name__ == "__main__":
    app.run(debug=True)