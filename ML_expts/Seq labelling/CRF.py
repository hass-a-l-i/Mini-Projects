# going to label a sequence using conditional random fields (can also do same thing with RNN)
# dataset here is flight offers, with cols of destination, origin, seperator token, price, flag, irrelevant token
# want to create tagger that will use CRFs to label data in the sequence of flight offers above
# use CRF as it is conditional on assigning label to current sample by using neighbouring samples (neighbouring structure could be linear or grid in 2D etc...)
# CRF models decision boundary between different classes - is discriminative (generative model how data was generated then use to make new classifications)
# applies logistic regression classifier on sequential inputs
# to do this, apply the logistic regression classifier to a feature function which takes inputs such as set of vector inputs and position of data point we are predicting (is a step function)
# feature func of each data point dep on label of previous and current word to return binary output, each one has its own weight too
# works like gradient descent where parameters update iteratively until convergence  so we have learning rate alpha here too
# note less biased as assume dependency of labels instead of independent like generative

# going to define new section as below
def New_Section(x):
    print()
    print("--------------------------------------------------------------------------------")
    print("Section", x)
    print()


# first we add parts of speech as a feature (e.g. verbs and adverbs, or nouns and pronouns) as tags because the language is spanish => aids in translating
# to do this we use standford POS tagger
import json
import pandas as pd

# first load dataset
vuelos = pd.read_csv('vuelos.csv', index_col=0)
with pd.option_context('max_colwidth', 800):
    print(vuelos.loc[:40:5][['label']])

# now load pos tagger but also need java to access it
# we use ntlk module and the .jar file we have to do this
import os
java_path = "C:/Program Files (x86)/Common Files/Oracle/Java/javapath/java.exe"
os.environ['JAVAHOME'] = java_path
from nltk.tag.stanford import StanfordPOSTagger
spanish_postagger = StanfordPOSTagger('stanford_pos_tagger/models/spanish-ud.tagger',
                                      'stanford_pos_tagger/stanford-postagger.jar')
# test on spanish sentence to see what tags are returned
phrase = 'Amo el canto del cenzontle, pájaro de cuatrocientas voces.'
tags = spanish_postagger.tag(phrase.split())
print(tags)
New_Section(1)

# note tagger takes list of strings instead of full sentences so need to make list before passing it
# note the tag returned gives our word and its tag (wordy type e.g. noun etc...)

# as for each ML problem, approach varies
# here we can convert words into tokens that we pass to POS tagger => allows for string data to be converted into numerical form
from nltk.tokenize import TweetTokenizer
TWEET_TOKENIZER = TweetTokenizer()
def index_emoji_tokenize(string, return_flags=False):
    flag = ''
    ix = 0
    tokens, positions = [], []
    for t in TWEET_TOKENIZER.tokenize(string):
        ix = string.find(t, ix)
        if len(t) == 1 and ord(t) >= 127462:  # this is the code for 🇦
            if not return_flags: continue
            if flag:
                tokens.append(flag + t)
                positions.append(ix - 1)
                flag = ''
            else:
                flag = t
        else:
            tokens.append(t)
            positions.append(ix)
        ix = +1
    return tokens, positions

# let's test our tokenization function
label = vuelos.iloc[75]['label']  # 75th in label col
print(label)
print()
tokens, positions = index_emoji_tokenize(label, return_flags=True)
print(tokens)
print(positions)
New_Section(2)

# can now use a separate file with labels used to train our algo to make predictions
# labels are known for these - see to_label file

# now we need to add more features on top of POS tags
# features such as length of token (=word) , length of sentence, if number or uppercase etc...
# let's start with labelled data
labelled_data = pd.read_csv("to_label-done.csv")
print(labelled_data.head())
New_Section(3)

# now we create func to read all labels from our new data (flight offers), split into own lists
def read_whole_offers(dataset):
    current_offer = 0
    rows = []
    for _, row in dataset.iterrows():
        if row['offer_id'] != current_offer:
            yield rows
            current_offer = row['offer_id']
            rows = []
        rows.append(list(row.values))
    yield rows

offers = read_whole_offers(labelled_data)
offer_ids, tokens, positions, pos_tags, token_count, labels = zip(*next(offers))
print(offer_ids)
print(tokens)
print(positions)
print(pos_tags)
print(token_count)
print(labels)
New_Section(4)

# now we generate more features so that we can build training set
# features generated include length of token, length of offer, POS tag of token to left and right of current, if token uppercase or not
def generate_more_features(tokens, pos_tags):
    lengths = [len(l) for l in tokens]
    n_tokens = [len(tokens) for l in tokens]
    augmented = ['<p>'] + list(pos_tags) + ['</p>']  # markers before and after are just so we know this is our augmented dataset
    uppercase = [all([l.isupper() for l in token]) for token in tokens]
    return lengths, n_tokens, augmented[:len(tokens)], augmented[2:], uppercase

lengths, n_tokens, augmented1, augmented2, uppercase = generate_more_features(tokens, pos_tags)
print(lengths)
print(n_tokens)
print(augmented1)
print(augmented2)  # note we split augmented into two sets of 11 for training purposes, this list is to denote from which tags new features were generated
print(uppercase)
New_Section(5)

# now need to apply to all offers in dataset to generate all features for our main dataset from earlier and train
# now define helper funcs to help with punctuation and numeric token identification
import string
punctuation = set(string.punctuation)
def is_punctuation(token):
    return token in punctuation
def is_numeric(token):
    try:
        float(token.replace(",", ""))
        return True
    except:
        return False

# inputs to python crf must have token and features represented by pairs of values and ofc each token will have different features based on the factors above like position of token
# create func that takes dataset and returns features which we can feed to train the model
def featurise(sentence_frame, current_idx):
    current_token = sentence_frame.iloc[current_idx]
    token = current_token['token']
    position = current_token['position']
    token_count = current_token['token_count']
    pos = current_token['pos_tag']

    # Shared features across tokens, create dictionary for this
    features = {
            'bias': True,
            'word.lower': token.lower(),
            'word.istitle': token.istitle(),
            'word.isdigit': is_numeric(token),
            'word.ispunct': is_punctuation(token),
            'word.position': position,
            'word.token_count': token_count,
            'postag': pos,
    }

    if current_idx > 0:  # if word is not the first one, look at previous token and adjust features to reflect
        prev_token = sentence_frame.iloc[current_idx-1]['token']
        prev_pos = sentence_frame.iloc[current_idx-1]['pos_tag']
        features.update({
            '-1:word.lower': prev_token.lower(),
            '-1:word.istitle': prev_token.istitle(),
            '-1:word.isdigit': is_numeric(prev_token),
            '-1:word.ispunct': is_punctuation(prev_token),
            '-1:postag': prev_pos
        })
    else:
        features['BOS'] = True

    if current_idx < len(sentence_frame) - 1:  # if word is not the last one, look at next token instead
        next_token = sentence_frame.iloc[current_idx+1]['token']
        next_tag = sentence_frame.iloc[current_idx+1]['pos_tag']
        features.update({
            '+1:word.lower': next_token.lower(),
            '+1:word.istitle': next_token.istitle(),
            '+1:word.isdigit': is_numeric(next_token),
            '+1:word.ispunct': is_punctuation(next_token),
            '+1:postag': next_tag
        })
    else:
        features['EOS'] = True

    return features

# now lets use featurise on the first token (skipping first punctuation at start) can see output is basically our features that we extract using the keys we made in dictionary of features above
# here we make offer_0 = a sentence extracted from the data which is offer ID 0 only i.e. the first sentence only in out dataset
# sentence frame is our sentence as a dataframe (see offer_0)
features_labels = pd.read_csv("to_label-done.csv")
features_labels = features_labels[~features_labels['label'].isna()]
offer_0 = features_labels[features_labels['offer_id'] == 0]
print(offer_0)
total_features = featurise(offer_0, 1)
for key, value in total_features.items():
    print(key, value)
New_Section(6)

# need to now allow this to be iterated over entire sentence instead of one token only
def featurize_sentence(sentence_frame):
    labels = sentence_frame['label'].to_list()
    features = [featurise(sentence_frame, i) for i in range(len(sentence_frame))]
    return features, labels
# test it out on the 0 ID offers we made dataframe of tokens for, focusing on first token
features, labels = featurize_sentence(offer_0)
print(features[1])
print(labels[1])

# now since we work on sequences we want to provide CRF with sequences of data to work with, so it can train labelling off these sequences
# sequence will be the group of features for each token within a sentence = one sequence (list of features for each token within sentence)
# each token has label within each sequence
def rollup(dataset):
    sequences = []
    labels = []
    offers = dataset.groupby('offer_id')
    for name, group in offers:
        sqs, lbls = featurize_sentence(group)
        sequences.append(sqs)
        labels.append(lbls)
    return sequences, labels
all_sequences, all_labels = rollup(features_labels)
print(all_labels[0])
print(all_sequences[0])
New_Section(7)

# now data ready for training - split into training and testing
import sklearn
from sklearn.model_selection import train_test_split
train_docs, test_docs, train_labels, test_labels = train_test_split(all_sequences, all_labels)
print(len(train_docs), len(test_docs))  # sense check split
New_Section(8)

# now we can create CRF, x seq is features and y seq is labels
import pycrfsuite
trainer = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(train_docs, train_labels):
    trainer.append(xseq, yseq)
trainer.set_params({
    'c1': 1.0,  # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 200,
    'feature.possible_transitions': True})  # set parameters of CRF
# now we call method train that saves model to file that we can use to perform tests for new sentences
trainer.train('vuelax.crfsuite')

# now we have trained, we label "unseen" sequences to test our CRF trained tagger
crf_tagger = pycrfsuite.Tagger()
crf_tagger.open('vuelax.crfsuite')
# remember need sentences to be processed as we did before to be fed into model  as we did for training, we have preprocessed all data in advance above so just need to implement test docs now as it is ML friendly form
# lets predict and test tags for one of our test docs to see how accurate it is
predicted_tags = crf_tagger.tag(test_docs[2])
print("Predicted: ",predicted_tags)
print("Correct  : ",test_labels[2])
New_Section(9)

# now we evaluate the tagger with classification report
from sklearn.metrics import classification_report
all_true, all_pred = [], []
for i in range(len(test_docs)):
    all_true.extend(test_labels[i])  # our actual labels
    all_pred.extend(crf_tagger.tag(test_docs[i]))  # our predicted labels
print(classification_report(all_true, all_pred))
print()
print()
print()
# works very well!

New_Section(10)
print("TEST ON NEW TEXT")
# finally we can test on entirely new sentence
# to do this we need pipeline to go from new offer (i.e. new sentence) to fully labelled offer as we made above
# lets make a new offer - Without stops in the USA! 🇪🇬 any airport in México to Cairo, Egypt $13,677!
offer_text = "¡Sin pasar EE.UU! 🇪🇬¡Todo México a El Cairo, Egipto $13,677!"

# step 1 - tokenize using index emoji tokenize func
new_tokens, new_positions = index_emoji_tokenize(offer_text)
print(new_tokens)

# step 2 - POS tag the input sentence
_, pos_tags = zip(*spanish_postagger.tag(new_tokens))
print(pos_tags)
print(len(pos_tags), len(new_positions), len(new_tokens))

# step 3 - prepare data to be fed to CRF (featurize, but we change it to not return labels, and edit featurise func to only handle pos tag, position and token)
def featurise2(sentence_frame, current_idx):
    current_token = sentence_frame.iloc[current_idx]
    token = current_token['token']
    position = current_token['position']
    pos = current_token['pos_tag']

    # Shared features across tokens, create dictionary for this
    features = {
            'bias': True,
            'word.lower': token.lower(),
            'word.istitle': token.istitle(),
            'word.isdigit': is_numeric(token),
            'word.ispunct': is_punctuation(token),
            'word.position': position,
            'postag': pos,
    }

    if current_idx > 0:  # if word is not the first one, look at previous token and adjust features to reflect
        prev_token = sentence_frame.iloc[current_idx-1]['token']
        prev_pos = sentence_frame.iloc[current_idx-1]['pos_tag']
        features.update({
            '-1:word.lower': prev_token.lower(),
            '-1:word.istitle': prev_token.istitle(),
            '-1:word.isdigit': is_numeric(prev_token),
            '-1:word.ispunct': is_punctuation(prev_token),
            '-1:postag': prev_pos
        })
    else:
        features['BOS'] = True

    if current_idx < len(sentence_frame) - 1:  # if word is not the last one, look at next token instead
        next_token = sentence_frame.iloc[current_idx+1]['token']
        next_tag = sentence_frame.iloc[current_idx+1]['pos_tag']
        features.update({
            '+1:word.lower': next_token.lower(),
            '+1:word.istitle': next_token.istitle(),
            '+1:word.isdigit': is_numeric(next_token),
            '+1:word.ispunct': is_punctuation(next_token),
            '+1:postag': next_tag
        })
    else:
        features['EOS'] = True

    return features
def featurize_sentence2(sentence_frame):
    features = [featurise2(sentence_frame, i) for i in range(len(sentence_frame))]
    return features
titles = {'token': new_tokens, 'position': new_positions, 'pos_tag': pos_tags}
df = pd.DataFrame(titles)
features = featurize_sentence2(df)
print(features[0])

# now we load saved CRF model from earlier and apply it
crf_tagger2 = pycrfsuite.Tagger()
crf_tagger2.open('vuelax.crfsuite')
assigned_tags = crf_tagger2.tag(features)  # tag each token based on features
for assigned_tag, token in zip(assigned_tags, tokens):  #
    print(f"{assigned_tag} - {token}")
# tages are correct via inspection, o is origin, d is destination