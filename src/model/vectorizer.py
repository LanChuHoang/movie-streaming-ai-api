import re
import string

import contractions
import nltk
from joblib import load
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
stemmer = PorterStemmer()
stopwords_english = stopwords.words("english")
stopwords_english.remove("not")
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

vect_file_path = "./saved_models/tfidf_vectorizer.joblib"
print(f"Loading vectorizer from path {vect_file_path}...")
vectorizer = load(vect_file_path)
print(f"Done loading vectorizer")


def Negation(sentence):
    """
    Input: Tokenized sentence (List of words)
    Output: Tokenized sentence with negation handled (List of words)
    """
    temp = int(0)
    for i in range(len(sentence)):
        if sentence[i - 1] in ["not", "n't"]:
            antonyms = []
            for syn in wordnet.synsets(sentence[i]):
                syns = wordnet.synsets(sentence[i])
                w1 = syns[0].name()
                temp = 0
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
                max_dissimilarity = 0
                for ant in antonyms:
                    syns = wordnet.synsets(ant)
                    w2 = syns[0].name()
                    syns = wordnet.synsets(sentence[i])
                    w1 = syns[0].name()
                    word1 = wordnet.synset(w1)
                    word2 = wordnet.synset(w2)
                    if isinstance(word1.wup_similarity(word2), float) or isinstance(
                        word1.wup_similarity(word2), int
                    ):
                        temp = 1 - word1.wup_similarity(word2)
                    if temp > max_dissimilarity:
                        max_dissimilarity = temp
                        antonym_max = ant
                        sentence[i] = antonym_max
                        sentence[i - 1] = ""
    while "" in sentence:
        sentence.remove("")
    return sentence


def process_review(text):
    text = text.lower()
    text = re.sub("<.*?>+", " ", text)
    text = re.sub(r"https?:\/\/.*[\r\n]*", " ", text)
    text = re.sub(r"[0-9]", " ", text)  # removing number
    text = re.sub("\n", " ", text)
    text = contractions.fix(text)
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)

    tokens = tokenizer.tokenize(text)
    tokens = Negation(tokens)

    clean_text = []
    for token in tokens:
        if token not in stopwords_english:
            stem_word = stemmer.stem(token)
            clean_text.append(stem_word)
    return " ".join(clean_text)


def transform(reviews):
    return vectorizer.transform(reviews)
