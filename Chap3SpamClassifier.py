# %%
"""In this code we wanna build a spam classifier."""
# %% Setup

# Importing modules
from sklearn.metrics import precision_score, recall_score
# from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
import re
from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import email.policy
import email
import os
import tarfile
import urllib.request
import matplotlib as mpl
# import matplotlib.pyplot as plt
import sys
import sklearn

# Python ≥3.5 is required
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
assert sklearn.__version__ >= "0.20"

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


MODEL_PATH = "cache\\SpamClassifier"


# %% Fetching and loading the dataset
DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")


def fetch_spam_data(ham_url=HAM_URL, spam_url=SPAM_URL, spam_path=SPAM_PATH):
    """Fetch the spam dataset."""
    os.makedirs(spam_path, exist_ok=True)
    for filename, url in \
            (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):

        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()


# fetch_spam_data()


# loading the data
HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")

ham_filenames = [name for name in
                 sorted(os.listdir(HAM_DIR)) if len(name) > 20]

spam_filenames = [name for name in
                  sorted(os.listdir(SPAM_DIR)) if len(name) > 20]


def load_email(is_spam, filename, spam_path=SPAM_PATH):
    """Load emails."""
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


ham_emails = [load_email(is_spam=False, filename=name)
              for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name)
               for name in spam_filenames]


def get_email_structure(email):
    """Get emails structure."""
    if isinstance(email, str):
        return email

    payload = email.get_payload()

    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))

    else:
        return email.get_content_type()


def structures_counter(emails):
    """Count the email structures."""
    structures = Counter()
    for emailIns in emails:
        structure = get_email_structure(emailIns)
        structures[structure] += 1
    return structures


# print("ham_emails:", structures_counter(ham_emails).most_common())
# print("spam_emails", structures_counter(spam_emails).most_common())

# %% Preproccessing the dataset

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

html_spam_emails = [email for email in X_train[y_train == 1]
                    if get_email_structure(email) == "text/html"]

sample_html_spam = html_spam_emails[7]


def email_to_text(email):
    """Convert the email content to text."""
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue

        try:
            content = part.get_content()

        except:  # in case of encoding issues
            content = str(part.get_payload())

        if ctype == "text/plain":
            return content
        else:
            html = content

    if html:
        return BeautifulSoup(html, features="html.parser").get_text().strip()


# print(email_to_text(sample_html_spam)[:100], "...")

try:
    import nltk

    stemmer = nltk.PorterStemmer()
    # for word in ("Computations", "Computation", "Computing", "Computed",
    #              "Compute", "Compulsive"):
    #     print(word, "=>", stemmer.stem(word))

except ImportError:
    print("Error: stemming requires the NLTK module.")
    stemmer = None

try:
    import urlextract
    # may require an Internet connection to download root domain names

    url_extractor = urlextract.URLExtract()
    # print(url_extractor.find_urls(
    #     "Will it detect github.com and \
    #         https://youtu.be/7Pq-S557XQU?t=3m32s"))

except ImportError:
    print("Error: replacing URLs requires the urlextract module.")
    url_extractor = None


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    """This class counts the number of stems of the words in the email text and \
        categorize them."""

    def __init__(self,
                 strip_headers=True,
                 lower_case=True,
                 remove_punctuation=True,
                 replace_urls=True,
                 replace_numbers=True,
                 stemming=True):
        """Initialize the class with activating the features."""
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        """ّFit the model to the dataset."""
        return self

    def transform(self, X, y=None):
        """Transform the dataset."""
        X_transformed = list()

        for email_ins in X:
            """This for is for checking multi emails."""
            text = email_to_text(email_ins) or ""

            if self.lower_case:
                text = text.lower()

            if self.replace_urls and (url_extractor is not None):
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)

                for url in urls:
                    text = text.replace(url, " URL ")

            if self.replace_numbers:
                text = re.sub(
                    r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)

            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)

            word_counts = Counter(text.split())

            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()

                for word, count in word_counts.items():
                    """This for loop counts the number of stems."""

                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count

                word_counts = stemmed_word_counts

            X_transformed.append(word_counts)

        return np.array(X_transformed)


X_few = X_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    """This class counts the number of the words and determines \
         how much these words present in the email."""

    def __init__(self, vocabulary_size=1000):
        """
        Initialize the class with vocabulary_size.

        vocabulary_size: size of vocabulary
            (an ordered list of the most common words)
        """
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        """Fit the model to the dataset and set the vocabulary_ attribute."""
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():

                # The maximum number of words is limited to 10.
                total_count[word] += min(count, 10)

        # Seperating most common words
        most_common = total_count.most_common()[:self.vocabulary_size]

        self.vocabulary_ = {word: index + 1 for index,
                            (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):
        """Transform the dataset."""
        rows = list()
        cols = list()
        data = list()
        for row, word_count in enumerate(X):

            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)

        return csr_matrix((data, (rows, cols)),
                          shape=(len(X), self.vocabulary_size + 1))


vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)

"""
WordCounterToVectorTransformer doesn't have fit_transfrom method but
fit_transform method works because the class inherits TransformerMixin. \
"""
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)

# print(X_few_vectors.toarray())
# print(vocab_transformer.vocabulary_)

# %% Training a Classifier


preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)

# TEST
# log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
# score = cross_val_score(log_clf,
#                         X_train_transformed,
#                         y_train,
#                         cv=3,
#                         verbose=3)

# print(score.mean())


X_test_transformed = preprocess_pipeline.transform(X_test)

log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
log_clf.fit(X_train_transformed, y_train)

y_pred = log_clf.predict(X_test_transformed)

print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))
