
import json
import gzip
import math
import pickle
import os
import string
from collections import Counter

import spacy
import pyLDAvis as pyLDAvis
from lxml import etree
import numpy as np
from numpy import log, exp, argmax
from numpy.random import multinomial
# Construction 1
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
'''
TODO. Install the following packages in your virtualenv, if not installed yet
Install numpy
-------------
python -m pip install -U pip setuptools wheel
python -m pip install -U numpy
Install spacy
-------------
python -m pip install -U spacy
python -m spacy download en_core_web_sm
Install lxml
------------
python -m pip install lxml
Install scikit-learn
--------------------
python -m pip install -U scikit-learn
Install pyLDAvis
----------------
python -m pip install -U pyldavis
'''
'''
TODO. Following class is the Gibbs sampling algorithm for a Dirichlet Mixture
Model for the clustering short text documents.
You DON'T need to change any part of the following class; from 'BEGIN_GSDMM' to
'END_GSDMM'
Reference: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf
This implementation of GSDMM is from
  https://github.com/rwalk/gsdmm/tree/master/gsdmm
TODO. Download any Pubmed archive file from the repository
(https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/) and save it in the same
directory where this script file is.
'''


# 'BEGIN_GSDMM' (DO NOT CHAGE) ------------------------------------------------
class GSDMM:
    def __init__(self, K=8, alpha=0.1, beta=0.1, n_iters=30):
        '''
        :param K: int
            Upper bound on the number of possible clusters. Typically many
            fewer
        :param alpha: float between 0 and 1
            Alpha relates to the prior probability of a document choosing a
            cluster
        :param beta: float between 0 and 1
            Beta relates the homogeneity of the cluster
        :param n_iters:
        '''
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters

        # slots for computed variables
        self.number_docs = None
        self.vocab_size = None
        self.cluster_doc_count = [0 for _ in range(K)]
        self.cluster_word_count = [0 for _ in range(K)]
        self.cluster_word_distribution = [{} for i in range(K)]

    @staticmethod
    def from_data(K, alpha, beta, D, vocab_size, cluster_doc_count,
                  cluster_word_count, cluster_word_distribution):
        '''
        Reconstitute a GSDMM from previously fit data
        :param K:
        :param alpha:
        :param beta:
        :param D:
        :param vocab_size:
        :param cluster_doc_count:
        :param cluster_word_count:
        :param cluster_word_distribution:
        :return:
        '''
        mdl = GSDMM(K, alpha, beta, n_iters=30)
        mdl.number_docs = D
        mdl.vocab_size = vocab_size
        mdl.cluster_doc_count = cluster_doc_count
        mdl.cluster_word_count = cluster_word_count
        mdl.cluster_word_distribution = cluster_word_distribution
        return mdl

    @staticmethod
    def _sample(p):
        '''
        Sample with probability vector p from a multinomial distribution
        :param p: list
            List of probabilities representing probability vector for the
            multinomial distribution
        :return: int
            index of randomly selected output
        '''
        return [i for i, entry in enumerate(multinomial(1, p)) if entry != 0][0]

    def fit(self, docs, vocab_size):
        '''
        Cluster the input documents
        :param docs: list of list
            list of lists containing the unique token set of each document
        :param V: total vocabulary size for each document
        :return: list of length len(doc)
            cluster label for each document
        '''
        alpha, beta, K, n_iters, V = self.alpha, self.beta, self.K, self.n_iters, vocab_size

        D = len(docs)
        self.number_docs = D
        self.vocab_size = vocab_size

        # unpack to easy var names
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution
        cluster_count = K
        d_z = [None for i in range(len(docs))]

        # initialize the clusters
        for i, doc in enumerate(docs):

            # choose a random  initial cluster for the doc
            z = self._sample([1.0 / K for _ in range(K)])
            d_z[i] = z
            m_z[z] += 1
            n_z[z] += len(doc)

            for word in doc:
                if word not in n_z_w[z]:
                    n_z_w[z][word] = 0
                n_z_w[z][word] += 1

        for _iter in range(n_iters):
            total_transfers = 0

            for i, doc in enumerate(docs):

                # remove the doc from it's current cluster
                z_old = d_z[i]

                m_z[z_old] -= 1
                n_z[z_old] -= len(doc)

                for word in doc:
                    n_z_w[z_old][word] -= 1

                    # compact dictionary to save space
                    if n_z_w[z_old][word] == 0:
                        del n_z_w[z_old][word]

                # draw sample from distribution to find new cluster
                p = self.score(doc)
                z_new = self._sample(p)

                # transfer doc to the new cluster
                if z_new != z_old:
                    total_transfers += 1

                d_z[i] = z_new
                m_z[z_new] += 1
                n_z[z_new] += len(doc)

                for word in doc:
                    if word not in n_z_w[z_new]:
                        n_z_w[z_new][word] = 0
                    n_z_w[z_new][word] += 1

            cluster_count_new = sum([1 for v in m_z if v > 0])
            print("In stage %d: transferred %d clusters with %d clusters populated" % (
                _iter, total_transfers, cluster_count_new))
            if total_transfers == 0 and cluster_count_new == cluster_count and _iter > 25:
                print("Converged.  Breaking out.")
                break
            cluster_count = cluster_count_new
        self.cluster_word_distribution = n_z_w
        return d_z

    def score(self, doc):
        '''
        Score a document
        Implements formula (3) of Yin and Wang 2014.
        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf
        :param doc: list[str]: The doc token stream
        :return: list[float]: A length K probability vector where each
                              component represents the probability of the
                              document appearing in a particular cluster
        '''
        alpha, beta, K, V, D = self.alpha, self.beta, self.K, self.vocab_size, self.number_docs
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution

        p = [0 for _ in range(K)]

        #  We break the formula into the following pieces
        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        #  lN1 = log(m_z[z] + alpha)
        #  lN2 = log(D - 1 + K*alpha)
        #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))
        #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))

        lD1 = log(D - 1 + K * alpha)
        doc_size = len(doc)
        for label in range(K):
            lN1 = log(m_z[label] + alpha)
            lN2 = 0
            lD2 = 0
            for word in doc:
                lN2 += log(n_z_w[label].get(word, 0) + beta)
            for j in range(1, doc_size + 1):
                lD2 += log(n_z[label] + V * beta + j - 1)
            p[label] = exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        pnorm = sum(p)
        pnorm = pnorm if pnorm > 0 else 1
        return [pp / pnorm for pp in p]

    def choose_best_label(self, doc):
        '''
        Choose the highest probability label for the input document
        :param doc: list[str]: The doc token stream
        :return:
        '''
        p = self.score(doc)
        return argmax(p), max(p)


# 'END_GSDMM' ---------------------------------------------------------------

def prepare_vis_data(mdl, docs, vocab, K):
    def prepare_data():
        nonlocal mdl, docs, K
        doc_topic_dists = [mdl.score(doc) for doc in docs]
        doc_lengths = [len(doc) for doc in docs]
        doc_topic_dists2 = [[v if not math.isnan(v) else 1 / K for v in d]
                            for d in doc_topic_dists]
        doc_topic_dists2 = [d if sum(d) > 0 else [1 / K] * K for d in
                            doc_topic_dists]
        matrix = []
        for cluster in mdl.cluster_word_distribution:
            total = sum(cluster.values())
            row = [cluster.get(k, 0) / total for k in vocab]
            matrix.append(row)
        return matrix, doc_topic_dists2, doc_lengths

    out = pyLDAvis.prepare(*prepare_data(), vocab.keys(), vocab.values(),
                           sort_topics=False, mds='mmds')
    return out


# TODO. Use the downloaded filename for data_file below
# data_file = 'your_file_name.xml.gz'
mdl_file = 'trained-mdl.pkl'  # Trained topic model will be stored here
data_file = gzip.open('//Users/youssefkindo/Downloads/pubmed21n0008.xml.gz', 'rb')


def preprocess():
    '''
    Read document from an archived medline file, preprocess the documents, and
    build a vocabulary. NOTE, we use only titles (not the abstract body texts).
    We will run a GSDMM model for short texts.
    Preprocessing steps:
    1. Convert text to lowercase and build a vocabulary
    2. Tokenization
    3. Remove Stopwords and length-1 character words
    4. Remove unique tokens (words that occur only once in the corpus)
    5. Remove empty documents
    :return:
        - docs (list of list[str]) - list of documents where each document is a
          list of tokens
        - vocab (Counter) - vocabulary in the Counter structure
    '''
    docs = []  # List of documents
    raw_docs = []
    vocab = Counter()  # Counter keys are terms, values are term frequencies

    # TODO. Instantiate your tokenizer model
    # You can use spaCy tokenizer (https://spacy.io/api/tokenizer)
    # [YOUR CODE HERE]
    nlp = English()
    # Create a blank Tokenizer with just the English vocab
    tokenizer = Tokenizer(nlp.vocab)


    # TODO. Read documents. While reading, extract titles, apply
    # preprocessing steps, and update docs and vocab.
    # [YOUR CODE HERE]
    tree = etree.parse(data_file)
    root = tree.getroot()
    for node in root.iter():
        if node.tag == "ArticleTitle":
            text = node.text.lower()
            if text:
                raw_docs.append(text)
                text = text.translate(str.maketrans('', '', string.punctuation))
                filtered_sentence = remove_stopwords(text)
                tokens = tokenizer(filtered_sentence)
                docs.append([t.text for t in tokens])
                vocab.update(filtered_sentence.split(" "))


    # TODO. Check the stats before returning.
    # Your docs and vocab should look like below:
    #
    # len(docs) = 29860         : slightly less than the total # of docs
    # len(vocab) = 20356
    # vocab.most_common(10)
    # [('patients', 1559), ('study.', 1025), ('cancer', 959), ('cell', 929),
    #  ('treatment', 919), ('study', 855), ('clinical', 852), ...]

    print(len(docs))
    print(len(vocab))
    print(vocab.most_common(10))

    return docs, vocab, raw_docs


if __name__ == '__main__':
    # Check if there's a saved model
    data, mdl, docs, vocab,raw_docs = None, None, None, None,None
    if os.path.exists(mdl_file):
        print('Reading from the trained model file.')
        with open(mdl_file, 'rb') as f:
            data = pickle.load(f)
            mdl = data['mdl']
            docs = data['docs']
            vocab = data['vocab']
            raw_docs = data['raw_docs']

    else:
        # TODO. Define your preprocessing function above. Ideally, this will
        # return the preprocessed list of documents and the vocabulary.
        docs, vocab, raw_docs = preprocess()

        # TODO. Instantiate a GSDMM model and train using your preprocessed
        # documents and vocabulary. Set the parameters appropriately. To train,
        # use `fit` method of your model.
        # [YOUR CODE HERE]
        mdl = GSDMM()
        labels = mdl.fit(docs, len(vocab))
        # Save model

        with open(mdl_file, 'wb') as f:
            data = {
                'mdl': mdl,
                'docs': docs,
                'vocab': vocab,
                'raw_docs': raw_docs,
            }
            pickle.dump(data, f)

    # Analysis ================================================================

    # Following line should print the number of docs per cluster:
    # #docs per cluster: [2733, 3494, 1172, 2666, 3587, 2435, 4254, 5160, 2620, 1739]
    print('#docs per cluster: ', mdl.cluster_doc_count)

    # TODO. Print clusters ordered by #docs, and print the top k words in
    # term-frequency for each cluster.
    # Expected printouts are as below:
    # print('word dis per cluster: ', mdl.cluster_word_distribution)
    for index, cluster_words in enumerate(mdl.cluster_word_distribution):
        words = []
        for w in sorted(cluster_words, key=cluster_words.get, reverse=True):
            words.append(w)
        print("- Cluster #",index,words[:5])
    # - Cluster #7: ['health', 'care', 'study.', 'study', 'medical']
    # - Cluster #6: ['cell', 'cells', 'cancer', 'human', 'expression']
    # - Cluster #4: ['case', 'patients', 'treatment', 'cancer', 'cell']
    # - Cluster #1: ['patients', 'risk', 'study.', 'clinical', 'treatment']
    # - Cluster #0: ['analysis', 'based', 'detection', 'human', 'method']
    # - Cluster #3: ['patients', 'aortic', 'acute', 'coronary', 'artery']    e
    # - Cluster #8: ['synthesis', 'effect', 'based', 'novel', 'activity']
    # - Cluster #5: ['effects', 'patients', 'cognitive', 'study.', 'disease.']
    # - Cluster #9: ['analysis', 'new', 'novel', 'characterization', 'resistance']
    # - Cluster #2: ['effect', 'effects', 'water', 'carbon', 'performance']
    #
    # [YOUR CODE HERE]

    # Visualization ===========================================================
    # TODO. If everything's done correctly, following lines will generate a
    # visualization for the clusters. Google 'pyldavis' to see how it looks
    # like. Open the html file on your browser and take a screenshot. Include
    # the screenshot in your submission.
    vis_data = prepare_vis_data(mdl, docs, vocab, K=10)
    pyLDAvis.save_html(vis_data, 'clusters-vis.html')

    # Labels for document classification ======================================
    # TODO. Use the `choose_best_label` method of your model to label all the
    # documents. This will be used in the later classification task. labels
    # should be a list of cluster id's.
    labels = []
    for doc in docs:
        label, confidence = mdl.choose_best_label(doc)
        labels.append(label)
    print('Choosing the best label for each document...')
    # [YOUR CODE HERE]
    assert len(labels) == len(docs), "#labels should match #docs"

    # Document classification using SVM =======================================
    # From here, follow this [tutorial](https://tinyurl.com/ye9e88vx)

    # TODO. Prepare datasets for training and testing (Allocate 10% of your
    # data for testing). You can use scikit-learn `train_test_split`
    # (see. https://tinyurl.com/y4rnqugj)
    # [YOUR CODE HERE]
    X_train, X_test, y_train, y_test = train_test_split(raw_docs, labels, test_size=.1, random_state=1234)

    # TODO. Build a pipeline for document vectorization, as step 4 in the
    # tutorial. Use SGDClassifier for the SVM model.
    # [YOUR CODE HERE]
    # ex., text_clf_svm = Pipeline([ ... ])
    text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                             ('clf-svm',
                              SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])

    # TODO. Train. You should properly use the training set for your training,
    # and testing set for testing exclusively.
    print('Training a SVM document classifier...')
    # [YOUR CODE HERE]
    # ex., _ = text_clf_svm.fit()
    text_clf_svm.fit(X_train, y_train)

    # Test and Evaluate =======================================================
    # TODO. test your svm model using the test set, and print accuracy
    # [YOUR CODE HERE]
    # ex., predicted_svm = text_clf_svm.predict()
    predicted_svm = text_clf_svm.predict(X_test)
    print("predicted",predicted_svm)
    print("y_test: ",y_test[:3])
    score =  np.mean(np.array(predicted_svm) == np.array(y_test))  # Your evaluation score
    predicted_svm = None
    print('Accuracy: {:3f}'.format(score))

    # Extra credits ===========================================================
    # TODO (Optional). I've got 76% accuracy. You can do extra work to improve
    # accuracy, and the following will be the extra credits available towards
    # your final exam. You get more credits, if you achieve more.
    #



