# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy as np

# shuffle
from random import shuffle

# logging
import logging
import os.path
import sys
import cPickle as pickle

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

sources = {'../train-neg.txt': 'TRAIN_NEG', '../train-pos.txt': 'TRAIN_POS', '../train-unsup.txt': 'TRAIN_UNS'}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

model.build_vocab(sentences.to_array())

for epoch in range(1):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm())

def get_vectors(model, corpus, size):
    for z in corpus:
        print z
    vecs = [np.array(model[z[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

#print np.array(model['TRAIN_POS_7']).reshape((1, 100))
vec = get_vectors(model, sentences.to_array(), 100)
#print vec
model.save('../imdb.d2v')

