import pandas as pd
import sklearn.manifold
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.test.utils import common_texts
import re
import random
from tqdm import tqdm
import itertools
import textract
import matplotlib.pyplot as plt
from sklearn.decomposition import  PCA


def clean_data(sentence):

    sentence = re.sub(r'[^A-Za-z0-9\s.]', r'', str(sentence).lower())
    sentence = re.sub(r'\n', r' ', sentence)
    # sentence = " ".join([word for word in sentence.split() if word not in stopWords])

    return sentence
# one = open(r'projs.txt',encoding = "utf-8")
# data = one.read()
data =  textract.process('CV-ZiyuZhao-1101.docx')
# print(data)
data = data.splitlines()
data = list(filter(None, data))
data = pd.DataFrame(data)

data[0] = data[0].map(lambda x: clean_data(x))
tmp_corpus = data[0].map(lambda x: x.split('.'))

corpus = []
for i in range(len(tmp_corpus)):
    for line in tmp_corpus[i]:
        words = [x for x in line.split()]
        corpus.append(words)


print(corpus)
num_of_sentences = len(corpus)
num_of_words = 0
for line in corpus:
    num_of_words += len(line)

print('Num of sentences - %s'%(num_of_sentences))
print('Num of words - %s'%(num_of_words))

size = 100
window_size = 2
epochs = 100
min_count = 1
workers = 4

model = Word2Vec(corpus, sg=1, window=window_size, size=size, min_count=min_count, workers=workers, iter=epochs, sample=0.01)

model.build_vocab(sentences=corpus, update=True)

# Training the model
for i in tqdm(range(50)):
    model.train(sentences=corpus, epochs=50, total_examples=model.corpus_count)

model.save('w2v_model')
model = Word2Vec.load('w2v_model')

# t = sklearn.manifold.TSNE(n_components=2, random_state=0)
t = PCA(n_components=2)
all_word_vectors_matrix = model.wv.vectors



print(model.wv.vocab)
all_word_vectors_matrix_2d = t.fit_transform(all_word_vectors_matrix)



points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[model.wv.vocab[word].index])
            for word in model.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)


sns.set_context("poster")
points.plot.scatter("x", "y", s=10)
plt.show()