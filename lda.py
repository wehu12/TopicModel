from gensim import corpora, models, similarities
from itertools import chain
import nltk
from nltk.corpus import stopwords
from operator import itemgetter
import re
import os

path = '/Users/linahu/Documents/Developer/Yelp/name_reviews/'
files = [os.path.join(path,fn) for fn in next(os.walk(path))[2]]

porter = nltk.PorterStemmer()

stoplist = stopwords.words('english')
texts=[]
for f in files:
    document = open(f).read()
    texts.append([porter.stem(word) for word in document.decode('utf-8','ignore').lower().split() if word not in stoplist])

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

#lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
#lsi.print_topics(20)

n_topics = 10
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)


for i in range(0, n_topics):
    temp = lda.show_topic(i, 10)
    terms = []
    for term in temp:
        terms.append(term[1])
    print "Top 10 terms for topic #" + str(i) + ": "+ ", ".join(terms)

for (word,score) in tfidf[corpus[2]]:
    print dictionary[word],':'+score




