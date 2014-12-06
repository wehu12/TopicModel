from gensim import corpora, models, similarities
from itertools import chain
import nltk
from nltk.corpus import stopwords
from operator import itemgetter
import re
import string
import os
import numpy as np
from collections import Counter

def preprocess(t):  # removes puctuations, stopwords etc. in a string; returns a list of words
    porter = nltk.PorterStemmer()
    stoplist = stopwords.words('english')
    exclude = string.punctuation
    exclude.replace('$',"")
    t = " ".join(t.split('\n'))
    t.translate(string.maketrans("",""), exclude)
    result = [porter.stem(word) for word in t.decode('utf-8','ignore').lower().split() if word not in stoplist]
    return result

def train_corpus(trainset):
    path = '/Users/linahu/Documents/Developer/Yelp/top500business/'+trainset
    files = [os.path.join(path,fn) for fn in next(os.walk(path))[2] if fn[-4:]==".txt"]
    texts=[]
    print "Processing files..."
    for (i,f) in enumerate(files): # each file contains all the reviews for one restaurant
        #split a file into individual reviews
        documents = open(f).read().split('\n\n')
        for d in documents:
            texts.append(preprocess(d))

        if i%100==0: print i,"files processed..."
    print "Building dictionary and corpus.."
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary,corpus

def save_models(savepath):
    print "Saving models..."
    dictionary.save(os.path.join(savepath,'reviews.dict'))
    corpora.MmCorpus.serialize(os.path.join(savepath,'review_corpus.mm'), corpus)
    lda.save(os.path.join(savepath,'lda1.model'))
    lda1.save(os.path.join(savepath,'lda2.model'))


def train_models(savepath): #train lda and lsi models, save the model on disk
    #dictionary,corpus =train_corpus('top500buz')
    #load corpus from disc
    dictionary = corpora.Dictionary.load(os.path.join(savepath,'reviews.dict'))
    corpus = corpora.MmCorpus(os.path.join(savepath,'review_corpus.mm'))
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    for n_topics in[5,10,15,20]:
        print "LSI result for n_topics=", n_topics
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
        with open(os.path.join(savepath,"output","lsi_topic"+str(n_topics)+".txt"),'w') as output:
            for (i,topic) in enumerate(lsi.print_topics(n_topics)):
                print "topic",i,":",topic
                output.write(' '.join([word.split('*')[1] for word in topic.split('+')]))
        print "Building lda models..."
        lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
        lda1 = models.LdaModel(corpus, id2word=dictionary, num_topics=n_topics)
        print "LDA - with tfidf - result for n_topics=",n_topics
        with open(os.path.join(savepath,"output","lda_tfidf_topic"+str(n_topics)+".txt"),'w') as output:
            for (i,topic) in enumerate(lda.print_topics(n_topics)):
                print "topic",i,":",topic
                output.write(' '.join([word.split('*')[1] for word in topic.split('+')]))
        print "LDA - without tfidf - result for n_topics=",n_topics
        with open(os.path.join(savepath,"output","lda_topic"+str(n_topics)+".txt"),'w') as output:
            for (i,topic) in enumerate(lda1.print_topics(n_topics)):
                print "topic",i,":",topic
                output.write(' '.join([word.split('*')[1] for word in topic.split('+')]))


    #    for i in range(0, n_topics):
     #       temp = lda.show_topic(i, 10)
      #      terms = []
       #     for term in temp:
        #        terms.append(term[1])
         #   print "Top 10 terms for topic #" + str(i) + ": "+ ", ".join(terms)

     # persist the model
     #save_models(savepath)


def load_models(savepath):
    dictionary = corpora.Dictionary.load(os.path.join(savepath,'reviews.dict'))
    #corpus = corpora.MmCorpus(os.path.join(savepath,'review_corpus.mm'))
    lda1 =models.LdaModel.load(os.path.join(savepath,'lda1.model'))
    lda2 = models.LdaModel.load(os.path.join(savepath,'lda2.model'))
    return dictionary,lda1,lda2

def getTopicForReview (lda,dictionary,review):
    words = preprocess(review)
    review_vec = dictionary.doc2bow(words)
    topic_vec = []
    topic_vec = lda[review_vec]
    word_count_array = np.empty((len(topic_vec), 2), dtype = np.object)
    for i in range(len(topic_vec)):
        word_count_array[i, 0] = topic_vec[i][0]
        word_count_array[i, 1] = topic_vec[i][1]
    idx = np.argsort(word_count_array[:, 1])
    idx = idx[::-1]
    word_count_array = word_count_array[idx]
    final = []
    final = lda.print_topic(word_count_array[0, 0], 10)
    review_topic = final.split('*') ## as format is like "probability * topic"
    return word_count_array[0,0]

def tagReviews(lda,dictionary,file):
    stoplist = stopwords.words('english')
    exclude = string.punctuation
    exclude.replace('$',"")
    documents = open(file).read().split('\n\n')
    result=[]
    for i in range(20):
        result.append([])
    for d in documents:
        d=d.translate(string.maketrans("",""), exclude)
        topic=getTopicForReview(lda,dictionary,d)
        result[topic] +=[ word for word in d.lower().split() if word not in stoplist]
    return result


def outputWordFreq(reviews,filename):
    with open(filename,'w') as output:
        output.write(",".join(["text", "count","topic"])+"\n")
        for (topic,words) in enumerate(reviews):
            freq = Counter(words).most_common(50)
            for word in freq:
                output.write(",".join([word[0], str(word[1]),str(topic)])+"\n")

def outputDist(reviews,filename):
    with open(filename,'w') as output:
        output.write(",".join(["topic", "pct","label"])+"\n")
        for (topic,words) in enumerate(reviews):
            if len(words)>0:
                output.write(",".join([str(topic), str(math.log(len(words))),"Topic "+str(topic)])+"\n")


#main program

savepath = '/Users/linahu/Documents/Developer/Yelp/TopicModel/models'
#train_models(path)
dictionary,lda1,lda2 = load_models(savepath)

#tag a few restaurants
N = 10
path = '/Users/linahu/Documents/Developer/Yelp/top500business/top500buz'
files = [os.path.join(path,fn) for fn in next(os.walk(path))[2] if fn[-4:]==".txt"]

reviews = tagReviews(lda1,dictionary,files[400]) #Switch Restaurant & Wine Bar
outputWordFreq(reviews,'/Users/linahu/Documents/Developer/Yelp/TopicModel/visualization/wordcount.csv')
outputDist(reviews,'/Users/linahu/Documents/Developer/Yelp/TopicModel/visualization/dist.csv')

#329 rain
reviews = tagReviews(lda1,dictionary,files[139]) #Golden Nugget hotel and casino
outputWordFreq(reviews,'/Users/linahu/Documents/Developer/Yelp/TopicModel/visualization/wordcount1.csv')
outputDist(reviews,'/Users/linahu/Documents/Developer/Yelp/TopicModel/visualization/dist1.csv')

