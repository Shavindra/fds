# packages to store and manipulate data
import gensim
import pandas as pd
import json

pd.set_option('display.max_columns', None)

# Load data
#../merged-cleansed.json
print('Loading data file...')
with open('../merged-cleansed.json') as f:
    file_data = json.load(f)

##### Select random sample #####
print('Selecting sample data...', str(len(file_data)))

import random
data = random.sample(file_data, 500)
# data = file_data

##### Create twitter dataframe #####
tweets = pd.DataFrame({
    'created_at': list(map(lambda tweet: tweet['created_at'], data)),
    'text': list(map(lambda tweet: tweet['text'], data)),
    'lang': list(map(lambda tweet: tweet['user']['lang'], data)),
    'screen_name': list(map(lambda tweet: tweet['user']['screen_name'], data)),
    'retweet_count': list(map(lambda tweet: tweet['retweet_count'], data)),
    'favorite_count': list(map(lambda tweet: tweet['favorite_count'], data)),
})

# tweets.sort_values(by='lang', ascending=True, inplace=True)
tweets_text = tweets['text'].tolist()

print('Processing tweet texts...')

#####  Process data #####
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

##### Stop words #####
# update stopwords with list of possible irrelevant values.
from numpy import loadtxt
custom_stopwords = []
with open('custom-stopwords.txt') as f:
    custom_stopwords = f.read().split('\n')

# print(custom_stopwords)

##### tokenize using gensim #####
processed_list = [gensim.utils.simple_preprocess(txt, deacc=True, min_len=3) for txt in tweets_text]

# Create a set of stopwords
stop_words = stopwords.words('english')
stop_words.extend(custom_stopwords)
stop = set(stop_words)

# print(stop)

# Create a set of punctuation words
exclude = set(string.punctuation)

# This is the function makeing the lemmatization
lemma = WordNetLemmatizer()


print('Processing clean docs...')

# In this function we perform the entire cleaning
def clean(doc_list):
    stop_free = " ".join(i for i in doc_list if i not in stop)
    punc_free = ' '.join(ch for ch in stop_free.split() if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Importing Gensim
import gensim
from gensim import corpora

doc_clean = [clean(word_list).split() for word_list in processed_list]

##### Creating LDA model #####
# Creating the term dictionary of our courpus, where every unique term is assigned an index
dictionary = corpora.Dictionary(doc_clean)

print('Term matrix...')
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
def term_matrix(doc):
    return dictionary.doc2bow(doc)

# Corpus
doc_term_matrix = [term_matrix(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
print('Training model...')
num_topics = 3
num_words = 4
ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word = dictionary, passes=100)

print('Processing topics...')
# Print 2 topics and describe then with 4 words.
topics = ldamodel.print_topics(num_topics=num_topics, num_words=num_words)


##### Saving information #####
print('Printing topics...')

import time

print('Saving model...')
t = str(time.time())
ldamodel.save( t + 'lda.model')

print('Saving topics...')
with open('topic_file' + t, 'w') as topic_file:
    top_topics=ldamodel.top_topics(doc_term_matrix)
    topic_file.write('\n'.join('%s %s' %topic for topic in top_topics))


with open('meta' + t, 'w') as topic_file:
    top_topics=ldamodel.top_topics(doc_term_matrix)
    topic_file.write('\n num_topics ' + str(num_topics))
    topic_file.write('\n num_words ' + str(num_words))

i=0
for topic in topics:
    print ("Topic",i ,"->", topic)
    i+=1


##### Get topic in each sentence  #####

def format_topics_sentences(ldamodel=ldamodel, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=doc_term_matrix, texts=doc_clean)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)

##### Visualise topics #####
print('Visualising topics...')
# Visualise
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#15visualizethetopicskeywords

import IPython # still required
import pyLDAvis
from pyLDAvis import gensim

# Visualize the topics
# Visualize the topics in notebook
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# vis
vis = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, mds='mmds')
pyLDAvis.show(vis)

pyLDAvis.save_html(vis, 'topics-lda.html')

print('Save visualisation to json...')
with open('pylda-vis' + t + '.json', 'w') as vis_json:
    vis_json.write(pyLDAvis.save_json(vis))
