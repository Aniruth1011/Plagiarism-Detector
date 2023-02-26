#!/usr/bin/env python
# coding: utf-8



#!pip install gensim


import numpy as np
import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
stopword = stopwords.words('english')
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer




from gensim.models import Word2Vec
import gensim.downloader as api
w2v = api.load('word2vec-google-news-300')




def preprocessing(text):
    """
    This function will completely clean and return  the given raw text data by implementing 7 steps 
        1. Sentence Tokenzation
        2. Word Tokenization
        3. Removing Special Characters
        4. Remove Stopwords
        5. Lower All Words
        6. Remove irrevelant POS
        7. Lemmatization
    """
    sentences = sent_tokenize(text) #Tokenize Paragraph into sentences
    words = [word_tokenize(sentence) for sentence in sentences] #Tokenize each sentence into words
    corpus = [] #Stores each unique word for vacabulary
    for list_of_word in words:
        for each_word in list_of_word:
            corpus.append(each_word)
    corpus = [re.sub('[^a-zA-Z0-9]' , ' ' , word) for word in corpus] #Remove all Special Characters
    corpus = [word for word in corpus if word not in stopword] #Remove Stopwords
    corpus = [word for word in corpus if len(word)>1] #Remove any empty strings or single characters
    corpus = [word.lower() for word in corpus]
    irrevelant_speech_tags = ['CC' , 'IN' , 'DT' , 'PDT' ] #Irreveant Speech Tags : Ex : CC - Conjunction 
    corpus = [word[0] for word in nltk.pos_tag(corpus) if word[1] not in irrevelant_speech_tags]
    corpus = [word for word in corpus if word not in ['a' , 'an' , 'the']] #Remove Articles
    lemmatizer = WordNetLemmatizer() #Lemmatizer Object
    corpus = [lemmatizer.lemmatize(word) for word in corpus] #Lemmatize Words
    return corpus 


# In[4]:


def make_passages(text):
    """
    This function takes in raw text and returns a list full of passages from that given sentence
    A passage is defined as a group of sentences whose size if of at most 500 characters
    """
    sentences = sent_tokenize(text) #Tokenize Paragraph into sentences
    document_passages = []
    paragraph = ""
    for sentence in sentences:
        paragraph+=sentence
        if (len(paragraph)>500): #A Passage can have at most 500 characters
            document_passages.append(paragraph)
            paragraph = ""
        if (sentences.index(sentence) == len(sentences) - 1):
            #If we are at last sentence , then appen last passage directly
            document_passages.append(paragraph)
    return document_passages


# In[ ]:





# In[9]:


def ngrams(passage):
    """
    This function takes in the list of passages for a given file , 
    and then tokenizes it , removes stopwords and makes all remaining words into lowercase.
    It then returns a list containing all uni and bi grams of the following words.
    """
    cleaned_passage_ngrams = []
    corpus = word_tokenize(passage)
    corpus = [re.sub('[^a-zA-Z0-9]' , ' ' , word) for word in corpus] #Remove all Special Characters
    corpus = [word for word in corpus if word not in stopword] #Remove Stopwords
    corpus = [word for word in corpus if len(word)>1] #Remove any empty strings or single characters
    corpus = [word.lower() for word in corpus]
    cleaned_passage_ngrams.append((corpus))
    cleaned_passage_ngrams.append(list(nltk.bigrams(corpus)))
    cleaned_passage_ngrams_cleaned = []
    for ngram in cleaned_passage_ngrams:
        for item in ngram:
            cleaned_passage_ngrams_cleaned.append(item)
    return cleaned_passage_ngrams_cleaned




def matching_passage_ngrams(source_passage , check_passage):
    """
    This function is used to determine which set of passages 
    between many passages among a source and check set of passages
    has at least 40% matching uni and bi grams. 
    """
    doubtful_paragraphs_source = []
    doubtful_paragraphs_check = []
    source_passage_words = word_tokenize(source_passage)
    check_passage_words = word_tokenize(check_passage)
    ngram_source = ngrams((source_passage))
    ngram_check = ngrams((check_passage)) 
    count = 0
    for i in ngram_source:
        if i in ngram_check:
            count+=1 
    avg_words = ( len(ngram_source) + len(ngram_check) )/2
    return (count*100/avg_words)  #Gives Percentage of general similarity found




def get_matching_passages(source_passages , check_passages):
    """
    Find only those passages that have similarity between them
    """
    source_plagiarised_passage = []
    check_plagiarised_passage = []
    for source in source_passages:
        for check in check_passages:
            similarity = matching_passage_ngrams(source , check)
            if (similarity>=30):
                #There is considerable similarity . There is doubt that considerable similarity exists
                #Hence , We drill down these passages for further checking
                source_plagiarised_passage.append(source_passages.index(source))
                check_plagiarised_passage.append(check_passages.index(check))
    return (source_plagiarised_passage , check_plagiarised_passage)




def find_plagiarised_sentences(source_passage_ind , check_passage_ind , source_passages , check_passages):
    """
        This function will return 2 list of lists 
        The first list will give information on those sentences which seem to be plagiarised , but we are not so sure. 
        So we will go for a semantic analysis
        The second list will give information on those sentences which are definitely plagiarised
        The format will be 
        [Source_Passage_index , check_Passage_index , source_sentence_index , check_sentence_index]
    """
    copy_pasted = []
    semantic_doubt = []
    for s_index in source_passage_ind:
        s_passage = source_passages[s_index]
        s_passage = s_passage.split('.')
        for c_index in check_passage_ind:
            c_passage = check_passages[c_index]
            c_passage = c_passage.split('.')
            for s_sentence in s_passage:
                s_ngrams = ngrams(s_sentence)
                for c_sentence in c_passage:
                    c_ngrams = ngrams(c_sentence)
                    count = 0
                    for s_gram in s_ngrams:
                        for c_gram in c_ngrams :
                            if (s_gram == c_gram):
                                count+=1
                    avg_words = ( len(s_ngrams) + len(c_ngrams) )/2
                    if (avg_words != 0):
                        percentage_similarity = (count*100/avg_words)  #Gives Percentage of general similarity found
                    else:
                        percentage_similarity = 0
                    if (percentage_similarity < 20):
                        #These 2 sentences are not similar enough
                        continue
                    elif (percentage_similarity<30):
                        #Doubtful  . Hence we go for semantic checking
                        semantic_doubt.append([s_index , c_index , s_passage.index(s_sentence) , c_passage.index(c_sentence)])
                        #We can determine the required sentences under consideration using this nomenclature
                    else:
                        #Definitely Similar . Copy Pasting has happened
                        copy_pasted.append([s_index , c_index , s_passage.index(s_sentence) , c_passage.index(c_sentence)])
    return semantic_doubt , copy_pasted




def check_semantics(semantic_doubt , source_passages , check_passages):
    for information in semantic_doubt:
        p_index , c_index , sp_index , cp_index = information[0] , information[1] , information[2] , information[3]
        s_sentence = source_passages[p_index].split('.')[sp_index]
        c_sentence = check_passages[c_index].split('.')[cp_index]
        s_sentence = preprocessing(s_sentence)
        c_sentence = preprocessing(c_sentence)
        complete_match = 0
        semantic_match = 0
        semantic_value = []
        semantic_word = []
        indices = []
        semantic_mean = []
        for i in (s_sentence):
            for j in (c_sentence):
                if (i==j):
                    complete_match+=1
                else:
                    #print(i , j)
                    synset_words1 = [synonyms[0] for synonyms in w2v.most_similar(i)]
                    synset_proba1 = [synonyms[1] for synonyms in w2v.most_similar(i)]
                    synset_words2 = [synonyms[0] for synonyms in w2v.most_similar(j)]
                    synset_proba2 = [synonyms[1] for synonyms in w2v.most_similar(j)]
                    if j in synset_words1:
                        #There is a semantic_match
                        semantic_match+=1
                        semantic_value.append(synset_proba1[synset_proba1.index(j)])
                        semantic_word.append(synset_words1[synset_words1.index(j)])
                        #indices.append([])
                    if i in synset_words2:
                        semantic_match+=1
                        semantic_value.append(synset_proba2[(synset_words2.index(i))])
                        semantic_word.append(synset_words2[synset_words2.index(i)])
            semantic_mean.append(np.mean(semantic_word) * 100)
    return semantic_mean




def confirm_semantic(semantic_score , semantic_doubt):
    confirmed = []
    for semantic_mean in semantic_score:
        if semantic_mean > 0.4 :
            confirmed.append(semantic_doubt[semantic_score.index(semantic_mean)])
    return confirmed




def len_of_check(check_passages):
    c = 0
    for passage in check_passages:
        for j in (passage.split('.')):
            c+=1
    return c


def solution(source_read , check_read):
    corpus_source = preprocessing(source_read)
    corpus_check =  preprocessing(check_read)
    source_passages = make_passages(source_read)
    check_passages = make_passages(check_read)
    source_passage_ind , check_passage_ind = get_matching_passages(source_passages , check_passages)
    semantic_doubt , copy_pasted = find_plagiarised_sentences(source_passage_ind , check_passage_ind , source_passages , check_passages)
    copy_pasted_un = []
    for unique in copy_pasted:
        if unique not in copy_pasted_un:
            copy_pasted_un.append(unique)
    confirmed = []
    try:
        if semantic_doubt == None:
            pass
        else:
            semantic_score = check_semantics(semantic_doubt , source_passages , check_passages)
            confirmed = confirm_semantic(semantic_score , semantic_doubt)
    except KeyError:
        pass
    semantic_doubt = confirmed.copy()
    semantic_un = []
    for unique in semantic_doubt:
        if unique not in semantic_un:
            semantic_un.append(unique) 
    source_sent = []
    check_sent = []
    for i in copy_pasted_un:
        source_sent.append(source_passages[i[0]].split('.')[i[2]])
        check_sent.append(check_passages[i[1]].split('.')[i[3]])
    for i in semantic_un:
        source_sent.append(source_passages[i[0]].split('.')[i[2]])
        check_sent.append(check_passages[i[1]].split('.')[i[3]])
    return source_sent , check_sent
