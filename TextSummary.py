import numpy as np
import pandas as pd
import nltk 
#nltk.download('punkt') 
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class TextSummarizer:
    def __init__(self, *args, **kwargs):
        pass

    def summarizeText(self,text,number):
        mytextlist=[text]

        sentences = [] 
        for s in mytextlist:
            sentences.append(sent_tokenize(s))

        sentences = [y for x in sentences for y in x] 

        sentences[:5]

        # Extract word vectors
        word_embeddings = {}
        f = open('glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
        f.close()

        len(word_embeddings)

        # remove punctuations, numbers and special characters
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

        # make alphabets lowercase
        clean_sentences = [s.lower() for s in clean_sentences]

        #only if you did'nt download... Mani you've downloaded
        #nltk.download('stopwords')

        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')

        # function to remove stopwords
        def remove_stopwords(sen):
            sen_new = " ".join([i for i in sen if i not in stop_words])
            return sen_new

        # remove stopwords from the sentences
        clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

        sentence_vectors = []
        for i in clean_sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)

        #print("hi")
        # similarity matrix
        sim_mat = np.zeros([len(sentences), len(sentences)])

        from sklearn.metrics.pairwise import cosine_similarity

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

        import networkx as nx

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)

        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)


        #for reference
        
        # Extract top 10 sentences as the summary
        #print(len(ranked_sentences))
        #print(number)
        k=0
        count = int(number)

        result=[]
        for i in ranked_sentences:
            #print(i[1])
            result.append(i[1])
            k+=1
            #print(k)
            if(k==count):
                break
        return result

obj = TextSummarizer()
mytext= "Maria "