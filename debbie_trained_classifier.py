import os
import numpy as np
import pandas as pd
from shutil import copy2
import joblib
import argparse
import glob
from pathlib import Path
#for text pre-processing
import re, string
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#Functions for prepare the text

#convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text
 
# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
   

def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    #a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) + '_' + get_wordnet_pos(tag[1]) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    #a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) + '_' + tag[1] for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
    #return lemmatizer(preprocess(string))


#Start with classification process

#provide paths necessary for classification
parser = argparse.ArgumentParser()
parser.add_argument('-i', help= 'paste path to folder of pubmed abstracts')
parser.add_argument('-o', help= 'paste path to folder of output folder')
parser.add_argument('-w', help= 'paste path to workdir')
args = parser.parse_args()

#create output folder if not exist
if not os.path.exists(args.o):
    os.makedirs(args.o)

#work directory
if (args.w == None):
    work_dir=""
else:
    work_dir= args.w + "/"

#load trained model
debbie_model = joblib.load(work_dir + 'debbie_model.pkl')
#load trained vectorizer
debbie_vectorizer = joblib.load(work_dir + 'debbie_vectorizer.pkl')
#load trained transformer
#debbie_transformer = joblib.load(work_dir + 'transformer.pkl')

#get subfolders that my contain txt files
folders = glob.glob(args.i + '/**/', recursive=True)
#add root in folders
#folders.insert(0, args.i)

#set counters
total=0
total_relevant=0
total_non_relevant=0

#iterate over folders and classify
for f in folders:
    print ('folder:' + f)
    test_set = glob.glob(f + '/*.txt', recursive=True)
    if(len(test_set)>0):
        #extract features new abstracts
        #test_count = debbie_vectorizer.transform(test_set)
        #test_tfidf = debbie_transformer.transform(test_count)
        #Read text from file
        X_test = [Path(file).read_text() for file in test_set]
        #Apply corresponding preprocessing
        X_test_preprocessed = [finalpreprocess(string) for string in X_test]
        #vectorization 
        X_test_vectors_tfidf = debbie_vectorizer.transform(X_test_preprocessed)
        #run model
        y_predict = debbie_model.predict(X_test_vectors_tfidf)
        
        # the results of the classification
        results = dict(zip(test_set, y_predict))

        relevant_abstracts = []
        not_relevant_abstracts = []

        for key, value in results.items():
            if value == 'clinical':
                relevant_abstracts.append(key)
                filepathTokens = key.split('/')
                lastToken = filepathTokens[len(filepathTokens) - 1]
                copy2(key, args.o)
                meta=key.replace('.txt','.meta')
                metafile=lastToken.replace('.txt','.meta')
                copy2(meta, args.o)
                with open (meta, 'r+') as f, open(os.path.join(args.o, metafile), 'r+') as c:
                    content = f.read()
                    c.write( content  +  'clinical' + '\n')

            elif value == 'non-clinical':
                relevant_abstracts.append(key)
                filepathTokens = key.split('/')
                lastToken = filepathTokens[len(filepathTokens) - 1]
                copy2(key, args.o)
                meta=key.replace('.txt','.meta')
                metafile=lastToken.replace('.txt','.meta')
                copy2(meta, args.o)
                with open (meta, 'r+') as f, open(os.path.join(args.o, metafile), 'r+') as c:
                    content = f.read()
                    c.write( content +  'non-clinical' + '\n')
            else:
                not_relevant_abstracts.append(key)

        #save relevant abstracts to folder
        # for filename in test_set:
        #     if str(filename) in relevant_abstracts:
        #         copy2(filename, args.o)

        print('total number of abstracts collected in folder', (len(relevant_abstracts)+len(not_relevant_abstracts)))
        print('number of relevant abstracts in folder:', len(relevant_abstracts))
        print('number of non-relevant abstracts in folder:', len(not_relevant_abstracts))
        total =  total  + len(relevant_abstracts)+len(not_relevant_abstracts)
        total_relevant =  total_relevant  + len(relevant_abstracts)
        total_non_relevant =  total_non_relevant  + len(not_relevant_abstracts)
    else:
        print('Emtpy Folder')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('total number of abstracts collected:', total)
print('number of relevant abstracts:', total_relevant)
print('number of non-relevant abstracts:', total_non_relevant)
