import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import pandas as pd
from shutil import copy2
import joblib
import argparse
import glob

#privide paths necessary for classification
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
debbie_classifier = joblib.load(work_dir + 'svm_model.pkl')
#load trained vectorizer
debbie_vectorizer = joblib.load(work_dir + 'count_vect.pkl')
#load trained transformer
debbie_transformer = joblib.load(work_dir + 'transformer.pkl')

#read input files into classifier
#test_set = [os.path.join(args.i, f) for f in os.listdir(args.i)]
#read input files into classifier,  recursive and only txt files.

#get subfolders that my contain txt files
folders = glob.glob(args.i + '/**/', recursive=True)
#add root in folders
folders.insert(0, args.i)

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
        test_count = debbie_vectorizer.transform(test_set)
        test_tfidf = debbie_transformer.transform(test_count)

        #fit the model
        predicted = debbie_classifier.predict(test_tfidf)

        # the results of the classification
        results = dict(zip(test_set, predicted))

        relevant_abstracts = []
        not_relevant_abstracts = []

        for key, value in results.items():
            if value == 'clinical':
                relevant_abstracts.append(key)
                filepathTokens = key.split('/')
                lastToken = filepathTokens[len(filepathTokens) - 1]
                copy2(key, args.o)
                with open (key, 'r+') as f, open(os.path.join(args.o, lastToken), 'r+') as c:
                    content = f.read()
                    c.write('study type= clinical (classifier)' + '\n' + content)

            elif value == 'non-clinical':
                relevant_abstracts.append(key)
                filepathTokens = key.split('/')
                lastToken = filepathTokens[len(filepathTokens) - 1]
                copy2(key, args.o)
                with open (key, 'r+') as f, open(os.path.join(args.o, lastToken), 'r+') as c:
                    content = f.read()
                    c.write('study type= non-clinical (classifier)' + '\n' + content)
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
