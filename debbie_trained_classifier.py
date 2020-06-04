import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import pandas as pd
from shutil import copy2 
import joblib
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-i', help= 'paste path to folder of pubmed abstracts')
parser.add_argument('-o', help= 'paste path to folder of output folder')
parser.add_argument('-w', help= 'paste path to workdir')
args = parser.parse_args()

#read input files into classifier 
#test_set = [os.path.join(args.i, f) for f in os.listdir(args.i)]
#read input files into classifier,  recursive and only txt files. 
test_set = glob.glob(args.i + '/**/*.txt', recursive=True)

#create output folder if not exist
if not os.path.exists(args.o):
    os.makedirs(args.o)

#work directory 
if (args.w == None):
    work_dir=""
else:
    work_dir= args.w + "/"

#making labels 
def make_labels(data):
    files = data
    file_list = []
    train_labels = np.zeros(len(files))
    count = 0
    docID = 0
    for fil in files:
        file_list.append(fil)
        train_labels[docID] = 0
        filepathTokens = fil.split('/')
        lastToken = filepathTokens[len(filepathTokens) - 1]
        if lastToken.endswith("rand.txt"):
            train_labels[docID] = 1
            count = count + 1
        docID = docID + 1
    return train_labels, file_list

#make labels for test set
test_set_labels, test_file_list = make_labels(test_set)

#load trained model
debbie_classifier = joblib.load(work_dir + 'svm_model.pkl')
#load trained vectorizer
debbie_vectorizer = joblib.load(work_dir + 'count_vect.pkl')
#load trained transformer
debbie_transformer = joblib.load(work_dir + 'transformer.pkl')

test_count = debbie_vectorizer.transform(test_set)
test_tfidf = debbie_transformer.transform(test_count)

#fit the model
predicted = debbie_classifier.predict(test_tfidf)

# the results of the classification
results = dict(zip(test_set, predicted))
# df_results = pd.DataFrame.from_dict(results, orient='index')
# df_results.to_csv("/Users/austinmckitrick/git/debbie/DEBBIE_DATA/debbie_classifier_results.csv", sep=',')

relevant_abstracts = []
not_relevant_abstracts = []

for key, value in results.items():
    if value == 0.0:
        relevant_abstracts.append(key)
    elif value == 1.0:
        not_relevant_abstracts.append(key)

print('total number of abstracts collected', (len(relevant_abstracts)+len(not_relevant_abstracts)))
print('number of relevant abstracts:', len(relevant_abstracts))
print('number of non-relevant abstracts:', len(not_relevant_abstracts))

#save relevant abstracts to folder
for filename in test_set:
    if str(filename) in relevant_abstracts:
        copy2(filename, args.o)