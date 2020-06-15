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

total=0
total_relevant=0
total_non_relevant=0

#iterate over folders and classify
for f in folders:
    print ('folder:' + f)
    test_set = glob.glob(f + '/*.txt', recursive=True)
    if(len(test_set)>0):
        #make labels for test set
        test_set_labels, test_file_list = make_labels(test_set)
        
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
                
        #save relevant abstracts to folder
        for filename in test_set:
            if str(filename) in relevant_abstracts:
                copy2(filename, args.o)        
        
        print('total number of abstracts collected in folder', (len(relevant_abstracts)+len(not_relevant_abstracts)))
        print('number of relevant abstracts in folder:', len(relevant_abstracts))
        print('number of non-relevant abstracts in folder:', len(not_relevant_abstracts))    
        total =  total  + len(relevant_abstracts)+len(not_relevant_abstracts)  
        total_relevant =  total_relevant  + len(relevant_abstracts)  
        total_non_relevant =  total_non_relevant  + len(not_relevant_abstracts)
    else:
        print('Emtpy Folder')
    #make labels for test set
    test_set_labels, test_file_list = make_labels(test_set)
    
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
            
    #save relevant abstracts to folder
    for filename in test_set:
        if str(filename) in relevant_abstracts:
            copy2(filename, args.o)        
    
    print('total number of abstracts collected in folder', (len(relevant_abstracts)+len(not_relevant_abstracts)))
    print('number of relevant abstracts in folder:', len(relevant_abstracts))
    print('number of non-relevant abstracts in folder:', len(not_relevant_abstracts))    
    total =  total  + len(relevant_abstracts)+len(not_relevant_abstracts)  
    total_relevant =  total_relevant  + len(relevant_abstracts)  
    total_non_relevant =  total_non_relevant  + len(not_relevant_abstracts)

print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('total number of abstracts collected in folder', total)
print('number of relevant abstracts in folder:', total_relevant)
print('number of non-relevant abstracts in folder:', total_non_relevant)
