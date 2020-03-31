import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import pandas as pd
from shutil import copy2 

##INPUT FILES HERE ##
##training set = abstract_lake_training_srt folder in classifier repository
training_set = [os.path.join("/Users/austinmckitrick/git/debbie/DEBBIE_DATA/abstract_lake/abstract_lake_training_set", f) for f in os.listdir("/Users/austinmckitrick/git/debbie/DEBBIE_DATA/abstract_lake/abstract_lake_training_set")]
##test_set = pubmed abstracts folder 
test_set = [os.path.join("/Users/austinmckitrick/git/debbie/DEBBIE_DATA/abstract_lake/abstract_lake_polydioxanone/abstracts", f) for f in os.listdir("/Users/austinmckitrick/git/debbie/DEBBIE_DATA/abstract_lake/abstract_lake_polydioxanone/abstracts")]
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

##make a dictionary from the training set:
count_vect = CountVectorizer(input='filename', ngram_range=(1,2), max_df=0.9, min_df=0.0, stop_words= 'english') #decode_error='ignore' input='filename', encoding='cp850'
X_train_counts = count_vect.fit_transform(training_set)

#tf_idf calculation for the words in the training_set
tf_transformer = TfidfTransformer(use_idf= False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf  = tfidf_transformer.fit_transform(X_train_counts)

#make labels for training and test set
training_set_labels, file_list = make_labels(training_set)
test_set_labels, test_file_list = make_labels(test_set)

#train the model
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier().fit(X_train_tfidf, training_set_labels)

#bag of word and tf_idf of test set
X_test_count = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_test_count)

#fit the model
predicted = sgd_clf.predict(X_test_tfidf)

#the results of the classification
results = dict(zip(test_set, predicted))
df_results = pd.DataFrame.from_dict(results, orient='index')
df_results.to_csv("/Users/austinmckitrick/git/debbie/DEBBIE_DATA/classification_results_SGD1.csv", sep=' ')

relevant_abstracts = []
not_relevant_abstracts = []

for key, value in results.items():
    if value == 0.0:
        relevant_abstracts.append(key)
    elif value == 1.0:
        not_relevant_abstracts.append(key)

print('number of relevant abstracts:', len(relevant_abstracts))
print('number of non-relevant abstracts:', len(not_relevant_abstracts))

for filename in os.listdir('/Users/austinmckitrick/git/debbie/DEBBIE_DATA/abstract_lake/abstract_lake_polydioxanone/abstracts'):
    file_to_copy = os.path.join('/Users/austinmckitrick/git/debbie/DEBBIE_DATA/abstract_lake/abstract_lake_polydioxanone/abstracts', filename)
    if str(file_to_copy) in relevant_abstracts:
        copy2(file_to_copy, "/Users/austinmckitrick/git/debbie/Classifier/classifier_output")
#     else:
#         copy2(file_to_copy, "path_not_relevant")


