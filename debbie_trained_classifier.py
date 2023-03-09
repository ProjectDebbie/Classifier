from transformers import pipeline
import os
import sys
import argparse
from glob import glob
from pathlib import Path
from shutil import copy2

def run_classifier(input, output, checkpoint):
    pipe = pipeline("text-classification", model=checkpoint) 
    print("create pipeline done")
    #get subfolders that my contain txt files
    folders = glob(input + '/**/', recursive=True)
    #set counters
    total=0
    total_relevant=0
    total_non_relevant=0
    #iterate over folders and classify
    for f in folders:
        test_set = glob(f + '/*.txt', recursive=True)
        if(len(test_set)>0):
            #extract features new abstracts
            #test_count = debbie_vectorizer.transform(test_set)
            #test_tfidf = debbie_transformer.transform(test_count)
            #Read text from file
            X_test = [Path(file).read_text() for file in test_set]
            #run model
            y_predict = pipe(X_test, padding=True, truncation=True)
            #for index,abstracts in enumerate(X_test):
                
            # the results of the classification
            results = dict(zip(test_set, y_predict))
            print(results)
            relevant_abstracts = []
            not_relevant_abstracts = []
            for key, value in results.items():
                if value['label'] == 'clinical':
                    relevant_abstracts.append(key)
                    filepathTokens = key.split('/')
                    lastToken = filepathTokens[len(filepathTokens) - 1]
                    copy2(key, output)
                    meta=key.replace('.txt','.meta')
                    metafile=lastToken.replace('.txt','.meta')
                    copy2(meta, output)
                    with open (meta, 'r+') as f, open(os.path.join(output, metafile), 'r+') as c:
                        content = f.read()
                        c.write( content  +  'clinical' + '\n')
    
                elif value['label'] == 'non-clinical':
                    relevant_abstracts.append(key)
                    filepathTokens = key.split('/')
                    lastToken = filepathTokens[len(filepathTokens) - 1]
                    copy2(key, output)
                    meta=key.replace('.txt','.meta')
                    metafile=lastToken.replace('.txt','.meta')
                    copy2(meta, output)
                    with open (meta, 'r+') as f, open(os.path.join(output, metafile), 'r+') as c:
                        content = f.read()
                        c.write( content +  'non-clinical' + '\n')
                else:
                    not_relevant_abstracts.append(key)
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

if __name__ == '__main__':
    print("Start main process ")
    #add -term parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help= 'Input path')
    parser.add_argument('-o', help= 'Output path')
    parser.add_argument('-m', help= 'Model')
    
    #if term given, use term in search
    #if no term given, return only journal articles in english
    args = parser.parse_args()
    if (args.i == None):
        print("Please set the input folder")
        sys.exit(1)
        if not os.path.exists(args.i):
            print("Input folder not exist")
            sys.exit(1)
    if (args.o == None):
        print("Please set the output folder")
        sys.exit(1)
    if (args.m == None):
        print("Default model will be used")
        args.m = "javicorvi/DEBBIE_BioBERT"        
    print("Start classifier process")
    #create output folder if not exist
    if not os.path.exists(args.o):
        os.makedirs(args.o)
    print("Input Dir: " + args.i)
    print("Output Dir: " + args.o)
    print("Model: " + args.m)
    run_classifier(args.i, args.o, args.m)    
    print("Finish process")
