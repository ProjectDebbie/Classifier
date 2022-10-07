from Bio import Entrez
import argparse
import os
import sys
import xml.etree.ElementTree as ET
import codecs
import urllib

####
####This script was used to download the abstracts text files from the list of PMIS.
####It is not parametrized and was used only once to generate the folders:  
####

import shutil
directory = "clarence-dataset/training-set/abstracts/laboratory"
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        shutil.copyfile(f, "clarence-dataset/training-set/abstracts/"+filename.replace(".txt", "_non-clinical.txt"))

api_key="1cab2a4456d4e64e48a7c83a9cc9c101dc09"
const_no_abstract = 0
def remove_invalid_characters(text):
    text = text.replace("\n"," ").replace("\t"," ").replace("\r"," ")    
    return text

def readTitle(title_xml):
    if(title_xml!=None):
        title=''.join(title_xml.itertext())
        return title
    return ''

def readAbstract(abstract_xml):
    abstract = []
    for abstractText in abstract_xml.findall("AbstractText"):
        abstract.append("".join(abstractText.itertext()))
    abstract = " ".join(abstract)  
    return abstract

def search_query(query,reldate):
    Entrez.email = 'email@bsc.es'
    if (reldate is not None):
        handle = Entrez.esearch(db='pubmed',  
                                rettype="xml", 
                                retmode="text", 
                                term=query,
                                usehistory="y",
                                reldate=reldate)
    else:
        handle = Entrez.esearch(db='pubmed',  
                                rettype="xml", 
                                retmode="text", 
                                id=query,
                                usehistory="y")
    results = Entrez.read(handle)
    return results

#### fetch_details function will retireve information for each PMID it receives
# retrieves year, title, abstract text as of now
def fetch_details(query_background, outputfolder, registry_file,type):
    ids_list = []
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    values = {'db':'pubmed', 'rettype':'xml', 'retmode':'xml', 'id':query_background}
    data = urllib.parse.urlencode(values)
    data = data.encode('ascii') # data should be bytes
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as response:
        rpub = response.read()
        xmlResp = rpub.decode("utf-8") 
        docXml = ET.fromstring(xmlResp)
        with open(registry_file,'a') as list_files_standardized:
            for article in docXml.findall("PubmedArticle"):
                try:
                    year = "0000"
                    month = ''
                    title = ''
                    pmid = article.find("MedlineCitation").find("PMID").text
                    if(pmid+'.txt' not in ids_list):
                        article_xml = article.find("MedlineCitation").find("Article")
                        try: 
                            year = article_xml.find('Journal').find('JournalIssue').find('PubDate').find('Year').text
                        except Exception: 
                            try: 
                                medline_date = article_xml.find('Journal').find('JournalIssue').find('PubDate').find('MedlineDate').text
                                year = medline_date[0:4]
                            except Exception: 
                                print("Error with year:  " + pmid)  
                                year ='0000'
                        try: month = article_xml.find('Journal').find('JournalIssue').find('PubDate').find('Month').text
                        except Exception: 
                            try: 
                                medline_date = article_xml.find('Journal').find('JournalIssue').find('PubDate').find('MedlineDate').text
                                month = medline_date[5:]
                            except Exception:  
                                month =''
                        abstract_xml = article_xml.find("Abstract")
                        if(abstract_xml is not None):
                            abstract = readAbstract(abstract_xml)
                            title = readTitle(article_xml.find('ArticleTitle'))
                            if(abstract!=''):
                                with codecs.open(outputfolder+"/"+  pmid + '_' +type +'.txt', 'w',encoding='utf8') as txt_file:
                                    txt_file.write(remove_invalid_characters(title) + '\n' + remove_invalid_characters(abstract) + '\n')
                                    txt_file.flush()
                                    txt_file.close() 
                                
                                    
                        #else:
                            #print("Error with pmid no abstract:  " + pmid)      
                except Exception as inst:
                    print("Error with pmid: " + pmid)

def search(term_search, reldate, output, file_to_write):    
    retstart = 0
    retmax = 10000
    results = search_query(term_search,reldate)
    webenv=results["WebEnv"]
    querykey=results["QueryKey"]
    count = int(results['Count'])
    print ("Total abstracts in search query: " + str(count))
    search_while=True  
    first_time = True
    while (retstart < count):
        print ("Search from  " + str(retstart) +  " to " + str(retstart + retmax))
        results = fetch_details(webenv, querykey, retstart, retmax, output, ids_list, file_to_write, reldate)
        retstart=retstart+retmax
    print("Finish process")

 
if __name__ == '__main__':
    #add -term parameter
    
    
    
    
    background = open("dataset/background.txt", "r")
    query_background = background.read().split('\n')
    os.makedirs('dataset/background', exist_ok=True)
    output_background='dataset/background'
    clinical_gs_1092 = open("dataset/clinical_gs_1092.txt", "r")
    query_clinical_gs_1092 = clinical_gs_1092.read().split('\n')
    os.makedirs('dataset/clinical_gs_1092', exist_ok=True)
    output_clinical_gs_1092='clinical_gs_1092'
    laboratory_gs_1213 = open("dataset/laboratory_gs_1213.txt", "r")
    query_laboratory_gs_1213 = laboratory_gs_1213.read().split('\n')
    os.makedirs('dataset/laboratory_gs_1213', exist_ok=True)
    output_laboratory_gs_1213='dataset/laboratory_gs_1213'
    file_to_write = 'dataset/ids.txt'
    #if term given, use term in search
    #if no term given, return only journal articles in english
    
    
    
    
    #create output folder if not exist
    reldate=None
    
    
    fetch_details(query_background, output_background, file_to_write,'background')
    fetch_details(query_clinical_gs_1092, output_clinical_gs_1092, file_to_write,'clinical')
    fetch_details(query_laboratory_gs_1213,  output_laboratory_gs_1213, file_to_write,'non-clinical')
    
    #clarence data set
    
    clarence_background_training =  open("clarence-dataset/training-set/background_only_1076.txt")
    clarence_clinical_training =  open("clarence-dataset/training-set/clinical_only_1076.txt")
    
    clarence_background_testing =  open("clarence-dataset/testing-set/non biomaterials.txt")
    clarence_clinical_testing =  open("clarence-dataset/testing-set/clinical.txt")
    clarence_laboratory_testing =  open("clarence-dataset/testing-set/in_vivo_and_in_vitro.txt")
    clarence_laboratoryvitro_testing =  open("clarence-dataset/testing-set/in_vivo.txt")
    clarence_laboratoryvivo_testing =  open("clarence-dataset/testing-set/in_vitro.txt")
    
    
    query_clarence_background_training = clarence_background_training.read().split('\n')
    query_clarence_clinical_training = clarence_clinical_training.read().split('\n')
    
    query_clarence_background_testing = clarence_background_testing.read().split('\n')
    query_clarence_clinical_testing = clarence_clinical_testing.read().split('\n')
    query_clarence_laboratory_testing = clarence_laboratory_testing.read().split('\n')
    query_clarence_laboratoryvitro_testing = clarence_laboratoryvitro_testing.read().split('\n')
    query_clarence_laboratoryvivo_testing = clarence_laboratoryvivo_testing.read().split('\n')

    
    output_training = "clarence-dataset/training-set/abstracts"
    os.makedirs(output_training, exist_ok=True)
    output_testing =  "clarence-dataset/testing-set/abstracts"
    os.makedirs(output_testing, exist_ok=True)
    
    fetch_details(query_clarence_background_training, output_training, file_to_write,'background')
    fetch_details(query_clarence_clinical_training, output_training, file_to_write,'clinical')
    
    
    fetch_details(query_clarence_background_testing, output_testing, file_to_write,'background')
    fetch_details(query_clarence_clinical_testing, output_testing, file_to_write,'clinical')
    fetch_details(query_clarence_laboratory_testing, output_testing, file_to_write,'non-clinical')
    fetch_details(query_clarence_laboratoryvitro_testing, output_testing, file_to_write,'non-clinical')
    fetch_details(query_clarence_laboratoryvivo_testing, output_testing, file_to_write,'non-clinical')
    
    
    
