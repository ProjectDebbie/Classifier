# Classifier

This component performs binary classification of abstracts to determine if they are relevant or not relevant to the field of biomaterials.   

## Description  of the project

Since the goal of DEBBIE is to annotate biomaterial-specific research, a classifier is implemented into the pipeline to identify relevant abstracts. The SGD Classifier has been pre-trained on the [gold standard biomaterials collection](gold_standard_set) against a [background set](background_set). The components of the trained classifier are available for download as .pkl files. The classifer returns the relevant abstracts in a designated folder.    
  
Using manually a constructed test set made of the polydioxanone literature, the trained Classifier achieved 0.9011 Precision, 0.9623 Recall, and 0.9307 F1-score compared to the manual curator. 

Parameters:

-i input folder with plain text abstracts

-o output folder with relevant biomaterials abstracts in plain text 


## Built With

* [scikit-learn library] (https://scikit-learn.org)
* [Docker](https://www.docker.com/) - Docker Containers
* [Maven](https://maven.apache.org/) - Dependency Management

## Authors

* **Osnat Hakimi** - **Austin McKitrick** - **Javier Corvi** 


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE](LICENSE.txt) file for details
	
		
