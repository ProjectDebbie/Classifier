# DEBBIE Classifier

This component performs multiclass classification of abstracts to determine if they are relevant (either clinical or non-clinical studies) or not relevant to the field of biomaterials.   

## Description  of the project

A logistic regression model with stochastic gradient descent optimization (SGDClassifier) using term frequency times inverse document frequency (TF-IDF) as the main feature extraction mechanism was developed and trained to detect abstracts that have relevance to the biomaterials area (either clinical or non-clinical studies).  

The SGD Classifier has been pre-trained on 3 collections:
- [gold standard biomaterials non-clinical collection](https://github.com/ProjectDebbie/gold_standard_set/blob/master/laboratory_gs_1213.txt)
- [gold standard biomaterials clinical collection](https://github.com/ProjectDebbie/gold_standard_set/blob/master/clinical_gs_1092.txt)
- [background set](https://github.com/ProjectDebbie/background_set). 

The folder classifier-dataset contains the actual training and test set that was used for the development of the classifier.

The DEBBIE SGDClassifier model achieved an accuracy of 0.89, a recall of 0.86 and an F1-score of 0.87. In a more detailed analysis on recall, of the 609 biomaterials-relevant abstracts that were part of the test set (248 clinical and 361 non-clinical), 31 (5%) were miss-classified as not relevant. The SGDClassifier was developed using the Python library sklearn.  The performance results can be reproducible through the file multiclass_classifier_development.ypinb (jypiter notebook).

The components of the trained classifier are available for download as .pkl files. The classifer returns the relevant abstracts in a designated folder.    

## Docker
projectdebbie/classifier:2.1.0

## Execution 

	python3 debbie_trained_classifier.py -i /in -o /out 
	#or in with docker
	docker run --rm -u $UID -v ${PWD}/input_output:/in:ro -v ${PWD}/nlp_preprocessing_output:/out:rw projectdebbie/classifier:version python3 /usr/src/app/debbie_trained_classifier.py -i /in -o /out -w /usr/src/app

Parameters:

-i input folder with plain text abstracts

-o output folder with relevant biomaterials abstracts in plain text 

-w work folder

## Actual Version: 2.1.0, 2022-09-05
## [Changelog](https://github.com/ProjectDebbie/Classifier/blob/master/CHANGELOG)

## Built With

* [scikit-learn library](https://scikit-learn.org)
* [Docker](https://www.docker.com/) - Docker Containers

## Authors

* **Osnat Hakimi** - **Javier Corvi**  - **Austin McKitrick** 


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE](LICENSE) file for details

## Funding
<img align="left" width="75" height="50" src="eu_emblem.png"> This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No 751277
	
		
