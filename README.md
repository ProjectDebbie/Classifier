# DEBBIE Abstract Classifier

This component performs multiclass classification of abstracts to determine if they are relevant (either clinical or non-clinical studies) or not relevant to the field of biomaterials.   

## Description  of the project

A logistic regression model with stochastic gradient descent optimization (SGDClassifier) using term frequency times inverse document frequency (TF-IDF) as the main feature extraction mechanism was developed and trained to detect abstracts that have relevance to the biomaterials area (either clinical or non-clinical studies). Preprocessing and feature extraction includes: lowercase, removing punctuation, stop words, lemmatization and part-of-speech tagging. 

The DEBBIE SGDClassifier model achieved an presicion of 0.90, a recall of 0.88 and an F1-score of 0.89.

The components of the trained classifier are available for download as .pkl files. The classifer returns the relevant abstracts in a designated folder. 

## Model development and performance

The DEBBIE SGDClassifier has been pre-trained on 3 collections:

- [gold standard biomaterials non-clinical collection](https://github.com/ProjectDebbie/gold_standard_set/blob/master/laboratory_gs_1213.txt)
- [gold standard biomaterials clinical collection](https://github.com/ProjectDebbie/gold_standard_set/blob/master/clinical_gs_1092.txt)
- [background set](https://github.com/ProjectDebbie/background_set). 

The folder classifier-dataset contains the test set that was used for validate the classifier.

The best model was selected after a benchmarking on different vectorization approaches and models, including: Bag of Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF) and Word2Vec Vectorization; and the models Multinomial Naive Bayes, Logistic Regression, SVM and Linear classifiers (SVM and Logistic Regression) with stochastic gradient descent (SGD) training, also know as SGDClassifier. For the selection of the best model a 10 cross-validation using the training set was performed. 

The best model is a logistic regression model with stochastic gradient descent optimization (SGDClassifier) using TF-IDF.
The DEBBIE SGDClassifier model achieved an accuracy of 0.89, a recall of 0.86 and an F1-score of 0.87. In a more detailed analysis on recall, of the 609 biomaterials-relevant abstracts that were part of the test set (248 clinical and 361 non-clinical), 31 (5%) were miss-classified as not relevant. The DEBBIE SGDClassifier was developed using the Python library sklearn.  

### Classification report

![Classification Report](classification_report.png)

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

The performance results can be reproducible through the file debbie_classifier_performance.ypinb (jypiter notebook).
 
## Docker
projectdebbie/classifier:3.0.0

## Execution 

	python3 debbie_trained_classifier.py -i /in -o /out 
	#or in with docker
	docker run --rm -u $UID -v ${PWD}/input_output:/in:ro -v ${PWD}/nlp_preprocessing_output:/out:rw projectdebbie/classifier:version python3 /usr/src/app/debbie_trained_classifier.py -i /in -o /out -w /usr/src/app

Parameters:

-i input folder with plain text abstracts

-o output folder with relevant biomaterials abstracts in plain text 

-w work folder

## Actual Version: 3.0.0, 2022-11-11
## [Changelog](https://github.com/ProjectDebbie/Classifier/blob/master/CHANGELOG)

## Built With

* [scikit-learn library](https://scikit-learn.org)
* [Docker](https://www.docker.com/) - Docker Containers

## Authors

* **Javier Corvi**  - **Osnat Hakimi** - **Austin McKitrick** 


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE](LICENSE) file for details

## Funding
<img align="left" width="75" height="50" src="eu_emblem.png"> This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No 751277
	
		
