# Classifier

This component performs multiclass classification of abstracts to determine if they are relevant (either clinical or non-clinical studies) or not relevant to the field of biomaterials.   

## Description  of the project

Since the goal of DEBBIE is to annotate biomaterial-specific research, a classifier is implemented into the pipeline to identify relevant abstracts. The SGD Classifier has been pre-trained on 3 collections:
- [gold standard biomaterials non-clinical collection](https://github.com/ProjectDebbie/gold_standard_set/blob/master/laboratory_gs_1213.txt)
- [gold standard biomaterials clinical collection](https://github.com/ProjectDebbie/gold_standard_set/blob/master/clinical_gs_1092.txt)
- [background set](https://github.com/ProjectDebbie/background_set). 

The components of the trained classifier are available for download as .pkl files. The classifer returns the relevant abstracts in a designated folder.    

Parameters:

-i input folder with plain text abstracts

-o output folder with relevant biomaterials abstracts in plain text 

## Actual Version: 1.0.1, 2020-06-22
## [Changelog](https://github.com/ProjectDebbie/Classifier/blob/master/CHANGELOG)

## Built With

* [scikit-learn library](https://scikit-learn.org)
* [Docker](https://www.docker.com/) - Docker Containers

## Authors

* **Osnat Hakimi** - **Austin McKitrick** - **Javier Corvi** 


## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE](LICENSE) file for details

## Funding
<img align="left" width="75" height="50" src="eu_emblem.png"> This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No 751277
	
		
