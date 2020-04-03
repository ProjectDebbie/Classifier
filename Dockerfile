FROM python:3.7.5-slim
WORKDIR /usr/src/app

RUN pip install sklearn
RUN pip install numpy
RUN pip install pandas 
RUN pip install joblib
RUN pip install argparse

COPY debbie_trained_classifier.py .
COPY	count_vect.pkl .
COPY	svm_model.pkl .
COPY	transformer.pkl .

CMD [ "python", "debbie_trained_classifier.py" ]