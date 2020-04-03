FROM python:3.7.5-slim
ADD debbie_trained_classifier.py /
ADD . /input
ADD . /output

RUN pip install sklearn
RUN pip install numpy
RUN pip install pandas 
RUN pip install joblib
RUN pip install argparse

COPY	count_vect.pkl count_vect.pkl
COPY	svm_model.pkl svm_model.pkl
COPY	transformer.pkl transformer.pkl

CMD [ "python", "./debbie_trained_classifier.py" ]