FROM python:3.7.5-slim
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY constraints.txt .

RUN pip install -r requirements.txt -c constraints.txt

COPY debbie_trained_classifier.py .
COPY debbie-abstract-classifier-model-transformers-1.0.0 debbie-abstract-classifier-model-transformers-1.0.0

CMD [ "python", "debbie_trained_classifier.py" ]
