FROM python:3.7.5-slim
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY constraints.txt .

RUN pip install -r requirements.txt -c constraints.txt

RUN [ "python3", "-c", "import nltk; nltk.download('stopwords', download_dir='/usr/share/nltk_data'); nltk.download('punkt', download_dir='/usr/share/nltk_data'); nltk.download('averaged_perceptron_tagger', download_dir='/usr/share/nltk_data'); nltk.download('wordnet', download_dir='/usr/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/share/nltk_data')" ]

COPY debbie_trained_classifier.py .

CMD [ "python", "debbie_trained_classifier.py" ]
