FROM alpine:3.9
WORKDIR /usr/local/share/debbie_classifier

ARG	DEBBIE_CLASSIFIER_VERSION=1.0
COPY	docker-build.sh /usr/local/bin/docker-build.sh
COPY	src src	
COPY	pom.xml .

RUN chmod u=rwx,g=rwx,o=r /usr/local/share/debbie_classifier -R

RUN	docker-build.sh ${DEBBIE_CLASSIFIER_VERSION}