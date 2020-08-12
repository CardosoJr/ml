#!/bin/bash -e

cp -r  ../../generic_model/ ./
cp -r  ../../main.py ./main.py
cp -r  ../../gpu_config.yaml ./gpu_config.yaml


PROJECT_ID=$(gcloud config config-helper --format "value(configuration.properties.core.project)")
CONTAINER_NAME=generic_model_tf
TAG_NAME='latest'

docker build -t ${CONTAINER_NAME} .
docker tag ${CONTAINER_NAME} gcr.io/${PROJECT_ID}/${CONTAINER_NAME}:${TAG_NAME}
# docker push gcr.io/${PROJECT_ID}/${CONTAINER_NAME}:${TAG_NAME}

rm -rf generic_model
rm main.py 
rm gpu_config.yaml