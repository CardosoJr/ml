#!/bin/bash -e

mkdir -p ./modelos
mkdir -p ./modelos/generic_model

cp -r  ../../lib/ ./lib/
cp -r  ../../modelos/generic_model/ ./modelos
cp -r  ../../main.py ./main.py
cp -r  ../../config_geral.yaml ./config_geral.yaml
cp -r  ../../gpu_config.yaml ./gpu_config.yaml


PROJECT_ID=$(gcloud config config-helper --format "value(configuration.properties.core.project)")
CONTAINER_NAME=esteira/generic_model
TAG_NAME='latest'

docker build -t ${CONTAINER_NAME} .
docker tag ${CONTAINER_NAME} gcr.io/${PROJECT_ID}/${CONTAINER_NAME}:${TAG_NAME}
docker push gcr.io/${PROJECT_ID}/${CONTAINER_NAME}:${TAG_NAME}

rm -rf lib
rm -rf modelos
rm main.py 
rm config_geral.yaml
rm gpu_config.yaml