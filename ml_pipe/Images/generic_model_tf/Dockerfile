FROM google/cloud-sdk:297.0.1
FROM tensorflow/tensorflow:2.2.0-gpu

RUN apt-get update -y && apt-get install -y -q ca-certificates python3-dev python3-setuptools python3-pip unzip

COPY generic_model ./generic_model
COPY requirements.txt main.py run.sh gpu_config.yaml ./

RUN pip3 install -r requirements.txt

ENTRYPOINT ["bash", "./run.sh"]

