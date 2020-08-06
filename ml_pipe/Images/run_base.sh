config_file=$1

mkdir -p ./run

cp $config_file ./run/config.yaml

docker run --gpus all -v $PWD"/run":/data gcr.io/sas-auto-marketprice-analytics/esteira/generic_model:latest base