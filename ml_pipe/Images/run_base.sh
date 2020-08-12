mode=$1
config_file=$2

mkdir -p ./run

cp $config_file ./run/config.yaml

if [ "$mode" = "pt" ]; then
    docker run --gpus all -v $PWD"/run":/data generic_model_pt:latest base
fi

if [ "$mode" = "tf" ]; then
    docker run --gpus all -v $PWD"/run":/data generic_model_tf:latest base
fi