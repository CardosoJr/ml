# Running Docker Image - Hyperopt tunning

### Bulding

Build Image from main path modelagem/. For example: bash Images/generic_model/dockerbuild.sh

### Running

run using: 
docker run --gpus all generic_model_tf:latest params

The params are (in order):
1) Mode: (backtest, tunning)
2) Time per run (in seconds): only in tuning mode
3) Number of runs: only in tuning mode