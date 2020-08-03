#!/bin/bash
# -*- coding: utf-8 -*-

source ../../env/bin/activate

mode=$1

if [ "$mode" = "backtest" ]; then
    python main.py generic_model dev run_backtest /data/config.yaml
fi

if [ "$mode" = "tuning" ]; then
    time_per_run=$2
    num_runs=$3

    for i in $(seq 1 $num_runs)
    do
        echo $i 
        timeout $time_per_run python main.py generic_model dev run_hyperopt_search /data/config.yaml

        rm -rf ./data/BacktestReport
        rm -rf ./data/TaskModelMetrics
        rm -rf ./data/TaskPredict
        rm -rf ./data/TaskPrepareDS
        rm -rf ./data/TaskTrainModel
    done
fi




