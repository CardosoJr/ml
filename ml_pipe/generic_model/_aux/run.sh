#!/bin/bash
source /home/jupyter/Anilton/RENOV2/env/bin/activate

num=10

for i in $(seq 1 $num)
do
    echo $i 
    timeout 10800 python /home/jupyter/Anilton/RENOV2/0-Repo/market-pricing/Experiments/xgboost/main.py run_hyperopt_search /home/jupyter/Anilton/RENOV2/0-Repo/market-pricing/Experiments/xgboost/code/experimentos/config_hyperopt_202001.yaml &> log.txt
    
    sudo cp log.txt bkp/log_$i.txt
    sudo rm log.txt
	sudo rm -rf /home/jupyter/Anilton/RENOV2/teste/hyperopt_1/data/BacktestReport
	sudo rm -rf /home/jupyter/Anilton/RENOV2/teste/hyperopt_1/data/TaskModelMetrics
	sudo rm -rf /home/jupyter/Anilton/RENOV2/teste/hyperopt_1/data/TaskPredict
	sudo rm -rf /home/jupyter/Anilton/RENOV2/teste/hyperopt_1/data/TaskPrepareDS
	sudo rm -rf /home/jupyter/Anilton/RENOV2/teste/hyperopt_1/data/TaskTrainModel
done