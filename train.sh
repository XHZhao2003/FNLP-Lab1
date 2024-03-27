# !/bin/bash
cd src
for i in $(seq 1 10)
do
    python train.py 0.01 0.001 0.99 $i >> ../result/train_result.txt
done