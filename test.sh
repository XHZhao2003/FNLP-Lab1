# !/bin/bash
cd src
for i in $(seq 1 10)
do
    python test.py $i >> ../result/test_result.txt
done