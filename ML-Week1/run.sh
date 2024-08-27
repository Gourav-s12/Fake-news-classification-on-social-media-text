chmod +x ./t1_t3_preprocess.py
chmod +x ./svm.py
chmod +x ./knn.py
chmod +x ./nn.py
chmod +x ./log.py
chmod +x ./fast.py
chmod +x ./evaluation.py

#!/bin/bash

echo "Running t1_t3_preprocess.py"
python3 t1_t3_preprocess.py

echo "Running log.py"
echo "log-" > log.txt
python3 log.py >> log.txt

echo "Running svm.py"
echo "svm-" >> log.txt
python3 svm.py >> log.txt

echo "Running knn.py"
echo "knn-" >> log.txt
python3 knn.py >> log.txt

echo "Running kmeans.py"
echo "kmeans-" >> log.txt
python3 kmeans.py >> log.txt

echo "Running nn.py"
echo "nn-" >> log.txt
python3 nn.py >> log.txt

echo "Running fast.py"
echo "fasttext-" >> log.txt
python3 fast.py >> log.txt

echo "Running evaluation.py"
python3 evaluation.py
python3 evaluation.py >> log.txt

read -p "Press any key to continue..."
