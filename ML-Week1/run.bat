@echo off

echo Running t1_t3_preprocess.py
python t1_t3_preprocess.py

echo Running log.py
echo "log-" > log.txt
python log.py >> log.txt

echo Running svm.py
echo "svm-" >> log.txt
python svm.py >> log.txt

echo Running knn.py
echo "knn-" >> log.txt
python knn.py >> log.txt

echo Running kmeans.py
echo "kmeans-" >> log.txt
python kmeans.py >> log.txt

echo Running nn.py
echo "nn-" >> log.txt
python nn.py >> log.txt

echo Running fast.py
echo "fasttext-" >> log.txt
python fast.py >> log.txt

echo Running evaluation.py
python evaluation.py
python evaluation.py >> log.txt

pause