chmod +x ./PreprocessAndVectorize.py
chmod +x ./Cnn.py
chmod +x ./Dnn.py
chmod +x ./Lstm.py
chmod +x ./runeval.py

#!/bin/bash

echo "Running PreprocessAndVectorize.py"
python3 PreprocessAndVectorize.py

echo "Running Dnn.py"
echo "Dnn-" > log.txt
python3 Dnn.py >> log.txt

echo "Running runeval.py"
python3 runeval.py
python3 runeval.py Dnn ./Dnn_model.pth >> log.txt

echo "Running Cnn.py"
echo "Cnn-" >> log.txt
python3 Cnn.py >> log.txt

echo "Running runeval.py"
python3 runeval.py
python3 runeval.py Cnn ./Cnn_model.pth >> log.txt

echo "Running Lstm.py"
echo "Lstm-" >> log.txt
python3 Lstm.py >> log.txt

echo "Running runeval.py"
python3 runeval.py
python3 runeval.py Lstm ./Lstm_model.pth >> log.txt

read -p "Press any key to continue..."
