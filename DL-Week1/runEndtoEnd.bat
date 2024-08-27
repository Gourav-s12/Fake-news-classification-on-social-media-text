@echo off

echo Running PreprocessAndVectorize.py
python PreprocessAndVectorize.py

@REM echo Running Dnn.py
@REM echo "Dnn-" > log.txt
@REM python Dnn.py >> log.txt

echo Running runeval.py
echo "runeval-" > log.txt
python runeval.py Dnn ./Dnn_model.pth >> log.txt

@REM echo Running Cnn.py
@REM echo "Cnn-" >> log.txt
@REM python Cnn.py >> log.txt

echo Running runeval.py
echo "runeval-" >> log.txt
python runeval.py Cnn ./Cnn_model.pth >> log.txt

@REM echo Running Lstm.py
@REM echo "Lstm-" >> log.txt
@REM python Lstm.py >> log.txt

echo Running runeval.py
echo "runeval-" >> log.txt
python runeval.py Lstm ./Lstm_model.pth >> log.txt

pause