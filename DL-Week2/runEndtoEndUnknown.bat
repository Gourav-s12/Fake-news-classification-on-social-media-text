@echo off


echo Running evaltestcustom.py for Dnn
echo "Dnn evaltestcustom-" > logUnknown.txt
python evaltestcustom.py Dnn_model_bert-base-uncased.pth ./unknown.csv Dnn bert-base-uncased >> logUnknown.txt
python evaltestcustom.py Dnn_model_bert-base-cased.pth ./unknown.csv Dnn bert-base-cased >> logUnknown.txt
python evaltestcustom.py Dnn_model_covid-twitter-bert.pth ./unknown.csv Dnn covid-twitter-bert >> logUnknown.txt
python evaltestcustom.py Dnn_model_SocBERT-base.pth ./unknown.csv Dnn SocBERT-base >> logUnknown.txt
python evaltestcustom.py Dnn_model_twhin-bert-base.pth ./unknown.csv Dnn twhin-bert-base >> logUnknown.txt

echo Running evaltestcustom.py for Cnn
echo "Cnn evaltestcustom-" >> logUnknown.txt
python evaltestcustom.py Cnn_model_bert-base-uncased.pth ./unknown.csv Cnn bert-base-uncased >> logUnknown.txt
python evaltestcustom.py Cnn_model_bert-base-cased.pth ./unknown.csv Cnn bert-base-cased >> logUnknown.txt
python evaltestcustom.py Cnn_model_covid-twitter-bert.pth ./unknown.csv Cnn covid-twitter-bert >> logUnknown.txt
python evaltestcustom.py Cnn_model_SocBERT-base.pth ./unknown.csv Cnn SocBERT-base >> logUnknown.txt
python evaltestcustom.py Cnn_model_twhin-bert-base.pth ./unknown.csv Cnn twhin-bert-base >> logUnknown.txt

echo Running evaltestcustom.py for AutoModel
echo "AutoModel evaltestcustom-" >> logUnknown.txt
python evaltestcustom.py AutoModelForSequenceClassification_bert-base-uncased.pth ./unknown.csv AutoModel bert-base-uncased >> logUnknown.txt
python evaltestcustom.py AutoModelForSequenceClassification_bert-base-cased.pth ./unknown.csv AutoModel bert-base-cased >> logUnknown.txt
python evaltestcustom.py AutoModelForSequenceClassification_covid-twitter-bert.pth ./unknown.csv AutoModel covid-twitter-bert >> logUnknown.txt
python evaltestcustom.py AutoModelForSequenceClassification_SocBERT-base.pth ./unknown.csv AutoModel SocBERT-base >> logUnknown.txt
python evaltestcustom.py AutoModelForSequenceClassification_twhin-bert-base.pth ./unknown.csv AutoModel twhin-bert-base >> logUnknown.txt

pause