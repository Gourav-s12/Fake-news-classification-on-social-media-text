@echo off

echo Running Preprocess.py
python Preprocess.py

echo Running Vectorize.py
echo "Vectorize-" > logg.txt
python Vectorize.py bert-base-uncased   
python Vectorize.py bert-base-cased   
python Vectorize.py covid-twitter-bert   
python Vectorize.py twhin-bert-base   
python Vectorize.py SocBERT-base 

echo Running Dnn.py
echo "Dnn-" >> logg.txt
python Dnn.py bert_vectors_train_bert-base-uncased.pkl bert_vectors_val_bert-base-uncased.pkl bert-base-uncased
python dnn.py bert_vectors_train_bert-base-cased.pkl bert_vectors_val_bert-base-cased.pkl bert-base-cased
python dnn.py bert_vectors_train_covid-twitter-bert.pkl bert_vectors_val_covid-twitter-bert.pkl covid-twitter-bert
python dnn.py bert_vectors_train_SocBERT-base.pkl bert_vectors_val_SocBERT-base.pkl SocBERT-base
python dnn.py bert_vectors_train_twhin-bert-base.pkl bert_vectors_val_twhin-bert-base.pkl twhin-bert-base

echo Running runeval.py for Dnn
echo "Dnn runeval-" >> logg.txt
python runeval.py bert_vectors_test_bert-base-uncased.pkl Dnn_model_bert-base-uncased.pth Dnn bert-base-uncased >> logg.txt
python runeval.py bert_vectors_test_bert-base-cased.pkl Dnn_model_bert-base-cased.pth Dnn bert-base-cased >> logg.txt
python runeval.py bert_vectors_test_covid-twitter-bert.pkl Dnn_model_covid-twitter-bert.pth Dnn covid-twitter-bert >> logg.txt
python runeval.py bert_vectors_test_SocBERT-base.pkl Dnn_model_SocBERT-base.pth Dnn SocBERT-base >> logg.txt
python runeval.py bert_vectors_test_twhin-bert-base.pkl Dnn_model_twhin-bert-base.pth Dnn twhin-bert-base >> logg.txt

echo Running Cnn.py
echo "Cnn-" >> logg.txt
python cnn.py bert_vectors_train_bert-base-uncased.pkl bert_vectors_val_bert-base-uncased.pkl bert-base-uncased
python cnn.py bert_vectors_train_bert-base-cased.pkl bert_vectors_val_bert-base-cased.pkl bert-base-cased
python cnn.py bert_vectors_train_covid-twitter-bert.pkl bert_vectors_val_covid-twitter-bert.pkl covid-twitter-bert
python cnn.py bert_vectors_train_SocBERT-base.pkl bert_vectors_val_SocBERT-base.pkl SocBERT-base
python cnn.py bert_vectors_train_twhin-bert-base.pkl bert_vectors_val_twhin-bert-base.pkl twhin-bert-base

echo Running runeval.py for Cnn
echo "Cnn runeval-" >> logg.txt
python runeval.py bert_vectors_test_bert-base-uncased.pkl Cnn_model_bert-base-uncased.pth Cnn bert-base-uncased >> logg.txt
python runeval.py bert_vectors_test_bert-base-cased.pkl Cnn_model_bert-base-cased.pth Cnn bert-base-cased >> logg.txt
python runeval.py bert_vectors_test_covid-twitter-bert.pkl Cnn_model_covid-twitter-bert.pth Cnn covid-twitter-bert >> logg.txt
python runeval.py bert_vectors_test_SocBERT-base.pkl Cnn_model_SocBERT-base.pth Cnn SocBERT-base >> logg.txt
python runeval.py bert_vectors_test_twhin-bert-base.pkl Cnn_model_twhin-bert-base.pth Cnn twhin-bert-base >> logg.txt

echo Running AutoModel.py
echo "AutoModel-" >> logg.txt
python AutoModel.py train_split.csv val_split.csv bert-base-uncased
python AutoModel.py train_split.csv val_split.csv bert-base-cased
python AutoModel.py train_split.csv val_split.csv covid-twitter-bert
python AutoModel.py train_split.csv val_split.csv SocBERT-base
python AutoModel.py train_split.csv val_split.csv twhin-bert-base

echo Running runeval.py for AutoModel
echo "AutoModel runeval-" >> logg.txt
python runeval.py test_split.csv AutoModelForSequenceClassification_bert-base-uncased.pth AutoModel bert-base-uncased >> logg.txt
python runeval.py test_split.csv AutoModelForSequenceClassification_bert-base-cased.pth AutoModel bert-base-cased >> logg.txt
python runeval.py test_split.csv AutoModelForSequenceClassification_covid-twitter-bert.pth AutoModel covid-twitter-bert >> logg.txt
python runeval.py test_split.csv AutoModelForSequenceClassification_SocBERT-base.pth AutoModel SocBERT-base >> logg.txt
python runeval.py test_split.csv AutoModelForSequenceClassification_twhin-bert-base.pth AutoModel twhin-bert-base >> logg.txt

pause