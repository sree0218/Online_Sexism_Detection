# Online_Sexism_Detection

Step 1: Installing dependancies
The 'semeval_10.yaml' has listed all the dependancies required by all the models. Run the below command to create a conda environment 

conda create -n semeval_10.yaml

Activate conda environment by

conda activate semeval_10

Step 2: Running Model1-ML
It contains the Logistic regression, Decison Tree, Xgboost, Random Forest models.
The training data is present in the 'data' folder and classification report will be saved in the 'result' folder.
You need to run the 'semeval_10_ML.ipynb' file.

Step 3: Running Model2-MLP
It contains the MLP model developed of TFIDF features. The each Task A, B, C is trained independently using same model architecture. 
The training data is present in the 'data' folder and classification report, loss curve, confusion matrix will be saved in the 'result' folder.
You need to run 'semeval_10_MLP.ipynb' file.

Step 4: Running Model3-BiLSTM
It contains the BiLSTM model. It uses golve embeddings. The embedding file will be downloaded on the runtime. Since the file size in the GB, it might take a while to
run this model for the first time. If you already have the 'glove.6B.300d.txt' file, you may capy it the 'glove' folder to reduce the execution speed.
The training data is present in the 'data' folder and classification report, confusion matrix will be saved in the 'result' folder.
You need to run 'semeval_10_BiLSTM.ipynb' file.

Step 5: Running Model4-HAM
It contains the Hierarchical Attention Model. The training data (csv files) is present inside the 'data' folder. The tsv file are genereted by this
code and used as intermediatary supporting files. 
The 'log' folder contains the running status of the code. All the info and error logs are added in the 'model_log.txt' file.
The classification report, trained model, loss curve will be saved in the 'result' folder.
The 'utils' conatins the data loading and preprocessing part.
You can change the model parameter and they are located in the main() function of the 'train.py' file. You should only change the params present in the
'YOU MAY EDIT THE BELOW PARAM' section.
You need to run the 'train.py' file.
