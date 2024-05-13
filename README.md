# SMS-Spam-and-Ham-classfication

This Python script is designed to perform spam detection using a character-level n-gram model. Here's a brief summary of its main components:  
Data Loading and Preprocessing: The load_dataset function reads a dataset from a file. The split_dataset function splits the dataset into training and testing sets. The preprocess_dataset function preprocesses the dataset by tokenizing and lemmatizing the messages.  
Model Training: The Spam_and_Ham_Separation function separates the spam and ham messages from the training dataset. The Frequency_Model function trains a frequency model for both spam and ham messages.  
Prediction: The Predict function predicts whether a given message is spam or ham based on the trained models.  
Evaluation: The evaluate_model function evaluates the model on the testing data, calculating the accuracy and confusion matrix. The precision_from_confusion_matrix, recall_from_confusion_matrix, and f1_score_from_confusion_matrix functions calculate the precision, recall, and F1 score from the confusion matrix, respectively. The macro_average_precision_score, recall_macro_average, and F1_macro_average functions calculate the macro-average precision, recall, and F1 score.  
Execution: The script then loads a dataset, preprocesses it, splits it into training and testing sets, trains the frequency models, evaluates the model on the testing data, and prints the evaluation metrics.
