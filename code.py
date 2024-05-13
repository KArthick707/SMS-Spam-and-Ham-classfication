from collections import defaultdict
import numpy as np
#from nltk.lm import Vocabulary
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# here data loading is done
def load_dataset(file_path):
    # Load dataset from file
    with open(file_path, 'r') as file:
        dataset = file.readlines()
    return dataset


# splitted into training and testing which is 80% and 20% respectively
def split_dataset(dataset, split_ratio=0.8):
    training_data, testing_data = train_test_split(dataset, test_size=1 - split_ratio, random_state=42)
    return training_data, testing_data


# here data is preprocessed
def preprocess_dataset(dataset):
    preprocessed_data = []
    for line in dataset:
        label, message = line.strip().split('\t')
        preprocessed_message = Tokenization(message)
        preprocessed_data.append((label, preprocessed_message))
    return preprocessed_data


def Tokenization(message):
    # Character Tokenization
    tokens = list(message.lower())
    # Lemmatization is not in full effect
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_message = lemmatized_tokens
    return lemmatized_message


def Spam_and_Ham_Separation(dataset):
    spam_messages = []
    ham_messages = []
    for label, message in dataset:
        if label == 'spam':
            spam_messages.append(message)
        else:
            ham_messages.append(message)
    return spam_messages, ham_messages


def Frequency_Model(messages):
    fourgram_frequencies = defaultdict(int)
    total_fourgrams = 0
    #vocab = Vocabulary()  # Create a new Vocabulary object

    # Count four-gram frequencies
    for message in messages:
        for i in range(len(message) - 3):
            fourgram = (message[i], message[i + 1], message[i + 2], message[i + 3])
            fourgram_frequencies[fourgram] += 1  # And here
            total_fourgrams += 1
            #vocab.update([fourgram])  # Update the vocabulary with the fourgram

    # Convert frequencies to probabilities
    fourgram_probabilities = defaultdict(float)
    for fourgram, frequency in fourgram_frequencies.items():
        #if vocab.lookup(fourgram) != vocab.unk_label:  # Check if the fourgram is in the vocabulary
        fourgram_probabilities[fourgram] = frequency / total_fourgrams

    return fourgram_probabilities


def Log_Probability(message, model):
    log_prob = 0.0
    for i in range(len(message) - 3):
        fourgram = (message[i], message[i + 1], message[i + 2], message[i + 3])  # And here
        if fourgram in model:
            log_prob += np.log(model[fourgram])
        else:
            # Laplace smoothing for unseen fourgrams
            log_prob += np.log(1e-10)  # To avoid log(0)
    return log_prob


def Predict(message, spam_model, ham_model):
    spam_log_prob = Log_Probability(message, spam_model)
    ham_log_prob = Log_Probability(message, ham_model)
    if spam_log_prob <= ham_log_prob:
        return 'ham'
    else:
        return 'spam'


def precision_from_confusion_matrix(Conf_mat):
    num_class = Conf_mat.shape[0]
    precision = np.zeros(num_class)
    for i in range(num_class):
        true_positives = Conf_mat[i, i]
        false_positives = np.sum(Conf_mat[:, i]) - true_positives
        precision[i] = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) != 0 else 0

    return precision


def recall_from_confusion_matrix(Conf_mat):
    num_classes = Conf_mat.shape[0]
    recall = np.zeros(num_classes)
    for i in range(num_classes):
        true_positives = Conf_mat[i, i]
        false_negatives = np.sum(Conf_mat[i, :]) - true_positives
        recall[i] = true_positives / (true_positives + false_negatives) \
            if (true_positives + false_negatives) != 0 else 0
    return recall


def f1_score_from_confusion_matrix(Conf_mat):
    num_classes = Conf_mat.shape[0]
    f1_scores = np.zeros(num_classes)

    for i in range(num_classes):
        true_positives = Conf_mat[i, i]
        false_positives = np.sum(Conf_mat[:, i]) - true_positives
        false_negatives = np.sum(Conf_mat[i, :]) - true_positives
        precision = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) \
            if (true_positives + false_negatives) != 0 else 0
        f1_scores[i] = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) != 0 else 0

    return f1_scores


def macro_average_precision_score(Precision):
    num_class = len(Precision)
    macro_average_P = sum(Precision) / num_class
    return macro_average_P


def recall_macro_average(Recall):
    num_class = len(Recall)
    macro_average_R = sum(Recall) / num_class
    return macro_average_R


# Calculating Macroaverage of the F1_Score
def F1_macro_average(Precision, Recall):
    total_precision = sum(Precision)
    total_recall = sum(Recall)
    macro_precision = total_precision / len(Precision)
    macro_recall = total_recall / len(Recall)
    macro_average_F1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)

    return macro_average_F1


def evaluate_model(testing_data, spam_model, ham_model):
    true_labels = [label for label, _ in testing_data]
    predicted_labels = [Predict(message, spam_model, ham_model) for _, message in testing_data]
    accuracy = accuracy_score(true_labels, predicted_labels)
    confusion_mat = confusion_matrix(true_labels, predicted_labels, labels=['ham', 'spam'])
    return accuracy, confusion_mat


# Load dataset
file_path = r"C:\Users\karthick\OneDrive\Desktop\sms+spam+collection\SMSSpamCollection"
dataset = load_dataset(file_path)

# Preprocess dataset
preprocessed_dataset = preprocess_dataset(dataset)

# Split dataset into training and testing
training_data, testing_data = split_dataset(preprocessed_dataset)

print("Training dataset size:", len(training_data))
print("Testing dataset size:", len(testing_data))
print("\n")

# Separate spam and ham messages from the training dataset
spam_messages, ham_messages = Spam_and_Ham_Separation(training_data)

# Train frequency model for spam messages
spam_model = Frequency_Model(spam_messages)
# print("Spam Model:\n", spam_model)


# Train frequency model for ham messages
ham_model = Frequency_Model(ham_messages)
# print("Ham Model:\n", ham_model)
# Evaluate model on testing data
accuracy, Conf_mat = evaluate_model(testing_data, spam_model, ham_model)

print("Accuracy:\n", accuracy)
print("Confusion Matrix:\n", Conf_mat)

# Calculate Precision, Recall, F1 Score, and Macro Average Precision
Precision = precision_from_confusion_matrix(Conf_mat)
print("Precision:\n", Precision)

Recall = recall_from_confusion_matrix(Conf_mat)
print("Recall:\n", Recall)

F1_Score = f1_score_from_confusion_matrix(Conf_mat)
print("F1 Score:\n", F1_Score)

Macro_Avg_Precision = macro_average_precision_score(Precision)
print("Macro Average Precision:", Macro_Avg_Precision)

Macro_Avg_Recall = recall_macro_average(Recall)
print("Macro Average Recall:", Macro_Avg_Recall)

Macro_Avg_F1_Score = F1_macro_average(Precision, Recall)
print("Macro Average F1 Score:", Macro_Avg_F1_Score)
