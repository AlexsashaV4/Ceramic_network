import numpy as np
import Old_code.data_loader as data_loader
import Old_code.topic_model as topic_model
import Old_code.classifier as classifier
from sklearn.preprocessing import LabelEncoder
import sys

def evaluate_saved_model(model_path, overlap=False):
    print(f"Evaluating Saved Model: {model_path}")
    
    # Load Data
    train_docs, train_labels_str = data_loader.load_data('bootstrap')
    test_docs, test_labels_str = data_loader.load_data('original')
    
    le = LabelEncoder()
    all_labels = train_labels_str + test_labels_str
    le.fit(all_labels)
    y_train = le.transform(train_labels_str)
    y_test = le.transform(test_labels_str)
    
    # Load Model
    model = topic_model.load_model(model_path)
    
    # Extract Features
    p_tw_w, p_w_tw, words = topic_model.get_word_topic_probs(model)
    word_to_index = {w: i for i, w in enumerate(words)}
    
    X_train = classifier.vectorize_documents(train_docs, p_tw_w, word_to_index)
    X_test = classifier.vectorize_documents(test_docs, p_tw_w, word_to_index)
    
    # Train Classifier
    clf, scaler = classifier.train_classifier(X_train, y_train)
    
    # Evaluate
    acc, report = classifier.evaluate_classifier(clf, scaler, X_test, y_test)
    
    result_str = f"Model: {'Overlapping' if overlap else 'Non-Overlapping'}\n"
    result_str += f"Accuracy: {acc:.4f}\n"
    result_str += "Classification Report:\n"
    result_str += report + "\n"
    
    return result_str

if __name__ == "__main__":
    output = ""
    try:
        output += evaluate_saved_model("sbm_nonoverlap.pkl", overlap=False)
    except Exception as e:
        output += f"Non-Overlapping Eval Failed: {e}\n"
        
    try:
        output += evaluate_saved_model("sbm_overlap.pkl", overlap=True)
    except Exception as e:
        output += f"Overlapping Eval Failed: {e}\n"
        
    with open("results.txt", "w") as f:
        f.write(output)
    
    print("Evaluation Complete. Saved to results.txt")
