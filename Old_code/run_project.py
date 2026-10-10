import numpy as np
import os
import Old_code.data_loader as data_loader
import Old_code.topic_model as topic_model
import Old_code.classifier as classifier
import Old_code.generator as generator
from sklearn.preprocessing import LabelEncoder

def run_experiment(overlap=False):
    print(f"\n{'='*30}")
    print(f"Running Experiment: {'Overlapping' if overlap else 'Non-Overlapping'} SBM")
    print(f"{'='*30}")

    # 1. Load Data
    print("Loading Training Data (Bootstrap)...")
    train_docs, train_labels_str = data_loader.load_data('bootstrap')
    print(train_docs)
    print(f"Train Docs: {len(train_docs)}")

    print("Loading Test Data (Original)...")
    test_docs, test_labels_str = data_loader.load_data('original')
    print(f"Test Docs: {len(test_docs)}")

    # Encode labels
    le = LabelEncoder()
    # Fit on all possible labels to be safe, though they should be the same [CW, EFG, MI, LFG]
    all_labels = train_labels_str + test_labels_str
    le.fit(all_labels)
    y_train = le.transform(train_labels_str)
    y_test = le.transform(test_labels_str)
    print(f"Classes: {le.classes_}")

    # 2. Fit SBM on Training Data
    print("Fitting SBM Model...")
    # Using n_min=None to keep all words for now, or maybe n_min=2 to filter noise
    model = topic_model.fit_sbm(train_docs, overlap=overlap, n_min=1)
    
    # Save Model
    model_name = f"sbm_{'overlap' if overlap else 'nonoverlap'}.pkl"
    topic_model.save_model(model, model_name)
    print(f"Model saved to {model_name}")

    # 3. Extract Features
    print("Extracting Features...")
    p_tw_w, p_w_tw, words = topic_model.get_word_topic_probs(model)
    print(f"Topics: {p_tw_w.shape[0]}, Vocabulary: {len(words)}")
    
    word_to_index = {w: i for i, w in enumerate(words)}
    
    X_train = classifier.vectorize_documents(train_docs, p_tw_w, word_to_index)
    X_test = classifier.vectorize_documents(test_docs, p_tw_w, word_to_index)
    
    # 4. Train Classifier
    print("Training Classifier (SVM)...")
    clf, scaler = classifier.train_classifier(X_train, y_train)
    
    # 5. Evaluate
    print("Evaluating on Test Data...")
    acc, report = classifier.evaluate_classifier(clf, scaler, X_test, y_test)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)
    
    return model, p_tw_w, p_w_tw, words, word_to_index

def demo_generator(model, p_tw_w, p_w_tw, words, word_to_index):
    print(f"\n{'='*30}")
    print("Demonstrating Generator")
    print(f"{'='*30}")
    
    # Pick a random doc from training to simulate incomplete sequence
    # Let's verify with the user's example: "('Wet Clay', 'Modelling', 'Pressure')" - but capitalized as in data
    # In data: ('Wet clay', 'Modelling', 'Coiling')
    
    # Let's take the first 3 tokens of a document
    train_docs, _ = data_loader.load_data('bootstrap')
    sample_doc = train_docs[0]
    incomplete_seq = sample_doc[:2] # First 2 steps
    
    print(f"Incomplete Sequence: {incomplete_seq}")
    
    suggestions = generator.generate_continuation(incomplete_seq, p_tw_w, p_w_tw, word_to_index, words, top_k_words=3)
    
    print("Suggested Next Techniques:")
    for i, s in enumerate(suggestions):
        print(f"{i+1}. {s}")

if __name__ == "__main__":
    # Experiment 1: Non-Overlapping
    # Note: sbm non-overlapping might be faster/simpler baseline
    model_no, p_tw_w, p_w_tw, words, w2i = run_experiment(overlap=False)
    
    # Experiment 2: Overlapping
    model_o, p_tw_w_o, p_w_tw_o, words_o, w2i_o = run_experiment(overlap=True)
    
    # Demo with Overlapping model (usually more expressive)
    demo_generator(model_o, p_tw_w_o, p_w_tw_o, words_o, w2i_o)
