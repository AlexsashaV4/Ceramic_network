import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def vectorize_documents(documents, p_tw_w, word_to_index):
    """
    Convert documents to vectors by averaging the topic probabilities of their words.
    
    Args:
        documents (list of list of str): List of docs.
        p_tw_w (np.array): P(Topic | Word) matrix (Topics x Words).
        word_to_index (dict): Map from word string to index in p_tw_w.
        
    Returns:
        X (np.array): Document vectors (Docs x Topics).
    """
    n_docs = len(documents)
    n_topics = p_tw_w.shape[0]
    
    X = np.zeros((n_docs, n_topics))
    
    for i, doc in enumerate(documents):
        # Gather vectors for words in this doc
        word_vectors = []
        for word in doc:
            if word in word_to_index:
                idx = word_to_index[word]
                word_vectors.append(p_tw_w[:, idx])
        
        if word_vectors:
            # Average them
            X[i] = np.mean(word_vectors, axis=0)
        else:
            # Handle empty/unknown doc - leave as zeros or uniform? 
            # Zeros is safer for sparse interactions, but uniform might be better probability-wise.
            # Let's keep zeros for now.
            pass
            
    return X

def train_classifier(X_train, y_train):
    """
    Train a classifier. SBM features are probability distributions, 
    so maybe standard scaling helps? SVM is generally robust.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train_scaled, y_train)
    
    return clf, scaler

def evaluate_classifier(clf, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report
