import Old_code.sbmtm as sbmtm
import numpy as np
import os
import graph_tool.all as gt

def fit_sbm(documents, overlap=False, n_min=None):
    """
    Fit an SBM model to the documents.
    
    Args:
        documents (list of list of str): The corpus.
        overlap (bool): Whether to use overlapping SBM.
        n_min (int): Minimum count for words to be included.
        
    Returns:
        model: Trained sbmtm model.
    """
    model = sbmtm.sbmtm()
    
    # sbmtm expects a list of list of strings
    # For overlapping SBM, we must set counts=False
    counts = not overlap
    
    # User specifically requested this setup for overlapping
    model.make_graph(documents, n_min=n_min, counts=counts)
    
    # Fit the model
    # Set seed as requested by user for reproducibility
    gt.seed_rng(32)
    model.fit(overlap=overlap, n_init=1) 
    
    return model

def get_word_topic_probs(model):
    """
    Extract P(Topic | Word) = p_tw_w from the model.
    
    Returns:
        p_tw_w (np.array): Matrix of shape (n_topics, n_words).
        p_w_tw (np.array): Matrix of shape (n_words, n_topics).
        word_list (list): List of words corresponding to the columns of p_tw_w.
    """
    # Get groups at level 0 (lowest level usually)
    groups = model.get_groups(l=0)
    
    p_tw_w = groups['p_tw_w'] # shape (B_w, V) -> (Topics, Words)
    
    # We also need P(w|tw) for the generator
    p_w_tw = groups['p_w_tw']
    
    return p_tw_w, p_w_tw, model.words

def save_model(model, path):
    model.save_model(path)

def load_model(path):
    model = sbmtm.sbmtm()
    model.load_model(path)
    return model
