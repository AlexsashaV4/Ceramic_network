import numpy as np

def generate_continuation(incomplete_seq, p_tw_w, p_w_tw, word_to_index, index_to_word, top_k_words=5):
    """
    Generate possible next techniques for an incomplete sequence.
    
    Args:
        incomplete_seq (list of str): The partial document.
        p_tw_w (np.array): P(Topic | Word)
        p_w_tw (np.array): P(Word | Topic)
        word_to_index (dict): Map word to index.
        index_to_word (list): Map index to word.
        top_k_words (int): Number of suggestions to return.
        
    Returns:
        suggestions (list of str): List of suggested next techniques.
    """
    # 1. Infer topic mixture of the current sequence P(t | doc_partial)
    # We approximate this by averaging P(t | w) for w in incomplete_seq
    # This is similar to how we vectorized documents for classification
    
    if not incomplete_seq:
        return []

    word_vectors = []
    for word in incomplete_seq:
        if word in word_to_index:
            idx = word_to_index[word]
            word_vectors.append(p_tw_w[:, idx])
            
    if not word_vectors:
        return []
        
    # P(t | doc) vector
    doc_topic_dist = np.mean(word_vectors, axis=0)
    
    # 2. Predict next word P(w_next | doc)
    # P(w | doc) = sum_t P(w | t) * P(t | doc)
    # p_w_tw is (Words x Topics), doc_topic_dist is (Topics,)
    # Result is (Words,) - probability of each word given the doc topics
    
    # p_w_tw from sbmtm is actually (V x Bw) -> (Words, Topics)? 
    # Let's check topic_model.py comments. 
    # sbmtm.py: p_w_tw is V x B_w (Words by Topics)
    
    # Ensure shapes match
    # doc_topic_dist shape: (n_topics,)
    # p_w_tw shape: (n_words, n_topics) - wait, sbmtm.py says V x B_w.
    
    # Let's double check sbmtm output in topic_model.py
    # "p_w_tw = groups['p_w_tw']"
    # In sbmtm.py: "p_w_tw = n_wb/np.sum(n_wb,axis=0)[np.newaxis,:]" -> shape (V, Bw)
    
    # So we compute dot product:
    # P(w | doc) = p_w_tw @ doc_topic_dist
    
    p_w_doc = np.dot(p_w_tw, doc_topic_dist)
    
    # 3. Rank words
    # We want words that are NOT already in the sequence (assuming no repeats? 
    # Actually techniques might repeat but usually an operatorie chain is a sequence of unique steps 
    # or at least not immediately repeating the same triplet usually.
    # The user manual says "generate of possible techniques for incomplete sequences", 
    # so we just rank high prob words.
    
    top_indices = np.argsort(p_w_doc)[::-1]
    
    suggestions = []
    for idx in top_indices:
        word = index_to_word[idx]
        if word not in incomplete_seq:
            suggestions.append(word)
            if len(suggestions) >= top_k_words:
                break
                
    return suggestions
