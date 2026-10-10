import os
import re
import numpy as np
import glob
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')  # Old_code/data, wherever this is run from
NUM_CULTURES = 4 

def normalize_triplet(t: str) -> str:
    """
    Canonical form: (A, B, C) with single spaces after commas.
    Accepts variants with quotes.
    """
    t = t.strip().strip('"').strip("'")
    # remove internal single quotes
    t = t.replace("',", ",").replace("'", "")
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    # optional: enforce space after commas
    parts = [p.strip() for p in t.strip("()").split(",")]
    if len(parts) == 3:
        return f"({parts[0]}, {parts[1]}, {parts[2]})"
    return t

def load_pottery_data(filename, culture_name):
    documents = []
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found.")
        return []
        
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            triplets = re.findall(r"\('([^']+)', '([^']+)', '([^']+)'\)", line.strip())
            if triplets:
                doc_triplets = [f"({t[0]}, {t[1]}, {t[2]})" for t in triplets]
                documents.append({
                    'vessel_id': f"{culture_name}_{line_num}",
                    'culture': culture_name,
                    'triplets': doc_triplets,
                    'triplets_text': ' '.join(doc_triplets),
                    'num_triplets': len(doc_triplets)
                })
    return documents

def extract_vocabulary(docs):
    """Extract vocabulary and term frequencies from documents"""
    vocab_counter = Counter()
    for doc in docs:
        vocab_counter.update(doc['triplets'])
    vocabulary = list(vocab_counter.keys())
    term_frequencies = dict(vocab_counter)
    return vocabulary, term_frequencies

def create_resampled_documents(docs, group_size=3, num_new_docs=100):
    resampled_docs = []
    culture_groups = defaultdict(list)
    for doc in docs:
        culture_groups[doc['culture']].append(doc)
    
    for culture, culture_docs in culture_groups.items():
        if not culture_docs:
            continue
        for _ in range(num_new_docs):
            group = np.random.choice(culture_docs, size=group_size, replace=True)
            combined_triplets = []
            vessel_ids = []
            for vessel in group:
                combined_triplets.extend(vessel['triplets'])
                vessel_ids.append(vessel['vessel_id'])
            
            if combined_triplets:
                resampled_docs.append({
                    'vessel_id': f"RESAMPLED_{culture}_{len(resampled_docs)+1}",
                    'original_vessels': vessel_ids,
                    'culture': culture,
                    'triplets': combined_triplets,
                    'triplets_text': ' '.join(combined_triplets),
                    'num_triplets': len(combined_triplets),
                    'group_size': len(group)
                })
    return resampled_docs

def load_and_process_data(remove_common_k=5, resample_group_size=5, num_resamples=100):
    """
    Main entry point to load, clean, and potentially resample data.
    
    Returns:
        tuple: (original_docs, resampled_docs)
               Each is a list of dictionaries with 'triplets', 'culture', etc.
    """
    print("Loading archaeological datasets...")
    # Paths based on user snippet + DATA_DIR
    cw_docs = load_pottery_data(os.path.join(DATA_DIR, 'clustered_paths_CW.csv'), 'CW')
    mi_docs = load_pottery_data(os.path.join(DATA_DIR, 'clustered_paths_MI.csv'), 'MI')
    efg_docs = load_pottery_data(os.path.join(DATA_DIR, 'clustered_paths_EFG.csv'), 'EFG')
    lfg_docs = load_pottery_data(os.path.join(DATA_DIR, 'clustered_paths_LFG.csv'), 'LFG')
    fbw_docs = load_pottery_data(os.path.join(DATA_DIR, 'clustered_paths_FBW.csv'), 'FBW')
    
    print(f"Dataset summary: CW={len(cw_docs)}, MI={len(mi_docs)}, EFG={len(efg_docs)}, LFG={len(lfg_docs)}, FBW={len(fbw_docs)}")

    # Combine based on NUM_CULTURES logic from user
    if NUM_CULTURES == 4:
        # User logic: exclude last 2 from each?
        all_docs = cw_docs[:-2] + mi_docs[:-2] + efg_docs[:-2] + lfg_docs[:-2]
    elif NUM_CULTURES == 2:
        all_docs = cw_docs + fbw_docs
    else:
        all_docs = cw_docs + mi_docs + efg_docs + lfg_docs # Default fallback

    print(f"Total documents before filtering: {len(all_docs)}")

    # Preprocessing: Remove most common triplets
    if remove_common_k > 0:
        triplet_counter = Counter()
        for doc in all_docs:
            triplet_counter.update(doc['triplets'])
        
        most_common_triplets = dict(triplet_counter.most_common(remove_common_k))
        print(f"Removing {remove_common_k} most common triplets: {list(most_common_triplets.keys())}")
        
        for doc in all_docs:
            doc['triplets'] = [t for t in doc['triplets'] if t not in most_common_triplets]
    
    # Resampling (Bootstrapping)
    print(f"Creating resampled documents (Group Size={resample_group_size})...")
    resampled_docs = create_resampled_documents(all_docs, group_size=resample_group_size, num_new_docs=num_resamples)
    
    return all_docs, resampled_docs

def load_data(data_type='bootstrap'):
    """
    Wrapper for backward compatibility or simple usage.
    
    Args:
        data_type: 'original' -> returns processed original docs
                   'bootstrap' -> returns resampled docs (group_size=5 default)
    
    Returns:
        documents (list of list of str): Triplets
        labels (list of str): Culture labels
    """
    original, resampled = load_and_process_data(remove_common_k=5, resample_group_size=5)
    
    target_docs = resampled if data_type == 'bootstrap' else original
    
    documents = [d['triplets'] for d in target_docs]
    labels = [d['culture'] for d in target_docs]
    
    # Also return the full dicts if needed? Standard signature returns just these two.
    # To support the advanced plotting that needs 'triplets' and 'culture' keys in dicts, 
    # the calling scripts (make_plots.py) might need to call load_and_process_data directly 
    # or we can attach the dicts to this return? 
    # For now, we stick to the signature `documents, labels`. 
    # But wait, make_plots.py reconstructed dicts from these lists. That works fine.
    
    return documents, labels

if __name__ == "__main__":
    # Test the loader logic similar to user's snippet
    orig, resamp = load_and_process_data()
    
    print("\nSample Original Doc:")
    if orig:
        print(orig[0]['triplets'][:5])
        
    print("\nSample Resampled Doc:")
    if resamp:
        print(resamp[0]['triplets'][:5])
    
    # Vocabulary Check
    vocab, _ = extract_vocabulary(orig)
    print(f"\nVocabulary size (Original): {len(vocab)}")
