import pandas as pd
import numpy as np
import graph_tool.all as gt
from collections import Counter, defaultdict
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jensenshannon
from Old_code.sbmtm import sbmtm 

# Helper Functions
def normalize_triplet(t: str) -> str:
    """Canonical form: (A, B, C) with single spaces after commas."""
    t = t.strip().strip('"').strip("'")
    t = t.replace("',", ",").replace("'", "")
    t = re.sub(r"\s+", " ", t)
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

def create_resampled_documents(docs, group_size=3, num_new_docs=100):
    resampled_docs = []
    culture_groups = defaultdict(list)
    for doc in docs:
        culture_groups[doc['culture']].append(doc)
    
    for culture, culture_docs in culture_groups.items():
        if not culture_docs: continue
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

def js_divergence(p, q):
    return jensenshannon(p, q)**2

# Plots
def plot_analysis(classifier, test_data_dict):
    """
    Generate plots comparing model topics to empirical culture distributions.
    """
    if not os.path.exists("./plots_test"):
        os.makedirs("./plots_test")
        
    model = classifier.model
    groups = model.get_groups(l=0)
    p_w_tw = groups['p_w_tw'] # (Word, Topic)
    words = model.words
    vocab_index = {w: i for i, w in enumerate(words)}
    
    # 1. Topic Distributions Heatmap
    plt.figure(figsize=(12, 10))
    # Transposing to have Topics on X axis if p_w_tw is (Words, Topics) but usually it's (Words, Topics) in sbmtm?
    # Actually sbmtm.get_groups returns p_w_tw as (V, Bw) -> (Words, Topics)
    plt.imshow(p_w_tw, origin='lower', aspect='auto', interpolation='none', cmap='viridis')
    plt.title(r'Word-Topic membership $P(w | tw)$')
    plt.xlabel('Topic')
    plt.ylabel('Word index')
    plt.colorbar()
    plt.savefig('./plots_test/p_w_tw.png')
    plt.close()

    # 2. Empirical vs Topic Comparison
    print("\nGenerating Empirical vs Topic Comparison Plots...")
    n_topics = p_w_tw.shape[1]
    
    for culture, docs in test_data_dict.items():
        # Empirical Distribution
        all_triplets = []
        for seq in docs:
            all_triplets.extend(seq)
        counts = Counter(all_triplets)
        total = sum(counts.values()) or 1
        
        emp_dist = np.zeros(len(words))
        for t, c in counts.items():
            if t in vocab_index:
                emp_dist[vocab_index[t]] = c / total
        
        # Find best matching topic
        # p_w_tw is (V, Bw), so we iterate columns
        js_scores = [js_divergence(emp_dist, p_w_tw[:, k]) for k in range(n_topics)]
        best_topic_idx = np.argmin(js_scores)
        best_topic_dist = p_w_tw[:, best_topic_idx]
        
        print(f"Culture {culture} best matches Topic {best_topic_idx} (JS={js_scores[best_topic_idx]:.4f})")
        
        # Plot Top Words overlap
        sort_idx = np.argsort(emp_dist)[::-1][:20] # Top 20 empirical
        
        plt.figure(figsize=(12, 8))
        y = np.arange(len(sort_idx))
        plt.barh(y - 0.2, emp_dist[sort_idx], height=0.4, alpha=0.7, label=f'Empirical {culture}')
        plt.barh(y + 0.2, best_topic_dist[sort_idx], height=0.4, alpha=0.7, label=f'Topic {best_topic_idx}')
        plt.yticks(y, [words[i] for i in sort_idx])
        plt.xlabel('Probability')
        plt.title(f'{culture} Empirical vs Topic {best_topic_idx} (Model)')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'./plots_test/{culture}_vs_Topic{best_topic_idx}.png')
        plt.close()

class VesselClassifier:
    """
    Topic-model-based classifier for ancient vessels using manufacturing techniques
    """
    def __init__(self):
        self.model = None
        self.culture_mapping = {}
        self.technique_vocab = set()
        
    def prepare_training_data(self, bootstrapped_data_dict):
        """Prepare training data from bootstrapped samples"""
        list_texts = []
        doc_labels = []
        doc_cultures = []
        
        for culture_key, sequences in bootstrapped_data_dict.items():
            # Extract culture name (e.g., 'CW' from 'RESAMPLED_CW')
            culture_name = culture_key.replace('RESAMPLED_', '')
            
            for idx, sequence in enumerate(sequences):
                list_texts.append(sequence)
                doc_labels.append(f"{culture_name}_{idx}")
                doc_cultures.append(culture_name)
                self.technique_vocab.update(sequence)
        
        return list_texts, doc_labels, doc_cultures
    
    def train(self, list_texts, doc_labels, overlap=False, n_init=5):
        self.model = sbmtm()
        self.model.make_graph(
            list_texts, 
            documents=doc_labels,
            counts=(not overlap),
            n_min=2
        )
        gt.seed_rng(32)
        self.model.fit(overlap=overlap, n_init=n_init, verbose=True)
        groups = self.model.get_groups(l=0)
        print(f"Inferred {groups['Bd']} document-groups and {groups['Bw']} word-groups")
        return groups
    
    def map_topics_to_cultures(self, doc_cultures):
        groups = self.model.get_groups(l=0)
        p_td_d = groups['p_td_d']  # (Bd, D)
        Bd = groups['Bd']
        
        culture_mapping = {}
        for topic_id in range(Bd):
            memberships = p_td_d[topic_id, :]
            culture_votes = defaultdict(float)
            for doc_idx, membership in enumerate(memberships):
                if membership > 0.01:
                    culture = doc_cultures[doc_idx]
                    culture_votes[culture] += membership
            
            if culture_votes:
                dominant_culture = max(culture_votes.items(), key=lambda x: x[1])[0]
                culture_mapping[topic_id] = dominant_culture
                print(f"Topic {topic_id} → {dominant_culture} (votes: {dict(culture_votes)})")
            else:
                culture_mapping[topic_id] = "UNKNOWN"
        
        self.culture_mapping = culture_mapping
        return culture_mapping

    def _predict_single(self, sequence):
        groups = self.model.get_groups(l=0)
        p_w_tw = groups['p_w_tw'] 
        Bw = groups['Bw']
        
        word_indices = []
        for triplet in sequence:
            try:
                word_idx = self.model.words.index(triplet)
                word_indices.append(word_idx)
            except ValueError:
                continue
        
        if not word_indices:
            return "UNKNOWN", {c: 1.0/len(set(self.culture_mapping.values())) for c in set(self.culture_mapping.values())}
        
        topic_scores = np.zeros(Bw)
        for word_idx in word_indices:
            log_probs = np.log(p_w_tw[word_idx, :] + 1e-10)
            topic_scores += log_probs
        
        topic_probs = np.exp(topic_scores - np.max(topic_scores))
        topic_probs /= topic_probs.sum()
        
        culture_probs = defaultdict(float)
        for topic_id, prob in enumerate(topic_probs):
            culture = self.culture_mapping.get(topic_id, "UNKNOWN")
            culture_probs[culture] += prob
        
        pred_culture = max(culture_probs.items(), key=lambda x: x[1])[0]
        return pred_culture, dict(culture_probs)

    def predict(self, test_sequences, test_labels=None):
        if test_labels is None:
            test_labels = [f"test_{i}" for i in range(len(test_sequences))]
        
        predictions = []
        probabilities = {}
        
        for idx, (sequence, label) in enumerate(zip(test_sequences, test_labels)):
            pred_culture, pred_probs = self._predict_single(sequence)
            predictions.append(pred_culture)
            probabilities[label] = pred_probs
        
        return predictions, probabilities

    def evaluate(self, test_sequences, true_cultures, test_labels=None):
        predictions, probabilities = self.predict(test_sequences, test_labels)
        accuracy = accuracy_score(true_cultures, predictions)
        report = classification_report(true_cultures, predictions)
        cm = confusion_matrix(true_cultures, predictions)
        
        print("="*60)
        print("CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot Confusion Matrix
        unique_labels = sorted(list(set(true_cultures)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('./plots_test/confusion_matrix.png')
        plt.close()
        
        return {
            'accuracy': accuracy,
            'predictions': predictions
        }
    
    def generate_technique_sequence(self, culture, max_length=10):
        culture_topics = [tid for tid, c in self.culture_mapping.items() if c == culture]
        if not culture_topics:
            return []
        
        groups = self.model.get_groups(l=0)
        p_w_tw = groups['p_w_tw'] 
        topic_id = culture_topics[0] # Pick first associated topic
        word_probs = p_w_tw[:, topic_id]
        word_probs /= word_probs.sum()
        
        sequence = []
        for _ in range(max_length):
            word_idx = np.random.choice(len(self.model.words), p=word_probs)
            sequence.append(self.model.words[word_idx])
        return sequence

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


def main_pipeline():
    # 1. Load Data
    print("Loading datasets...")
    data_dir = './data'
    cw_docs = load_pottery_data(f'{data_dir}/clustered_paths_CW.csv', 'CW')
    mi_docs = load_pottery_data(f'{data_dir}/clustered_paths_MI.csv', 'MI')
    efg_docs = load_pottery_data(f'{data_dir}/clustered_paths_EFG.csv', 'EFG')
    lfg_docs = load_pottery_data(f'{data_dir}/clustered_paths_LFG.csv', 'LFG')
    fbw_docs = load_pottery_data(f'{data_dir}/clustered_paths_FBW.csv', 'FBW')
    
    # Combine (Scenario: 4 cultures exclusion)
    # Using specific exclusion logic mentioned in history/user's code
    all_docs = cw_docs[:-2] + mi_docs[:-2] + efg_docs[:-2] + lfg_docs[:-2]
    
    # Preprocessing: Remove top 5 common triplets
    print("Removing most common triplets...")
    triplet_counter = Counter()
    for doc in all_docs:
        triplet_counter.update(doc['triplets'])
    most_common = dict(triplet_counter.most_common(5))
    
    for doc in all_docs:
        doc['triplets'] = [t for t in doc['triplets'] if t not in most_common]
        
    # 2. Bootstrapping (Resample 15)
    print("Bootstrapping (Group Size 15)...")
    resampled_15 = create_resampled_documents(all_docs, group_size=15, num_new_docs=30)
    
    # Prepare Bootstrapped Data Dict for Classifier
    bootstrapped_data = defaultdict(list)
    for doc in resampled_15:
        key = f"RESAMPLED_{doc['culture']}"
        bootstrapped_data[key].append(doc['triplets'])
    
    # Prepare Test Data Dict (Original clustered paths)
    test_data = {
        'CW': [d['triplets'] for d in cw_docs],
        'MI': [d['triplets'] for d in mi_docs],
        'EFG': [d['triplets'] for d in efg_docs],
        'LFG': [d['triplets'] for d in lfg_docs]
    }

    # 3. Classifier Pipeline
    print("Initializing Classifier...")
    classifier = VesselClassifier()
    
    # Train
    print("Preparing training data...")
    train_texts, train_labels, train_cultures = classifier.prepare_training_data(bootstrapped_data)
    
    print(f"Training on {len(train_texts)} documents (Resampled 15)...")
    classifier.train(train_texts, train_labels, overlap=True, n_init=2) # n_init small for speed in test
    
    classifier.map_topics_to_cultures(train_cultures)
    
    # Evaluate
    print("Evaluating on Original Test Data...")
    test_seqs_flat = []
    test_cultures_flat = []
    test_ids_flat = []
    
    for cult, seqs in test_data.items():
        for i, seq in enumerate(seqs):
            test_seqs_flat.append(seq)
            test_cultures_flat.append(cult)
            test_ids_flat.append(f"{cult}_test_{i}")
            
    classifier.evaluate(test_seqs_flat, test_cultures_flat, test_ids_flat)
    
    # 4. Analysis Plots
    plot_analysis(classifier, test_data)
    
    # 5. Generator Demo
    print("\nGenerator Demo:")
    for culture in ['CW', 'MI', 'EFG', 'LFG']:
        seq = classifier.generate_technique_sequence(culture, max_length=5)
        print(f"{culture}: {seq}")

    return classifier

if __name__ == "__main__":
    main_pipeline()
