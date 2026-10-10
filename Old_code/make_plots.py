import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import Old_code.data_loader as data_loader
import Old_code.topic_model as topic_model
import Old_code.classifier as classifier
import os
import pandas as pd
from collections import Counter
from Old_code.sbmtm import sbmtm

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix", filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_topic_word_heatmap(p_w_tw, words, top_k=20, filename="topic_word_heatmap.png"):
    """
    Plot heatmap of top words (rows) vs topics (cols).
    """
    p_tw_w = p_w_tw.T # Switch to Topics x Words
    
    # Filter for top words
    max_probs = np.max(p_tw_w, axis=0)
    top_indices = np.argsort(max_probs)[::-1][:top_k]
    
    top_p = p_tw_w[:, top_indices]
    top_words = [words[i] for i in top_indices]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(top_p, annot=False, cmap='viridis', xticklabels=top_words, yticklabels=[f"Topic {i}" for i in range(p_tw_w.shape[0])])
    plt.title(f"Top {top_k} Words vs Topics")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def compute_word_cluster_matrix(p_td_d, docs, vocab):
    # p_td_d is (groups x documents)
    cluster_ids = np.argmax(p_td_d, axis=0)
    n_clusters = p_td_d.shape[0]
    word_index = {w: i for i, w in enumerate(vocab)}
    counts = np.zeros((len(vocab), n_clusters), dtype=float)
    cluster_label_counts = [Counter() for _ in range(n_clusters)]

    for doc_idx, doc in enumerate(docs):
        if doc_idx >= len(cluster_ids): 
            # In case docs length doesn't match p_td_d columns (e.g. if graph filtered empty docs)
            continue
        cluster = cluster_ids[doc_idx]
        cluster_label_counts[cluster][doc.get('culture', 'Unknown')] += 1
        triplet_counts = Counter(doc['triplets'])
        for triplet, freq in triplet_counts.items():
            word_pos = word_index.get(triplet)
            if word_pos is not None:
                counts[word_pos, cluster] += freq

    cluster_names = []
    for i, label_count in enumerate(cluster_label_counts):
        if label_count:
            majority_label, majority_count = label_count.most_common(1)[0]
            total_docs = sum(label_count.values())
            cluster_names.append(f"{majority_label} (C{i}, {majority_count}/{total_docs})")
        else:
            cluster_names.append(f"Cluster {i}")
    return counts, cluster_names

def plot_word_cluster_share(counts, vocab, cluster_names, top_words=50, cmap='magma', x_label="SBM document clusters", save='model'):
    totals = counts.sum(axis=0, keepdims=True) + 1e-12
    percentages = counts / totals
    
    # Select top words based on overall frequency
    word_totals = counts.sum(axis=1)
    indices = np.argsort(word_totals)[::-1]
    
    if top_words is not None:
        indices = indices[:top_words]
    
    heatmap_df = pd.DataFrame(
        percentages[indices],
        index=np.array(vocab)[indices],
        columns=cluster_names
    )
    plt.figure(figsize=(18, 0.35 * len(heatmap_df)))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.4, cbar_kws={'label': 'Word usage within clusters'})
    plt.xlabel(x_label)
    plt.ylabel("Triplets")
    plt.tight_layout()
    filename = f"./plots/Word_usage_within_clusters_{save}.png"
    plt.savefig(filename, format='png', dpi=300)
    plt.close()
    print(f"Saved {filename}")

def plot_word_cluster_across_share(counts, vocab, cluster_names, top_words=50, cmap='viridis', x_label="SBM document clusters", save='model'):
    word_totals = counts.sum(axis=1, keepdims=True) + 1e-12
    percentages = counts / word_totals
    
    # Select top words based on overall frequency
    overall_totals = counts.sum(axis=1)
    indices = np.argsort(overall_totals)[::-1]

    if top_words is not None:
        indices = indices[:top_words]
        
    heatmap_df = pd.DataFrame(
        percentages[indices],
        index=np.array(vocab)[indices],
        columns=cluster_names
    )
    plt.figure(figsize=(18, 0.35 * len(heatmap_df)))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.4, cbar_kws={'label': 'Word usage across groups'})
    plt.xlabel(x_label)
    plt.ylabel("Triplets")
    plt.tight_layout()
    filename = f"./plots/Word_usage_across_clusters_{save}.png"
    plt.savefig(filename, format='png', dpi=300)
    plt.close()
    print(f"Saved {filename}")

def compute_word_culture_matrix(docs, vocab, culture_order=None):
    if culture_order is None:
        culture_order = sorted({doc.get('culture', 'Unknown') for doc in docs})
    word_index = {w: i for i, w in enumerate(vocab)}
    counts = np.zeros((len(vocab), len(culture_order)), dtype=float)
    culture_to_id = {c: idx for idx, c in enumerate(culture_order)}

    for doc in docs:
        culture = doc.get('culture', 'Unknown')
        if culture not in culture_to_id:
            continue
        col = culture_to_id[culture]
        triplet_counts = Counter(doc['triplets'])
        for triplet, freq in triplet_counts.items():
            word_pos = word_index.get(triplet)
            if word_pos is not None:
                counts[word_pos, col] += freq
    return counts, culture_order

def js_divergence(p, q):
    from scipy.spatial.distance import jensenshannon
    return jensenshannon(p, q)**2

def plot_empirical_vs_topic(docs_map, model_p_w_tw, words, culture_names=['CW', 'MI', 'EFG', 'LFG'], save_suffix=""):
    """
    Comparison of empirical word distribution per culture vs best matching SBM topic.
    """
    print("Computing Empirical vs Topic distributions...")
    
    # 1. Compute Empirical Distributions
    empirical_dists = {}
    vocab_index = {w: i for i, w in enumerate(words)}
    V = len(words)
    
    for culture in culture_names:
        c_docs = docs_map.get(culture, [])
        all_triplets = []
        for d in c_docs:
            all_triplets.extend(d['triplets'])
            
        counts = Counter(all_triplets)
        total = sum(counts.values()) or 1
        
        dist = np.zeros(V)
        for t, c in counts.items():
            if t in vocab_index:
                dist[vocab_index[t]] = c / total
        
        empirical_dists[culture] = dist

    # 2. Find best matching topic (min JS divergence)
    # model_p_w_tw shape is (V, Bw) -> P(w|tw). We access columns.
    n_topics = model_p_w_tw.shape[1]
    p_w_tw_T = model_p_w_tw.T # (Bw, V)
    
    if not os.path.exists("./plots/comparison"):
        os.makedirs("./plots/comparison")
        
    for culture in culture_names:
        emp_dist = empirical_dists[culture]
        
        # Calculate JS for all topics
        js_scores = [js_divergence(emp_dist, p_w_tw_T[k]) for k in range(n_topics)]
        best_mic_idx = np.argmin(js_scores)
        best_topic_dist = p_w_tw_T[best_mic_idx]
        
        print(f"Culture {culture} matches best with Topic {best_mic_idx} (JS={js_scores[best_mic_idx]:.4f})")
        
        # 3. Plot
        sort_idx = np.argsort(emp_dist)[::-1]
        top_k = 20
        top_idx = sort_idx[:top_k]
        
        plt.figure(figsize=(12, 10))
        y = np.arange(len(top_idx))
        
        plt.barh(y - 0.2, emp_dist[top_idx], height=0.4, alpha=0.6, label=f'Empirical {culture}')
        plt.barh(y + 0.2, best_topic_dist[top_idx], height=0.4, alpha=0.6, label=f'Topic {best_mic_idx}')
        
        plt.yticks(y, [words[i] for i in top_idx])
        plt.xlabel('Probability')
        plt.title(f'{culture} Empirical vs Topic {best_mic_idx} Distribution')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'./plots/comparison/{culture}_vs_Topic{best_mic_idx}_{save_suffix}.png')
        plt.close()


def generate_plots(model_path="sbm_overlap.pkl.pickle", overlap=True):
    print(f"Generating plots for {model_path}...")
    
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    # 1. Load Data & Model
    train_docs_list, train_labels_str = data_loader.load_data('bootstrap')
    test_docs_list, test_labels_str = data_loader.load_data('original')
    
    # Structure data for plotting functions
    train_docs_struct = [{'triplets': d, 'culture': c} for d, c in zip(train_docs_list, train_labels_str)]
    test_docs_struct = [{'triplets': d, 'culture': c} for d, c in zip(test_docs_list, test_labels_str)]
    all_docs_struct = train_docs_struct + test_docs_struct # Note: model was trained on bootstrap? 
    # Actually, model was trained on bootstrap data usually. Let's assume model corresponds to train_docs.
    # But wait, sbmtm make_graph processes the documents. The indices in p_td_d correspond to the documents in the graph.
    # If the model was trained on 'train_docs' (bootstrap), then p_td_d corresponds to train_docs.
    
    model = topic_model.load_model(model_path)
    
    # We need to know which docs are in the graph.
    # SBM graph nodes are documents + words.
    # In sbmtm.py: self.documents = [ g.vp['name'][v] for v in g.vertices() if g.vp['kind'][v]==0 ]
    # If document titles were not provided, they are indices.
    # If we assume the order is preserved (which it usually is in make_graph), we can use train_docs_struct.
    # However, make_graph takes 'list_texts'. 
    
    # Let's use train_docs_struct for the model visualization since that's what it was likely trained on.
    docs_for_model = train_docs_struct
    
    groups = model.get_groups(l=0)
    p_td_d = groups['p_td_d'] # (B_d, D)
    
    # Check dimensions
    if p_td_d.shape[1] != len(docs_for_model):
        print(f"Warning: Number of documents in model ({p_td_d.shape[1]}) does not match loaded data ({len(docs_for_model)}).")
        # Attempt to use only the first N docs if model is smaller? 
        # Or maybe the model was trained on a subset?
        # Proceeding with truncation if needed for safety, but this might misalign.
        docs_for_model = docs_for_model[:p_td_d.shape[1]]

    # 2. Confusion Matrix (Classifier)
    # Using existing logic for consistency
    le = LabelEncoder()
    all_labels = train_labels_str + test_labels_str
    le.fit(all_labels)
    y_train = le.transform(train_labels_str)
    y_test = le.transform(test_labels_str)
    classes = le.classes_
    
    p_tw_w, p_w_tw, words = topic_model.get_word_topic_probs(model)
    word_to_index = {w: i for i, w in enumerate(words)}
    
    X_train = classifier.vectorize_documents(train_docs_list, p_tw_w, word_to_index)
    X_test = classifier.vectorize_documents(test_docs_list, p_tw_w, word_to_index)
    
    # Ensure X_train matches y_train length
    min_len = min(len(X_train), len(y_train))
    X_train = X_train[:min_len]
    y_train = y_train[:min_len]

    clf, scaler = classifier.train_classifier(X_train, y_train)
    y_pred = clf.predict(scaler.transform(X_test))
    
    plot_name = "./plots/confusion_matrix_overlap.png" if overlap else "./plots/confusion_matrix_nonoverlap.png"
    plot_confusion_matrix(y_test, y_pred, classes, title=f"Confusion Matrix ({'Overlapping' if overlap else 'Non-Overlapping'})", filename=plot_name)
    print(f"Saved {plot_name}")

    # 3. NetTM Style Plots (Cluster Analysis)
    print("Generating NetTM style plots...")
    counts_matrix, cluster_names = compute_word_cluster_matrix(p_td_d, docs_for_model, model.words)
    save_suffix = 'overlap' if overlap else 'nonoverlap'
    
    plot_word_cluster_share(counts_matrix, model.words, cluster_names, top_words=40, save=save_suffix)
    plot_word_cluster_across_share(counts_matrix, model.words, cluster_names, top_words=40, save=save_suffix)

    # 4. Empirical Culture Plots (using all docs)
    # This shows word usage across actual Cultures, independent of SBM clusters
    # We can use all available data (bootstrap + original) or just original? 
    # Let's use original for strict empirical validation, or all for better stats.
    # The snippet used 'all_docs'.
    emp_counts_matrix, culture_labels = compute_word_culture_matrix(all_docs_struct, model.words, culture_order=['CW', 'EFG', 'MI', 'LFG'])
    plot_word_cluster_share(emp_counts_matrix, model.words, culture_labels, top_words=40, cmap='magma', x_label="Cultures", save='empirical')
    plot_word_cluster_across_share(emp_counts_matrix, model.words, culture_labels, top_words=40, cmap='viridis', x_label="Cultures", save='empirical_across')

    # 4b. Empirical vs Topic Comparison (JS Divergence)
    print("Generating Empirical vs Topic comparison plots...")
    # Group original docs by culture for empirical ground truth
    docs_by_culture = {}
    for d, c in zip(test_docs_list, test_labels_str): # Using original test data
        if c not in docs_by_culture:
            docs_by_culture[c] = []
        docs_by_culture[c].append({'triplets': d, 'culture': c})
        
    plot_empirical_vs_topic(docs_by_culture, p_w_tw, model.words, culture_names=['CW', 'MI', 'EFG', 'LFG'], save_suffix=save_suffix)

    # 5. SBMTM Native Plots
    print("Generating SBMTM native plots...")
    test_docs_list, test_labels_str = data_loader.load_data('original')
    
    # Structure data for plotting functions
    train_docs_struct = [{'triplets': d, 'culture': c} for d, c in zip(train_docs_list, train_labels_str)]
    test_docs_struct = [{'triplets': d, 'culture': c} for d, c in zip(test_docs_list, test_labels_str)]
    all_docs_struct = train_docs_struct + test_docs_struct # Note: model was trained on bootstrap? 
    # Actually, model was trained on bootstrap data usually. Let's assume model corresponds to train_docs.
    # But wait, sbmtm make_graph processes the documents. The indices in p_td_d correspond to the documents in the graph.
    # If the model was trained on 'train_docs' (bootstrap), then p_td_d corresponds to train_docs.
    
    model = topic_model.load_model(model_path)
    
    # We need to know which docs are in the graph.
    # SBM graph nodes are documents + words.
    # In sbmtm.py: self.documents = [ g.vp['name'][v] for v in g.vertices() if g.vp['kind'][v]==0 ]
    # If document titles were not provided, they are indices.
    # If we assume the order is preserved (which it usually is in make_graph), we can use train_docs_struct.
    # However, make_graph takes 'list_texts'. 
    
    # Let's use train_docs_struct for the model visualization since that's what it was likely trained on.
    docs_for_model = train_docs_struct
    
    groups = model.get_groups(l=0)
    p_td_d = groups['p_td_d'] # (B_d, D)
    
    # Check dimensions
    if p_td_d.shape[1] != len(docs_for_model):
        print(f"Warning: Number of documents in model ({p_td_d.shape[1]}) does not match loaded data ({len(docs_for_model)}).")
        # Attempt to use only the first N docs if model is smaller? 
        # Or maybe the model was trained on a subset?
        # Proceeding with truncation if needed for safety, but this might misalign.
        docs_for_model = docs_for_model[:p_td_d.shape[1]]

    # 2. Confusion Matrix (Classifier)
    # Using existing logic for consistency
    le = LabelEncoder()
    all_labels = train_labels_str + test_labels_str
    le.fit(all_labels)
    y_train = le.transform(train_labels_str)
    y_test = le.transform(test_labels_str)
    classes = le.classes_
    
    p_tw_w, p_w_tw, words = topic_model.get_word_topic_probs(model)
    word_to_index = {w: i for i, w in enumerate(words)}
    
    X_train = classifier.vectorize_documents(train_docs_list, p_tw_w, word_to_index)
    X_test = classifier.vectorize_documents(test_docs_list, p_tw_w, word_to_index)
    
    # Ensure X_train matches y_train length
    min_len = min(len(X_train), len(y_train))
    X_train = X_train[:min_len]
    y_train = y_train[:min_len]

    clf, scaler = classifier.train_classifier(X_train, y_train)
    y_pred = clf.predict(scaler.transform(X_test))
    
    plot_name = "./plots/confusion_matrix_overlap.png" if overlap else "./plots/confusion_matrix_nonoverlap.png"
    plot_confusion_matrix(y_test, y_pred, classes, title=f"Confusion Matrix ({'Overlapping' if overlap else 'Non-Overlapping'})", filename=plot_name)
    print(f"Saved {plot_name}")

    # 3. NetTM Style Plots (Cluster Analysis)
    print("Generating NetTM style plots...")
    counts_matrix, cluster_names = compute_word_cluster_matrix(p_td_d, docs_for_model, model.words)
    save_suffix = 'overlap' if overlap else 'nonoverlap'
    
    plot_word_cluster_share(counts_matrix, model.words, cluster_names, top_words=40, save=save_suffix)
    plot_word_cluster_across_share(counts_matrix, model.words, cluster_names, top_words=40, save=save_suffix)

    # 4. Empirical Culture Plots (using all docs)
    # This shows word usage across actual Cultures, independent of SBM clusters
    # We can use all available data (bootstrap + original) or just original? 
    # Let's use original for strict empirical validation, or all for better stats.
    # The snippet used 'all_docs'.
    emp_counts_matrix, culture_labels = compute_word_culture_matrix(all_docs_struct, model.words, culture_order=['CW', 'EFG', 'MI', 'LFG'])
    plot_word_cluster_share(emp_counts_matrix, model.words, culture_labels, top_words=40, cmap='magma', x_label="Cultures", save='empirical')
    plot_word_cluster_across_share(emp_counts_matrix, model.words, culture_labels, top_words=40, cmap='viridis', x_label="Cultures", save='empirical_across')

    # 4b. Empirical vs Topic Comparison (JS Divergence)
    print("Generating Empirical vs Topic comparison plots...")
    # Group original docs by culture for empirical ground truth
    docs_by_culture = {}
    for d, c in zip(test_docs_list, test_labels_str): # Using original test data
        if c not in docs_by_culture:
            docs_by_culture[c] = []
        docs_by_culture[c].append({'triplets': d, 'culture': c})
        
    plot_empirical_vs_topic(docs_by_culture, p_w_tw, model.words, culture_names=['CW', 'MI', 'EFG', 'LFG'], save_suffix=save_suffix)

    # 5. SBMTM Native Plots
    print("Generating SBMTM native plots...")
    graph_plot_name = f"./plots/sbmtm_graph_{save_suffix}.png"
    try:
        model.plot(filename=graph_plot_name, nedges=1000)
        print(f"Saved {graph_plot_name}")
    except Exception as e:
        print(f"Failed to plot graph: {e}")

    try:
        # Native plot_topic_dist saves to current dir, we move them
        model.plot_topic_dist(l=0)
        os.rename("p_w_tw_0.png", f"./plots/p_w_tw_{save_suffix}.png")
        os.rename("p_tw_d_0.png", f"./plots/p_tw_d_{save_suffix}.png")
        print(f"Saved native distribution plots for {save_suffix}")
    except Exception as e:
        print(f"Failed to plot topic dist: {e}")

if __name__ == "__main__":
    if os.path.exists("sbm_overlap.pkl.pickle"):
        generate_plots("sbm_overlap.pkl.pickle", overlap=True)
    else:
        print("sbm_overlap.pkl.pickle not found.")
        
    if os.path.exists("sbm_nonoverlap.pkl.pickle"):
        generate_plots("sbm_nonoverlap.pkl.pickle", overlap=False)
    else:
        print("sbm_nonoverlap.pkl.pickle not found.")
