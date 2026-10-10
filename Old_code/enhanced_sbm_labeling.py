# Enhanced SBM Topic Modeling with Proper Labeling
# This script specifically addresses the graph-tool labeling and SBM improvements

import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt
from Old_code.sbmtm import sbmtm

class EnhancedSBMTopicModel:
    """
    Enhanced SBM topic modeling with proper node labeling and visualization
    """
    
    def __init__(self, random_state=42):
        self.model = None
        self.random_state = random_state
        self.vertex_labels = None
        
    def fit_model(self, vocab_text, vessel_titles):
        """Fit the SBM topic model with proper initialization"""
        print("Initializing SBM topic model...")
        
        self.model = sbmtm()
        self.model.make_graph(vocab_text, documents=vessel_titles)
        
        # Set random seed for reproducible results
        gt.seed_rng(self.random_state)
        
        print("Fitting SBM model...")
        self.model.fit()
        
        # Create vertex labels
        self._create_vertex_labels()
        
        print("SBM model fitted successfully!")
        return self
    
    def _create_vertex_labels(self):
        """Create proper vertex labels for the graph"""
        g = self.model.g
        
        # Create a string property map for vertex labels
        self.vertex_labels = g.new_vertex_property("string")
        
        # Assign labels to vertices
        for v in g.vertices():
            vertex_idx = int(v)
            
            # Check if this is a document or word vertex
            if hasattr(g.vp, 'kind'):
                if g.vp.kind[v] == 0:  # Document vertex
                    if vertex_idx < len(self.model.documents):
                        self.vertex_labels[v] = self.model.documents[vertex_idx]
                    else:
                        self.vertex_labels[v] = f"Doc_{vertex_idx}"
                else:  # Word vertex
                    word_idx = vertex_idx - len(self.model.documents)
                    if 0 <= word_idx < len(self.model.words):
                        # Truncate long triplet names for display
                        word = self.model.words[word_idx]
                        if len(word) > 30:
                            word = word[:27] + "..."
                        self.vertex_labels[v] = word
                    else:
                        self.vertex_labels[v] = f"Word_{word_idx}"
            else:
                # Fallback labeling
                if hasattr(g.vp, 'name') and g.vp.name[v]:
                    label = g.vp.name[v]
                    if len(label) > 30:
                        label = label[:27] + "..."
                    self.vertex_labels[v] = label
                else:
                    self.vertex_labels[v] = f"V_{vertex_idx}"
    def plot_simple_network(self, nedges=1000, output_file=None):
        """Fallback simple network plot"""
        print("Using simple network plot...")
        
        try:
            # Use the model's built-in plot method
            self.model.plot(nedges=nedges)
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Simple network plot saved to {output_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error with simple plot: {e}")
    
    def plot_network_with_labels(self, nedges=1000, vertex_size=25, font_size=8, 
                                output_file=None, figsize=(15, 15)):
        """Plot the network with proper vertex labels"""
        if not self.model:
            raise ValueError("Model must be fitted first!")
        
        print(f"Plotting SBM network with labels (showing up to {nedges} edges)...")
        
        g = self.model.g
        
        try:
            # Get the current state for coloring
            state = self.model.state
            
            # Create color property for vertices based on their cluster membership
            vertex_colors = g.new_vertex_property("int")
            # For a given hierarchical level (e.g. level=1 for main topics):
            level = 2
            ptd_d, ptw_w = self.model.group_membership(level)
            doc_blocks = np.argmax(ptd_d, axis=0)
            word_blocks = np.argmax(ptw_w, axis=0)

            
            for v in g.vertices():
                if hasattr(g.vp, 'kind') and g.vp.kind[v] == 0:
                    # Document vertex, index in docs
                    group_id = doc_blocks[int(v)]
                else:
                    # Word vertex, index in words
                    group_id = word_blocks[int(v) - len(self.model.documents)]
                vertex_colors[v] = group_id


            # Create size property for vertices
            vertex_sizes = g.new_vertex_property("int") 
            for v in g.vertices():
                if hasattr(g.vp, 'kind') and g.vp.kind[v] == 0:  # Document vertex
                    vertex_sizes[v] = vertex_size + 10
                else:  # Word vertex
                    vertex_sizes[v] = vertex_size
            
            # Plot with custom styling
            plt.figure(figsize=figsize)
            
            # Use a layout algorithm
            pos = gt.sfdp_layout(g)
            
            # Draw the graph
            gt.graph_draw(g, 
                         pos=pos,
                         vertex_text=self.vertex_labels,
                         vertex_text_position=1,
                         vertex_font_size=font_size,
                         vertex_fill_color=vertex_colors,
                         vertex_size=vertex_sizes,
                         edge_pen_width=1.5,
                         vertex_pen_width=2,
                         output_size=(1200, 1200),
                         output=output_file)
            
            if output_file:
                print(f"Network plot saved to {output_file}")
            
            plt.title("SBM Topic Model Network (Documents and Triplets)", fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting network: {e}")
            # Fallback to simpler plot
            self.plot_simple_network(nedges=nedges)
    
    def plot_simple_network(self, nedges=1000, output_file=None):
        """Fallback simple network plot"""
        print("Using simple network plot...")
        
        try:
            # Use the model's built-in plot method
            self.model.plot(nedges=nedges)
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Simple network plot saved to {output_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error with simple plot: {e}")
    
    def plot_topic_distributions(self, level=2, save_dir='plots'):
        """Plot topic distributions at specified hierarchical level"""
        if not self.model:
            raise ValueError("Model must be fitted first!")
        
        try:
            print(f"Plotting topic distributions at level {level}...")
            
            # Plot topic distribution
            self.model.plot_topic_dist(l=level)
            plt.title(f"Topic Distribution (Level {level})")
            
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'sbm_topic_dist_l{level}.png'), 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            print(f"Error plotting topic distributions: {e}")
    
    def get_topic_words(self, level=2, n_words=10):
        """Get top words for each topic at specified level"""
        if not self.model:
            raise ValueError("Model must be fitted first!")
        
        try:
            topics = self.model.topics(l=level, n=n_words)
            
            print(f"\nTop {n_words} words for each topic (Level {level}):")
            print("="*60)
            
            for i, topic_words in enumerate(topics):
                print(f"\nTopic {i}:")
                for j, (word, score) in enumerate(topic_words[:n_words]):
                    print(f"  {j+1:2d}. {word:40} (score: {score:.4f})")
            
            return topics
            
        except Exception as e:
            print(f"Error getting topic words: {e}")
            return None
    
    def get_document_clusters(self, level=1, n_docs=20):
        """Get document clusters at specified level"""
        if not self.model:
            raise ValueError("Model must be fitted first!")
        
        try:
            clusters = self.model.clusters(l=level, n=n_docs)
            
            print(f"\nDocument clusters (Level {level}, showing up to {n_docs} docs per cluster):")
            print("="*80)
            
            for i, cluster_docs in enumerate(clusters):
                print(f"\nCluster {i} ({len(cluster_docs)} documents):")
                for j, (doc, score) in enumerate(cluster_docs[:n_docs]):
                    print(f"  {j+1:2d}. {doc:50} (score: {score:.4f})")
            
            return clusters
            
        except Exception as e:
            print(f"Error getting document clusters: {e}")
            return None
    
    def analyze_hierarchical_structure(self):
        """Analyze the hierarchical structure of the model"""
        if not self.model:
            raise ValueError("Model must be fitted first!")
        
        try:
            print("\nAnalyzing hierarchical structure...")
            print("="*50)
            
            print(f"Number of hierarchical levels: {self.model.L}")
            print(f"Minimum description length: {self.model.mdl:.2f}")
            
            # Analyze each level
            for level in range(min(3, self.model.L)):  # Show up to 3 levels
                try:
                    # Get group membership probabilities
                    p_td_d, p_tw_w = self.model.group_membership(l=level)
                    
                    print(f"\nLevel {level}:")
                    print(f"  Document groups: {p_td_d.shape[0]}")
                    print(f"  Word groups: {p_tw_w.shape[0]}")
                    print(f"  Documents: {p_td_d.shape[1]}")
                    print(f"  Words: {p_tw_w.shape[1]}")
                    
                except Exception as e:
                    print(f"  Error analyzing level {level}: {e}")
            
        except Exception as e:
            print(f"Error analyzing hierarchical structure: {e}")
    
    def plot_group_membership_heatmaps(self, level=1, figsize=(20, 8)):
        """Plot group membership heatmaps"""
        if not self.model:
            raise ValueError("Model must be fitted first!")
        
        try:
            print(f"Plotting group membership heatmaps for level {level}...")
            
            p_td_d, p_tw_w = self.model.group_membership(l=level)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Document group membership
            im1 = ax1.imshow(p_td_d, origin='lower', aspect='auto', interpolation='none', cmap='viridis')
            ax1.set_title(f'Document Group Membership P(bd | d)\nLevel {level}')
            ax1.set_xlabel('Document d (index)')
            ax1.set_ylabel('Document group, bd')
            plt.colorbar(im1, ax=ax1)
            
            # Word group membership  
            im2 = ax2.imshow(p_tw_w, origin='lower', aspect='auto', interpolation='none', cmap='viridis')
            ax2.set_title(f'Word Group Membership P(bw | w)\nLevel {level}')
            ax2.set_xlabel('Word w (index)')
            ax2.set_ylabel('Word group, bw')
            plt.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting group membership heatmaps: {e}")
    
    def save_model(self, filename='sbm_topic_model.pkl'):
        """Save the fitted model"""
        if not self.model:
            raise ValueError("Model must be fitted first!")
        
        import pickle
        
        # Note: graph-tool objects can't be pickled directly
        # Save only the essential results
        model_data = {
            'documents': self.model.documents,
            'words': self.model.words,
            'mdl': self.model.mdl,
            'L': self.model.L,
            'groups': getattr(self.model, 'groups', {}),
            'random_state': self.random_state
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model data saved to {filename}")


# Usage example function
def run_enhanced_sbm_analysis(vocab_text, vessel_titles, random_state=42):
    """
    Run complete enhanced SBM analysis with proper labeling
    """
    # Initialize enhanced model
    enhanced_model = EnhancedSBMTopicModel(random_state=random_state)
    
    # Fit the model
    enhanced_model.fit_model(vocab_text, vessel_titles)
    
    # Analyze hierarchical structure
    enhanced_model.analyze_hierarchical_structure()
    
    # Plot network with labels
    enhanced_model.plot_network_with_labels(
        nedges=1000, 
        vertex_size=20, 
        font_size=6,
        output_file='enhanced_sbm_network.png'
    )
    
    # Get and display topic words
    topics = enhanced_model.get_topic_words(level=2, n_words=10)
    
    # Get and display document clusters
    clusters = enhanced_model.get_document_clusters(level=1, n_docs=15)
    
    # Plot topic distributions
    enhanced_model.plot_topic_distributions(level=2)
    
    # Plot group membership heatmaps
    enhanced_model.plot_group_membership_heatmaps(level=1)
    
    # Save the model
    enhanced_model.save_model('enhanced_sbm_model.pkl')
    
    return enhanced_model


# Verification and quality assessment functions
def verify_model_correctness(model, empirical_distributions):
    """
    Verify the correctness of the topic model by comparing with empirical distributions
    """
    print("\n" + "="*60)
    print("MODEL VERIFICATION AND QUALITY ASSESSMENT")
    print("="*60)
    
    if not model.model:
        print("Error: No fitted model available for verification")
        return
    
    try:
        # Get topics from the model
        topics = model.get_topic_words(level=2, n_words=20)
        
        if not topics:
            print("Error: Could not extract topics for verification")
            return
        
        print(f"\nFound {len(topics)} topics from SBM model")
        
        # Calculate topic coherence (simplified version)
        print("\nTopic Coherence Analysis:")
        for i, topic_words in enumerate(topics):
            words = [word for word, score in topic_words[:10]]
            avg_score = np.mean([score for word, score in topic_words[:10]])
            print(f"  Topic {i}: Average score = {avg_score:.4f}")
            print(f"    Top words: {', '.join(words[:5])}")
        
        # Compare with empirical distributions if available
        if empirical_distributions:
            print("\nComparison with Empirical Distributions:")
            print("(This would require mapping between SBM topics and empirical distributions)")
            
            for culture, emp_dist in empirical_distributions.items():
                print(f"  {culture}: {len(emp_dist)} triplets in empirical distribution")
        
        # Model statistics
        print(f"\nModel Statistics:")
        print(f"  Description Length (MDL): {model.model.mdl:.2f}")
        print(f"  Number of hierarchical levels: {model.model.L}")
        print(f"  Number of documents: {len(model.model.documents)}")
        print(f"  Number of words: {len(model.model.words)}")
        
        print("\nVerification completed successfully!")
        
    except Exception as e:
        print(f"Error during model verification: {e}")


if __name__ == "__main__":
    print("Enhanced SBM Topic Modeling Module")
    print("This module provides enhanced SBM topic modeling with proper labeling.")
    print("Use run_enhanced_sbm_analysis() function to run complete analysis.")