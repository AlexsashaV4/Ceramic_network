import numpy as np 
import networkx as nx 
import os 
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import plotly.graph_objects as go
import plotly.colors as pc
from IPython.display import HTML
import gravis as gv

def graph_from_data(data):
    """Return a directed graph from the dataset where each row is a work-chain of the pot. 

    Each step in the work chain is a node in the graph, and each transition from one step to the next is an edge in the graph.

    **The edge weights are the number of times the transition appears in the dataset**, and the node 'count' attribute is the number of times the step appears in the dataset.

    Args:
        data: pandas dataset, where each row represents a work-chain and each column represents a step in the work-chain. (Thework-chains can have a different length)
    Returns:
        G: a directed graph where each node is a step in the work-chain and each edge is a transition from one step to the next.
    """
    G = nx.DiGraph()
    for i in range(len(data)):
        path = data.iloc[i].dropna().tolist()
        for j in range(len(path) - 1):
            G.add_edge(path[j], path[j + 1])
            G[path[j]][path[j + 1]]['weight'] = G[path[j]][path[j + 1]].get('weight', 0) + 1
            G.nodes[path[j]]['count'] = G.nodes[path[j]].get('count', 0) + 1

    start_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    total_start_count = sum(G.nodes[node].get('count', 0) for node in start_nodes)
    for node in start_nodes:
        G.nodes[node]['prob'] = G.nodes[node].get('count', 0) / total_start_count if total_start_count > 0 else 0 
    total_weights = {node: G.out_degree(node, weight='weight') for node in G.nodes()}
    for u, v in G.edges():
        G[u][v]['weight'] /= total_weights[u] if total_weights[u] > 0 else 1 
    nx.set_node_attributes(G, {node: G.degree(node) for node in G.nodes()}, 'deg')
    nx.set_node_attributes(G, {node: G.in_degree(node) for node in G.nodes()}, 'in_deg')
    nx.set_node_attributes(G, {node: G.out_degree(node) for node in G.nodes()}, 'out_deg')

    return G

def create_flow_layout(G, width=1000, height=800, margin=50):
    """
    For better visualisation of the graph, where on the left we have the initial nodes and on the right the final ones. If the graph has cycles, it will use a spring layout.
    just found the function "topological sort" on the internet, it is not mine, but I think it is a good way to create a flow layout for a directed graph.
    Returns:
        pos: Dictionary mapping nodes to (x,y) positions
    """


    source_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    final_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    
    # Like dijkstra algorithm 
    ranks = {node: -1 for node in G.nodes()}
    
    for source in source_nodes:
        ranks[source] = 0
    
    # This will only work if the graph is a DAG (Directed Acyclic Graph)
    if nx.is_directed_acyclic_graph(G):
        for node in nx.topological_sort(G):
            # Node's rank is at least its current rank
            current_rank = ranks[node]
            
            # For each successor, increse its rank by 1
            for successor in G.successors(node):
                ranks[successor] = max(ranks[successor], current_rank + 1)
        # This is to give a "Directionality to the graph"
    else:
        # If the graph is not a DAG, we need to handle cycles or unreachable nodes (if it has cycles there is some confusion in how we assign the ranks, haven't found a good solution yet)
        # use simple layoout es spring layout
        pos = nx.spring_layout(G, scale=600, seed=42)  # Use spring layout for better visualization
        return pos
        
    # Check and assign some position to nodes that weren't reached (cycles or unreachable from sources) (impossible to reach) so in theory never used
    max_rank = max(ranks.values()) if ranks.values() else 0
    for node, rank in ranks.items():
        if rank == -1:  # Node wasn't reached
            pred_ranks = [ranks[pred] for pred in G.predecessors(node) if ranks[pred] >= 0]
            if pred_ranks:
                ranks[node] = max(pred_ranks) + 1
            else:
                ranks[node] = max_rank // 2
    
    # Update max rank
    max_rank = max(ranks.values()) if ranks.values() else 0
    
    # So ranks is used as a collum placing the nodes in the graph, example rank = 1 --> place after on the right of the source nodes, rank = 2 --> place after the rank 1 nodes, etc.
    rank_counts = {}
    for node, rank in ranks.items():
        if rank not in rank_counts:
            rank_counts[rank] = 0
        rank_counts[rank] += 1
    
    # Assign positions based on ranks
    pos = {}
    for node, rank in ranks.items():
        # X-coordinate: distribute ranks evenly
        x = margin + (width - 2 * margin) * rank / max(max_rank, 1)
        
        # Y-coordinate: distribute evenly nodes within their rank
        # Find position of this node within its rank
        nodes_in_rank = [n for n, r in ranks.items() if r == rank]
        node_index = nodes_in_rank.index(node)
        total_in_rank = len(nodes_in_rank)
        
        if total_in_rank == 1:
            y = height / 2 
        else:
            y = margin + (height - 2 * margin) * node_index / (total_in_rank - 1)
        
        pos[node] = (x, y)
    
    # optimize positions to reduce edge crossings using the barycenter method
    for _ in range(3):
        for r in range(1, max_rank + 1):  # Start from rank 1 (not sources)
            nodes_at_rank = [node for node, rank in ranks.items() if rank == r]
            
            # For each node, calculate the average y-position of its predecessors
            barycenters = {}
            for node in nodes_at_rank:
                predecessors = list(G.predecessors(node))
                if predecessors:
                    avg_y = sum(pos[pred][1] for pred in predecessors) / len(predecessors)
                    barycenters[node] = avg_y
            
            # Sort nodes by barycenter
            if barycenters:
                sorted_nodes = sorted(barycenters.keys(), key=lambda n: barycenters[n])
                
                # Reposition nodes by their barycenter order
                for i, node in enumerate(sorted_nodes):
                    if len(sorted_nodes) == 1:
                        y = height / 2
                    else:
                        y = margin + (height - 2 * margin) * i / (len(sorted_nodes) - 1)
                    pos[node] = (pos[node][0], y)
    
    # Place source nodes on the leftmost 
    leftmost_x = margin
    for node in source_nodes:
        pos[node] = (leftmost_x - 150, pos[node][1])
    
    # Place sink nodes on the rightmost 
    rightmost_x = width - margin
    for node in final_nodes:
        pos[node] = (rightmost_x + 150 , pos[node][1])
    
    return pos

def cluster_nodes(data,  group_size = 3, num_data = 0, step = 1, coloring = False, position = None, save_paths_with_clustered_nodes = False, filename = "clustered_paths", type_edge_weight = 'probability'):
    #One of the most important function (I think is definetly correct), for triplets works well some adaptation need for other group sizes.
    """Ceramic graph from the dataset. (Maybe the most controversial function ahahhaa) #Generic group size still to be implemented!!
    The idea is to group the work steps into triplets (or any other group size),and each edge is a hypothetical  transition from one triplet to another.

    For example, consider the following work-chain:  Wet clay ; Modelling ; Pressure ; Wet smoothing ; Leather-hard ; Dry ; Open firing. 
    If group_size = 1, we will have the following nodes: Wet clay --> Modelling --> Pressure --> Wet smoothing --> Leather-hard --> Dry --> Open firing
    if the group_size = 3, we will have the following nodes: (Wet clay, Modelling, Pressure) --> (Modelling, Pressure, Wet smoothing) --> (Pressure, Wet smoothing, Leather-hard) --> (Wet smoothing, Leather-hard, Dry) --> (Leather-hard, Dry, Open firing)
    
    The edges are created between nodes that share the same step in the work-chain, for example:
    (Wet clay, Modelling, Pressure) --> (Modelling, Pressure, Wet smoothing) because the second node starts with the second step of the first node.

    Furthermore, when we cluster we save the sequence of triplets in a dataframe, where each row is a path and each column is a node triplet.

    Last but not least, we update the edge weights (probability) P(A,B,C) = P(C|B,A) * P(B|A) * P(A), where A, B, C are the triplets. For the first element we assume P(A) = 1. 

    Then when we group and calculate the path probabilities, we can use the edge weights to calculate the probability of a path as the product of the edge weights along the path and the initial P(A).

    On the other hand a more heuristic approach (I assume that they aare equivalent) is to use the edge weights as the number of times the transition appears in the dataset, and then normalize the edge weights by the outgoing degree of the source node.

    So one comes from the single subgraphs weights, the other comes directly from the dataset

    Maybe need a better explanation of the edge weights and how they are calculated.
    Returns:
        G: a directed graph where each node is a triplet of steps in the work-chain and each edge is a transition from one triplet to another.
    """

    if num_data != 0:
        tot_data=num_data
    else:
        tot_data = len(data)
    G = nx.DiGraph()
    if type_edge_weight not in ['from_data', 'probability']:
        raise ValueError("type_edge_weight must be either 'from_data' or 'probability'")
    if type_edge_weight == 'probability':
        G_tmp = graph_from_data(data)
    if save_paths_with_clustered_nodes:
        cluster_path = []
    # next iterate over the dataset and create the triplets
    for i in range(tot_data):
        path = data.iloc[i].dropna().tolist()
        tmp_clust_path = []
        for j in range(0, len(path) - group_size + 1, step):
            triplet = tuple(path[j:j + group_size])
            if len(triplet) == group_size:
                G.add_node(triplet)
                if j > 0:  # Ensure there is a previous triplet to connect to
                    prev_triplet = tuple(path[j - step:j - step + group_size])
                    if len(prev_triplet) == group_size:
                        G.add_edge(prev_triplet, triplet)
                        if type_edge_weight == 'from_data': 
                            G[prev_triplet][triplet]['weight'] = G[prev_triplet][triplet].get('weight', 0) + 1 
                        elif type_edge_weight == 'probability':
                            G[prev_triplet][triplet]['weight'] = G_tmp[prev_triplet[-1]][triplet[-1]]['weight'] # construct the edge weight from the previous triplet to the current triplet based on the "less coarse graph" o il graph di supporto come quello delle sole tecniche
                            # Entrambi gli approci dovrebbero essere equivalenti
                else:
                    # If it's the first triplet, we can assume it has no incoming edges, but now we will have a probabability to start from it P(triplet) = P(C|B,A) * P(B|A) * P(A)
                    if type_edge_weight == 'probability': # add a new attribute to the node with the probability of starting from it
                        G.nodes[triplet]['prob'] = np.prod([G_tmp[triplet[i]][triplet[i + 1]]['weight'] for i in range(group_size - 1)]) * 1 # P(A) = 1 for the first triplet we always start from "wet clay"
                    # generalise for any group size 
                    elif type_edge_weight == 'from_data':
                        G.nodes[triplet]['count'] = G.nodes[triplet].get('count', 0) + 1 
            if save_paths_with_clustered_nodes:
                tmp_clust_path.append(triplet)
        if save_paths_with_clustered_nodes:
            cluster_path.append(tmp_clust_path)
    if save_paths_with_clustered_nodes:
        df = pd.DataFrame(cluster_path)
        df.to_csv(f'{filename}.csv', index=False, header=False, sep=';')
    total_weights = {node: G.out_degree(node, weight='weight') for node in G.nodes()}
    for u, v in G.edges():
        G[u][v]['weight'] /= total_weights[u] if total_weights[u] > 0 else 1  
    for u, v in G.edges():
        G[u][v]['label'] = f"{G[u][v]['weight']:.2f}"

    if coloring:
        # Find start nodes (no incoming edges)
        start_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
        end_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]

        # Assign colors
        color_map = {}
        for node in G.nodes():
            if node in start_nodes:
                color_map[node] = 'green'
            if node in end_nodes:
                color_map[node] = 'red'

        degrees = np.array([G.degree(node) for node in G.nodes()])
        nx.set_node_attributes(G, {node: G.nodes[node].get('label', node) for node in G.nodes()}, 'label')
        if 'prob' in G.nodes[next(iter(G.nodes()))]:
            nx.set_node_attributes(G, {node: G.nodes[node].get('prob', 1.0) for node in G.nodes()}, 'prob')

        nx.set_node_attributes(G, {node: G.degree(node) for node in G.nodes()}, 'deg')
        nx.set_node_attributes(G, {node: G.in_degree(node) for node in G.nodes()}, 'in_deg')
        nx.set_node_attributes(G, {node: G.out_degree(node) for node in G.nodes()}, 'out_deg')

        norm = plt.Normalize(degrees.min(), degrees.max())
        cmap = plt.get_cmap(coloring)
        for node in G.nodes():
            if node not in color_map:
                color_value = cmap(norm(G.degree(node)))
                color_map[node] = f'rgba({int(color_value[0] * 255)}, {int(color_value[1] * 255)}, {int(color_value[2] * 255)}, 1)'
        nx.set_node_attributes(G, color_map, 'color')
        if position is not None:
            G = add_positions(G, position=position)
        return G
    else:
        # If no coloring is needed, just return the graph
        nx.set_node_attributes(G, {node: G.nodes[node].get('label', node) for node in G.nodes()}, 'label')
        if 'prob' in G.nodes[next(iter(G.nodes()))]:
            nx.set_node_attributes(G, {node: G.nodes[node].get('prob', 1.0) for node in G.nodes()}, 'prob')
        nx.set_node_attributes(G, {node: G.degree(node) for node in G.nodes()}, 'deg')
        nx.set_node_attributes(G, {node: G.in_degree(node) for node in G.nodes()}, 'in_deg')
        nx.set_node_attributes(G, {node: G.out_degree(node) for node in G.nodes()}, 'out_deg')
        if position is not None:
            G = add_positions(G, position=position)
        return G

def add_positions(G, width=1000, height=800, margin=50 , position=None):
    """Add positions to the nodes of the graph.
    Args:
        G: a directed graph (networkx DiGraph) where nodes may have low in+out degrees.
        position: networkx layout to use for the positions, if 'flow', it will use the create_flow_layout function.
    Returns:
        G: a new directed graph with positions added to the nodes.
    """
    if position == 'flow':
        # If positions are provided, use them directly
        pos = create_flow_layout(G, width=width, height=height, margin=margin)
    else:
        pos = position

    # Add coordinates as node annotations that are recognized by gravis
    for node, (x, y) in pos.items():
        G.nodes[node]['x'] = x
        G.nodes[node]['y'] = y

    return G

def contract_low_degree_nodes(G, position=None, remove_chain=False):
    """Contract low-degree nodes in a directed graph.
    This function identifies chains of low-degree nodes (in+out degree ≤ 2) and contracts them into a single node.

    So far working only for the case of triplets. Should be adapted to any type of node collapse!!
    Args:

        G: A directed graph (networkx DiGraph) where nodes may have low in+out degrees.
    Returns:
        G: A new directed graph with low-degree nodes contracted into single nodes.
        contracted_nodes: A set of new node IDs that were created by contracting low-degree nodes.
    """
    
    contracted_nodes = set()
    low_degree_nodes = [node for node in G.nodes() if G.in_degree(node) ==1 and G.out_degree(node) == 1]
    components = []
    visited = set()

    for node in low_degree_nodes:
        if node in visited:
            continue
            
        chain = [node]
        visited.add(node)
        
        current = node
        while True:
            successors = list(G.successors(current))
            if len(successors) != 1:
                break
                
            next_node = successors[0]
            if next_node in visited or next_node not in low_degree_nodes:
                break
                
            chain.append(next_node)
            visited.add(next_node)
            current = next_node
            
        current = node
        while True:
            predecessors = list(G.predecessors(current))
            if len(predecessors) != 1:
                break
                
            prev_node = predecessors[0]
            if prev_node in visited or prev_node not in low_degree_nodes:
                break
                
            chain.insert(0, prev_node)
            visited.add(prev_node)
            current = prev_node
        if len(chain) >= 1:
            components.append(chain)
    for chain in components:
        first_node = chain[0]
        last_node = chain[-1]
        if len(first_node) == 3 and len(last_node) == 3:
            new_node_id = f"{first_node[0]}-"
            for ch in range(1, len(chain), 1):
                new_node_id += f"{chain[ch][0]}-"
            
            new_node_id += f"{last_node[1]}-{last_node[2]}"
            #new_node_id = (f"{first_node[0]}", f"{first_node[1]}-{last_node[1]}", f"{last_node[2]}")
        else:
            # If nodes are not triplets, just combine their string representations
            new_node_id = f"{first_node}+{last_node}"
            
        # Mark this as a contracted node
        contracted_nodes.add(new_node_id)

        # Add the new node contracted node to the graph and remove the single nodes that were contracted, maintain the weights of the edges, because if we contracted a chain of node all the internal edges were 1
        if remove_chain == False:
            G.add_node(new_node_id, color='black', deg=1, in_deg=1, out_deg=1, label=new_node_id)
            # Find all incoming edges to the chain
            in_edges = []
            for node in chain:
                for pred in G.predecessors(node):
                    if pred not in chain:
                        in_edges.append((pred, new_node_id, G[pred][node].get('weight', 1)))
            
            # Find all outgoing edges from the chain 
            out_edges = []
            for node in chain:
                for succ in G.successors(node):
                    if succ not in chain:
                        out_edges.append((new_node_id, succ, G[node][succ].get('weight', 1)))
            
            # Add the new edges with weights
            for u, v, w in in_edges:
                G.add_edge(u, v, weight=w)
            for u, v, w in out_edges:
                G.add_edge(u, v, weight=w)
            
            # Remove the original nodes in the chain
            G.remove_nodes_from(chain)
        else:
            # remove chain
            for node in chain:
                predecessors = list(G.predecessors(node))
                successors = list(G.successors(node))
                
                # Connect predecessors to the successor of the chain
                for pred in predecessors:
                    for succ in successors:
                        G.add_edge(pred, succ, weight=G[pred][node].get('weight', 1))
            # Remove the original nodes in the chain
            G.remove_nodes_from(chain)

        if position is not None:
            G = add_positions(G, position=position)

    return G, components, contracted_nodes
    

def find_most_probable_path(G, start_node, end_node):
    """Find the most probable path in a directed graph from start_node to end_node.
    Args:
        G: A directed graph (networkx DiGraph).
        start_node: The starting node ID.
        end_node: The ending node ID.
    Returns:
    """
    path = []
    if start_node == 0:
        # start from the most probable initial node 
        start_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
        if start_nodes:
            # Choose the start node with the highest probability
            start_node = max(start_nodes, key=lambda n: G.nodes[n].get('prob', 0))
            path[0] = start_node
            current_node = start_node
    else:
        path = [start_node]
        current_node = start_node
    while current_node != end_node:
        neighbors = list(G.successors(current_node))
        if not neighbors:
            break  # No more neighbors to explore
        
        # Get the weights of the edges to the neighbors
        weights = [G[current_node][neighbor]['weight'] for neighbor in neighbors]
        
        # Normalize the weights to get probabilities
        total_weight = sum(weights)
        probabilities = [weight / total_weight for weight in weights]
        
        # Choose the next node based on the probabilities
        next_node = np.random.choice(neighbors, p=probabilities)
        path.append(next_node)
        current_node = next_node
    
    return path

def random_path_walk(G, max_steps=100):
    """Perform a random walk in a directed graph from start_node to end_node.
    Args:
        G: A directed graph (networkx DiGraph).
        start_node: The starting node ID.
        end_node: The ending node ID.
        max_steps: Maximum number of steps to take in the random walk.
    Returns:
        path: A list of node IDs representing the path taken in the random walk.
    """
    start_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    # chose the start node with the highest probability
    start_node = max(start_nodes, key=lambda n: G.nodes[n].get('prob', 0))
    
    path = [start_node]
    current_node = start_node
    steps = 0
    end_node = [node for node in G.nodes() if G.out_degree(node) == 0]
    
    while current_node not in end_node and steps < max_steps:
        neighbors = list(G.successors(current_node))
        if not neighbors:
            break  # No more neighbors to explore
        
        # Choose a random neighbor
        next_node = np.random.choice(neighbors)
        path.append(next_node)
        current_node = next_node
        steps += 1
    
    return path

def path_pdf(data):
    """Calculate the path probability distribution (from the dataset)!, by simple counting the presence.
    Args:
        data: pandas dataset, where each row represents a work-chain and each column represents a step in the work-chain.
    Returns:
        path_probabilities: A dictionary where keys are paths (tuples) and values are their probabilities.
    """
    path_counts = Counter()
    
    # Count the occurrences of each path in the data
    for i in range(len(data)):
        path = data.iloc[i].dropna().tolist()
        if len(path) > 1:
            path_counts[tuple(path)] += 1
    
    # Calculate the total number of paths
    total_paths = sum(path_counts.values())
    print(f"Total paths: {total_paths}") # check they should be the same as the number of rows in the dataset 
    
    # Create a probability distribution for each unique path
    path_probabilities = {path: count / total_paths for path, count in path_counts.items()}
    
    return path_probabilities

def path_pdf_weights(data, type_edge_weight = 'probability'): 
    """Calculate the path probability distribution from the dataset, by using the weights of the edges in the graph.

    Args:
        data: pandas dataset,  where each row represents a work-chain and each column represents a step in the work-chain.

    Returns:
        _type_: _description_
    """
    path_probabilities = {}
    G = graph_from_data(data)
    for i in range(len(data)):
        path = data.iloc[i].dropna().tolist()
        if len(path) > 1:
            prob = 1.0
            for j in range(len(path) - 1):
                u = path[j]
                v = path[j + 1]
                if j == 0 and type_edge_weight == 'from_data': ## Probability of starting node
                    prob *= G.nodes[u].get('count', 1) / sum(G.nodes[n].get('count', 1) for n in G.nodes())
                elif j == 0 and type_edge_weight == 'probability':
                    prob *= G.nodes[u].get('prob', 1)          
                
                # multiply the probability of the edge P(path = (C,B,A)) = P(C|B,A) * P(B|A) * P(A)
                if G.has_edge(u, v):
                    prob *= G[u][v]['weight']
                else:
                    prob *= 0
            path_probabilities[tuple(path)] = path_probabilities.get(tuple(path), 0) + prob
    # Normalize the path probabilities
    total_prob = sum(path_probabilities.values())
    if total_prob > 0:
        path_probabilities = {path: prob / total_prob for path, prob in path_probabilities.items()}
    else:
        path_probabilities = {path: 0 for path in path_probabilities}
    return path_probabilities

def path_pdf_clustered_graph(dataset,save_paths_with_clustered_nodes = False, filename = "./data/clustered_paths"):
    """Calculate the path probability distribution from the dataset, by using the weights of the edges in the graph.
        However the paths are constrained to the paths that are present in the dataset.
        at certin point we create the clustered graph from the dataset, and then we calculate the path probabilities based on the edges of the clustered graph.
    Args:
        data: pandas dataset, where each row represents a work-chain and each column represents a step in the work-chain.
    Returns:
        path_probabilities: A dictionary where keys are paths (tuples) and values are their probabilities.
    """
    # if cannot finde the file, create the clustered graph
    if not os.path.exists(f'{filename}.csv'):
        print(f"File {filename}.csv not found, creating the clustered graph...")
        G_clust = cluster_nodes(dataset, group_size=3, save_paths_with_clustered_nodes=save_paths_with_clustered_nodes, filename= filename)
    else:
        print(f"File {filename}.csv found, loading the clustered graph...")
        G_clust = graph_from_data(pd.read_csv(f'{filename}.csv', sep=';'))
    
    # Calculate the path probabilities based on the weights edges of the clustered graph
    path_probabilities = path_pdf_weights(pd.read_csv(f'{filename}.csv', sep=';'))
    # uno non serve all'altro perchè ho fatto tutto a partire dal dataset s
    return path_probabilities, G_clust 

def plot_path_pdf(path_probabilities,height = 400, width = 800 , show = False, save = False, filename = "path_probabilities.html"):
    """This function plot the path probability distribution as a bar chart and orders the paths from the most common to the least common.
    
    Args:
        path_probabilities: A dictionary where keys are paths (tuples) and values are their probabilities.
        title: Title of the plot.
    """
    paths = list(path_probabilities.keys())
    probabilities = list(path_probabilities.values())
    # sort paths by probability
    sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
    paths = [paths[i] for i in sorted_indices]
    probabilities = [probabilities[i] for i in sorted_indices]
    
    
    fig = go.Figure(data=[go.Bar(
        #x=[' -> '.join(path) for path in paths],
        y=probabilities,
        marker_color='green',
        hovertext=[' -> '.join(path) for path in paths],
    )])
    
    fig.update_layout(
        title='Path Probabilities',
        xaxis_title='Paths',
        yaxis_title='Probability',
        height=height,
        width=width
    )
    if show:
        fig.show()
    if save:
        fig.write_html(filename, include_plotlyjs='cdn', full_html=True)
    
    return fig


def plot_path_abundance_comparison(path_probabilities1, path_probabilities2, name1 = "Dataset 1", name2 = "Dataset 2", show = False, save= False, filename = "path_abundance_comparison.html"):
    """ This function plots the path abundance comparison between two datasets as a bar chart.
    """
    # Extract paths and probabilities from the first dataset
    paths1 = list(path_probabilities1.keys())
    probabilities1 = list(path_probabilities1.values())
    # Sort paths by probability
    sorted_indices1 = np.argsort(probabilities1)[::-1]  # Sort in descending order
    paths1 = [paths1[i] for i in sorted_indices1]
    probabilities1 = [probabilities1[i] for i in sorted_indices1]
    # Extract paths and probabilities from the second dataset
    paths2 = list(path_probabilities2.keys())
    probabilities2 = list(path_probabilities2.values())
    all_paths = set(paths1) | set(paths2)  # Union of all paths from both datasets
    fig = go.Figure(data=[
        go.Bar(
            #x=[str(path) for path in all_paths],
            y=probabilities1,
            name=name1,
            marker_color='blue',
            opacity=0.6,
            hovertext=[f"{path}: {prob:.2f}" for path, prob in zip(all_paths, probabilities1)]
        ),
        go.Bar(
            #x=[str(path) for path in all_paths],
            y=probabilities2,
            name=name2,
            marker_color='red',
            opacity=0.6,
            hovertext=[f"{path}: {prob:.2f}" for path, prob in zip(all_paths, probabilities2)],
        )
    ])
    fig.update_layout(
        title='Path Abundance Comparison',
        xaxis_title='Paths',
        yaxis_title='Probability',
        barmode='overlay',
        height=400,
        width=800
    )
    if show:
        fig.show()
    if save:
        fig.write_html(filename, include_plotlyjs='cdn', full_html=True)
    return fig

def plot_path_abundance_multiple_comparison(path_probabilities_list, names=None, show=False, save=False, filename="path_abundance_multiple_comparison.html"):

    """ This function plots the path abundance comparison between multiple datasets as a bar chart.
    """
    if names is None:
        names = [f"Dataset {i+1}" for i in range(len(path_probabilities_list))]
    
    fig = go.Figure()
    # fix the paths on the x axis before, so that the bars are aligned, set the x axis to the union of all paths without duplicates
    all_paths = set()
    for path_probabilities in path_probabilities_list:
        all_paths.update(path_probabilities.keys())
    all_paths = sorted(all_paths)  # Sort paths for consistent ordering
    # Create a bar for each dataset

    for i, path_probabilities in enumerate(path_probabilities_list):
        paths = list(path_probabilities.keys())
        probabilities = list(path_probabilities.values())
        # Sort paths by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        paths = [paths[i] for i in sorted_indices]
        probabilities = [probabilities[i] for i in sorted_indices]
        # Create a bar for the current dataset
        fig.add_trace(go.Bar(
            #x=[str(path) for path in all_paths],
            y=[path_probabilities.get(path, 0) for path in all_paths],  # Fill missing paths with 0
            name=names[i],
            marker_color=pc.qualitative.Plotly[i % len(pc.qualitative.Plotly)],
            opacity=0.6,
            hovertext=[f"{path}: {prob:.2f}" for path, prob in zip(all_paths, [path_probabilities.get(path, 0) for path in all_paths])],
        ))

    fig.update_layout(
        title='Path Abundance Comparison',
        xaxis_title='Paths',
        yaxis_title='Probability',
        barmode='overlay',
        height=400,
        width=800
    )
    if show:
        fig.show()
    if save:        
        fig.write_html(filename, include_plotlyjs='cdn', full_html=True)
    return fig  


"""
From here some functions are taken inspiration from: 
Author: E.J. Kroon

v1 June 27, 2023

Please refer to Kroon (in prep.) when using this program.

Pathfinder is meant to compare, generate, and simulate ceramic
chaînes opératoires. 

This is a module of pathfinder. It randomly generates a user-
specified number of chaînes opératoires and outputs these to
a .csv file. See Kroon (in prep. Section 4.2) for a description 
of this algorithm.

The module makes use of separate, open source libraries
in python: networkx, csv and random. See the documentation of 
these respective libaries. The script is provided as is under
a CC-BY 4.0 license.

"""


