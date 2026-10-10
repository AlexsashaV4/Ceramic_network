"""
Author: E.J. Kroon

v1 June 27, 2023

Please refer to Kroon (in prep.) when using this program.

Pathfinder is meant to compare, generate, and simulate ceramic
chaînes opératoires. This is one of two modules for guided 
random path generation. The algorithm below takes a link list 
as a random seed. This link list shows the frequencies of 
certain combinations of techniques in a prior dataset. The
algorithm then generates new paths for which the selection of
target nodes is proportional to the weight of the edge from a
source node to that target node.

Users can change other parameters as well, such as the default
value of links with a zero value in the random seed, the
multiplier for the value of links which do occur in the random
seed, the recursion factor, and the number of iterations. The 
combination of these variables enables the user to perform 
more complex simulations.

The script parses a list of all available neighbours to a 
source, where each neighbour occurs as many times on the list
as the edge weight from the source. Therefore, computation 
time increases significantly with larger edge weights.


This module makes use of separate, open source libraries in 
python: networkx, csv, and random. See the documentation for 
these libraries. The script is provided as is under a CC-BY 
4.0 license.

"""

import networkx as nx
import csv
from random import choice

#Set parameters based on user input
def guided_random_generator(s_f, d_v, l_m, r_f, i, o):
    file_nodes = 'ceramics/nodes_list.csv'
    file_links = s_f
    output_file = 'ceramics/seeded_output.csv'
    default_value = int(d_v)
    link_multiplier = int(l_m)
    recursion_factor = int(r_f)
    iterations = int(i)
    on_of_paths = int(o)

    #Extract node information from files
    with open(file_nodes, 'r', encoding='utf-8') as nodes_list:
        nodereader = csv.reader(nodes_list, delimiter=';')
        nodes = [n for n in nodereader][1:]
    node_ids = [n[0] for n in nodes]

    #Open link list (node_IDs for target, source) with headers
    with open(file_links, 'r', encoding='utf-8') as links_list:
        linkreader = csv.reader(links_list, delimiter=';')
        links = [tuple(l) for l in linkreader][1:]
        weighted_links = []
        #Apply default value and liunk mulitplier
        for link in links:
            if int(link[2]) == 0:
                weighted_links.append((link[0], link[1],int(link[2])+default_value))
            elif int(link[2])>0:
                weighted_links.append((link[0], link[1], int(link[2])*link_multiplier)) 

    #Create a directed network from nodes and links
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    G.add_weighted_edges_from(weighted_links)

    #Create dictionairies for each attribute in node_list
    node_names_dict ={}
    node_tolerance_dict={}

    #Fill dictionairies
    for node in nodes:
        node_names_dict[node[0]] = node[1]
        node_tolerance_dict[node[0]] = node[2], node[3], node[4]

    #Set dictionairies as node attributes
    nx.set_node_attributes(G, node_names_dict, 'node_name')
    nx.set_node_attributes(G, node_tolerance_dict, 'tolerance')

    weight = nx.get_edge_attributes(G, 'weight')

    #Function to break up generated paths and adjust edge weights
    def process_result(n_l):
        #Store generated path
        with open(output_file, 'a', newline = '') as f:
            writer = csv.writer(f, delimiter = ';')
            writer.writerow(n_l)
            del n_l[0]
            #Adjust the weight of edges in generated path
            for node in n_l:
                try:
                    weight = nx.get_edge_attributes(G, 'weight')
                    relevant_edge = (n_l[0], n_l[1])
                    new_value = weight[(n_l[0], n_l[1])] + 1*recursion_factor
                    nx.set_edge_attributes(G, {relevant_edge:{'weight':new_value}})
                    del n_l[0]
                except IndexError:
                    break

    #Set flag for simulation
    successes = 0
    while successes <= iterations:
        
        #Define empty node list and starting node
        node_list = ['W', '1',]

        #Set flag for pathfinder loop
        active = True

        #Pathfinder loop
        while active:

            #Find adjacent nodes to last node on list 
            potential_targets = list(G.neighbors(node_list[-1]))
            #determine hygrometric tolerance for each target
            selected_targets = []
            for potential_target in potential_targets:
                tol = G.nodes[potential_target]['tolerance']
                #If path state != node tolerance, remove target from list
                if not node_list[0] in tol:
                    potential_targets.remove(potential_target)
                #If node tolerance matches path state, append node as target    
                if node_list[0] in tol:
                    selected_targets.append(potential_target)
        
            #Modify the target list for observed distribution of technical sequences
            targets = []
            for selected_target in selected_targets:
                #Append the selected_target once for each time the link is observed
                wt = int(weight[node_list[-1], selected_target])
                for _ in range(wt):
                    targets.append(selected_target)

            #Randomly choose node from target list, except when list is empty
            try:
                new_node = choice(targets)
            except IndexError:
                #If no targets are listed, end cycle, unless path state is 'F'
                if node_list[0] == 'F':
                    #If no targets + path state = 'F' and process result
                    successes += 1
                    process_result(node_list)
                    active = False
                else:
                    #If no listed targets + path state is not 'F': end cycle
                    active = False
            # If the chosen target node is already in node list
            if new_node in node_list:
                if node_list[0] == 'F':
                #and path state is 'F', process results
                    successes += 1
                    process_result(node_list)
                    active = False
                else: 
                #In all other path states: a path loop ends the cycle
                    active = False
            else:    
                # Append new_node to node list
                node_list.append(new_node)
            
            #If the new_node implies hygrometric change, update path state
            if new_node == '8':
                node_list[0] = 'L'
            if new_node == '22':
                node_list[0] = 'D'
            if new_node == '45':
                node_list[0] = 'F'
            if new_node == '46':
                node_list[0] = 'F'
            if new_node == '47':
                node_list[0] = 'F'
            if new_node == '48':
                node_list[0] = 'F'
            elif new_node == '49':
                node_list[0] = 'F'

            #Then roll dice to see if path continues, if state == F
            if node_list[0] == 'F':
                choice_list = []
                odds = 0
                neighbour_list = list(G.neighbors(node_list[-1]))
                for neighbour in neighbour_list:
                    odds += int(weight[node_list[-1], neighbour])
                for o in range(odds):
                    choice_list.append('yes')
                total_paths = int(on_of_paths)+(successes*recursion_factor)
                for t in range(total_paths):
                    choice_list.append('no')
                decision = choice(choice_list)
                if decision == 'no':
                    successes += 1
                    process_result(node_list)
                    active = False
                else:
                    continue