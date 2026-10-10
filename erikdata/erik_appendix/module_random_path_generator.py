"""
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

import networkx as nx
import csv
from random import choice

def random_path_generator(number):

    file_nodes = 'ceramics/nodes_list.csv'
    file_links = 'ceramics/link_list.csv'
    file_output = 'ceramics/output_random_paths.csv'

    #Plot the graph from the files
    #Open list of nodes and skip headers
    with open(file_nodes, 'r', encoding='utf-8') as nodes_list:
        nodereader = csv.reader(nodes_list, delimiter=';')
        nodes = [n for n in nodereader][1:]

    #Attribute node IDs to nodes
    node_ids = [n[0] for n in nodes]

    #Open list of links (node_IDs for target, source) with headers
    with open(file_links, 'r', encoding='utf-8') as links_list:
        linkreader = csv.reader(links_list, delimiter=';')
        links = [tuple(l) for l in linkreader][1:]

    #Create a directed network object and add the nodes and links
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    G.add_weighted_edges_from(links)

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


    successes = 1

    #Pathfinder loop
    while successes <= number:
        #Define empty node list and starting node (i.e. wet clay)
        node_list = ['W', '1',]

        #Set flag for pathfinder loop
        active = True

        while active:

            #Find nodes adjacent to last node on list
            potential_targets = list(G.neighbors(node_list[-1]))
            #Determine hygrometric tolerance for each target
            targets = []
            for potential_target in potential_targets:
                tol = G.nodes[potential_target]['tolerance']
                #If path state != node tolerance, remove target from list
                if not node_list[0] in tol:
                    potential_targets.remove(potential_target)
                #If node tolerance = path state, append node as target    
                if node_list[0] in tol:
                    targets.append(potential_target)

            #Randomly choose node from target list, except when list is empty
            try:
                new_node = choice(targets)
            except IndexError:
                #If no targets are listed: end cycle unless path state is 'F'
                if node_list[0] == 'F':
                    #If no targets + path state is 'F', dump list in output file
                    with open(file_output, 'a', newline = '') as f:
                        writer = csv.writer(f, delimiter = ';')
                        writer.writerow(node_list)
                        successes += 1
                    active = False
                else:
                    #If no listed targets + path state is not 'F': end cycle
                    active = False
            # If the chosen target node is already in node list
            if new_node in node_list:
                if node_list[0] == 'F':
                #And path state is 'F', dump list in output file and end cycle
                    with open(file_output, 'a', newline = '') as f:
                        writer = csv.writer(f, delimiter = ';')
                        writer.writerow(node_list)
                        successes += 1
                    active = False
                else: 
                #In all other path states: a path loop ends the cycle
                    active = False
            else:    
                #Append selected node to node list
                if node_list[0] == 'F':
                    #Unless path state is F, then a diceroll determines whether:
                    dice = choice([0, 1])
                    if dice == 0:
                        #Path ends and is dumped in output_file without the target
                        with open(file_output, 'a', newline = '') as f:
                            writer = csv.writer(f, delimiter = ';')
                            writer.writerow(node_list)
                            successes += 1
                        #Which ends the cycle    
                        active = False
                    elif dice == 1:
                        #Or the target is appended and the cycle continues
                        node_list.append(new_node)
                else:
                    #In path states that are not F, append target to node_list
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