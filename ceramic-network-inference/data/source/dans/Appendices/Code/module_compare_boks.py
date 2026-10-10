"""
Author: E.J. Kroon

v1 June 27, 2023

Please refer to Kroon (in prep.) when using this program.

Pathfinder is meant to compare, generate, and simulate ceramic
chaînes opératoires. 

This is a module of pathfinder. It compares two user-specified 
bodies of knowledge using the Wasserstein distance and 
produces a third set of randomly generated chaînes opératoires
as control group. See Kroon (in prep. Section 4.2) for a 
description of this algorithm.

The .count() function used below to compile link lists is
time-consuming and contributes to long processing times for
large datasets.

Note that the script repeats a block which converts a list
of paths into edge weights three times. This is inelegant,
but it ensures no mix-up occurs with the values.

The algorithm generates a new batch of random paths for
every new comparison, meaning minor (second decimal) 
differences in Wasserstein distances between A and C occur 
between iterations. Kroon (in prep. Section 10.2) circumvents 
this issue by using the same set of randomly generated paths 
(ceramics/sim_Random). An alternative option is a higher 
number of randomly generated paths in BoK C. However, this
solution also increases computational cost.

The script makes use of separate, open source libraries
in python: csv, scipy, collections, networkx, and random. 
See the documentation of these respective modules. The script 
is provided as is under a CC-BY 4.0 license.

"""

import csv
from collections import Counter
import module_random_path_generator as r_g
from scipy.stats import wasserstein_distance

def compare_boks(bok_1, bok_2):
    #File with all links in network
    links_file = 'ceramics/link_list.csv'
    #Data files
    background_file = bok_1
    paths_file = bok_2
    #File with randomly generated paths for BoK C
    random_file = 'ceramics/output_random_paths.csv'
    #Output file for edge weights
    output_file = 'ceramics/output_comparisons.csv'

    #Open and read the file with all links in the network
    with open(links_file, 'r', encoding='utf-8') as link_list:
        link_reader = csv.reader(link_list, delimiter=';')
        links = [l for l in link_reader][1:]

    #List all source-target pairs in network
    list_pairs = []
    for link in links:
        pair = [link[0], link[1]]
        list_pairs.append(pair)
        n_links = len(list_pairs)

    link_ids = list(range(n_links+1))
    del link_ids[0]

    #Generate 1,000 paths for body of knowledge C
    n_r_p = 1000
    r_g.random_path_generator(n_r_p)

    #Determine the number of paths in BoK A and B
    with open(background_file, 'r', encoding='utf-8') as paths_list:
        pathreader = csv.reader(paths_list, delimiter=';')
        next(pathreader)
        paths = list(pathreader)
        n_of_background_paths = len(paths)+1

    with open(paths_file, 'r', encoding='utf-8') as i_paths_list:
        i_pathreader = csv.reader(i_paths_list, delimiter=';')
        next(i_pathreader)
        i_paths = list(i_pathreader)
        n_of_input_paths = len(i_paths)+1

    #Open and read the file for BoK A
    with open(background_file, 'r', encoding='utf-8') as data_list:
        data_reader = csv.reader(data_list, delimiter=';')
        data_links = list(data_reader)
        #Initiate list for pairs in the input file
        list_prs = []
        #Split the path list into pairs of a source node and target node
        for path in data_links:
            for idx, node in enumerate(path):
                src = node
                #try-except to catch the last node in the list
                try:
                    tgt = path[(idx+1)]
                except IndexError:                        
                    break
                #Append pair to a list of node pairs in paths
                list_prs.append([src, tgt])
    #Initiate a list for edge weights
    background_weights = []
    #For every pair in the network, count identical node pairs in input file
    for pair in list_pairs:
        weight = list_prs.count(pair)
        if weight > 0:
            #Store as percentages of paths in BoK A
            background_weights.append(weight/n_of_background_paths*100)
        else: 
            background_weights.append(0)

    #Open and read the file for BoK_B
    with open(paths_file, 'r', encoding='utf-8') as data_list:
        data_reader = csv.reader(data_list, delimiter=';')
        data_links = list(data_reader)
        #Initiate list for pairs in the input file
        list_prs = []
        #Split the path list into pairs of a source node and target node
        for path in data_links:
            for idx, node in enumerate(path):
                src = node
                #try-except to catch the last node in the list
                try:
                    tgt = path[(idx+1)]
                except IndexError:                        
                    break
                #Append pair to a list of node pairs in paths
                list_prs.append([src, tgt])
    #Initiate a list for edge weights
    paths_weights = []
    #For every pair in the network, count identical node pairs in input file
    for pair in list_pairs:
        weight = list_prs.count(pair)
        if weight > 0:
            #Store as percentage of paths in BoK B
            paths_weights.append(weight/n_of_input_paths*100)
        else: 
            paths_weights.append(0)

    #Open and read the file for BoK C
    with open(random_file, 'r', encoding='utf-8') as data_list:
        data_reader = csv.reader(data_list, delimiter=';')
        data_links = list(data_reader)
        #Initiate list for pairs in the input file
        list_prs = []
        #Split the path lists into pairs of a source node and target node
        for path in data_links:
            for idx, node in enumerate(path):
                src = node
                #try-except to catch the last node in the list
                try:
                    tgt = path[(idx+1)]
                except IndexError:                        
                    break
                #Append pair to a list of node pairs in paths
                list_prs.append([src, tgt])
    #Initiate a list for edge weights
    random_weights = []
    #For every pair in the network, count identical node pairs in input file
    #Note the .count() option is time expensive and suboptimal, so look for alternative
    for pair in list_pairs:
        weight = list_prs.count(pair)
        if weight > 0:
            random_weights.append((weight/n_r_p)*100)
        else: 
            random_weights.append(0)

    #Write the edge weights to an output file for inspection
    with open(output_file, 'a', newline = '') as f:
        writer = csv.writer(f, delimiter = ';')
        writer.writerow(link_ids)
        writer.writerow(background_weights)
        writer.writerow(paths_weights)
        writer.writerow(random_weights)

    #Print results for user
    print(f'\nWasserstein distance from A to B is:')
    print(wasserstein_distance(background_weights,paths_weights))
    print(f'\nWasserstein distance from A to C is:')
    print(wasserstein_distance(background_weights,random_weights))