"""
Author: E.J. Kroon

v1 June 27, 2023

Please refer to Kroon (in prep.) when using this program.

Pathfinder is meant to compare, generate, and simulate ceramic
chaînes opératoires. This is the converter module. This module
translates node IDs to node Names and vice versa in data files
based on a dictionary file (ceramics/nodes_list.csv). 

The module scans the first node of the first path, and converts
all strs from names to IDs or from IDs to names. This module is
integrated into the other ones. If conversion is needed for
other purposes, it is best to copy this module and re-program it
as a standalone script to avoid issues cropping up elsewhere.

Conversion may fail if the first row of the input file is empty
and further tests are required to determine/solve the issue.

This module makes use of a separate, open source library in 
python: csv. See the documentation for this library. The script
is provided as is under a CC-BY 4.0 license.

"""

import csv

def converter_tool(i_file, o_file):
    #specify .csv file with data to be converted
    input_file = i_file
    #Specify .csv file to export converted data to
    output_file = o_file
    #Specify .csv file with node ID and node names
    dictionary_file = 'ceramics/nodes_list.csv'

    #Open and read conversion file with the node IDs and node names
    with open(dictionary_file, 'r', encoding='utf-8') as dict_list:
        dict_reader = csv.reader(dict_list, delimiter=';')
        nodes = [n for n in dict_reader][1:]

    #Initiate dictionary for conversion
    node_names_dict ={}
    for node in nodes:
        node_names_dict[node[0]] = node[1]

    #Function for retrieving node name of a node ID from dictionary
    def get_key(technique):
        for key, value in node_names_dict.items():
            if technique == value:
                return key
    #Function for retrieving node ID of node namefrom dictionary
    def get_value(technique):
        for key, value in node_names_dict.items():
            if technique == key:
                return value

    #Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        input_reader = csv.reader(f, delimiter=';')
        for row in input_reader:
            #Omit empty cells
            path = list(filter(None,row[0:]))
            #If first entry is a node name, translate all values to node IDs
            if path[0] == 'Wet clay':
                output_path = []
                for technique in path:
                    output_path.append(get_key(technique))
                #Write list to output file   
                with open(output_file, 'a', newline="") as f:
                    writer = csv.writer(f, delimiter = ';')
                    writer.writerow(output_path)

            #Vice versa if the first entry is a node ID
            elif path[0] == 'F':
                output_path = []
                for technique in path:
                    output_path.append(get_value(technique))
                #Write lists to output file
                with open(output_file, 'a', newline="") as f:
                    writer = csv.writer(f, delimiter = ';')
                    writer.writerow(output_path)