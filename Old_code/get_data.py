# Let's examine the data structure properly by parsing the CSV files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

# Function to parse the triplet data correctly
def parse_triplet_file(filename):
    """Parse the triplet CSV file and extract individual triplets"""
    triplets_data = []
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Extract triplets using regex
            triplet_matches = re.findall(r"\('([^']+)', '([^']+)', '([^']+)'\)", line.strip())
            if triplet_matches:
                # Convert triplets to formatted strings
                doc_triplets = [f"({t[0]}, {t[1]}, {t[2]})" for t in triplet_matches]
                triplets_data.append({
                    'line_number': line_num,
                    'triplets': doc_triplets,
                    'num_triplets': len(doc_triplets)
                })
    
    return triplets_data

# Parse all culture files (in Old_code/data, wherever this is run from)
import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
cw_data = parse_triplet_file(os.path.join(DATA_DIR, 'clustered_paths_CW.csv'))
mi_data = parse_triplet_file(os.path.join(DATA_DIR, 'clustered_paths_MI.csv'))
efg_data = parse_triplet_file(os.path.join(DATA_DIR, 'clustered_paths_EFG.csv'))
lfg_data = parse_triplet_file(os.path.join(DATA_DIR, 'clustered_paths_LFG.csv'))

print(f"CW Culture: {len(cw_data)} documents")
print(f"MI Culture: {len(mi_data)} documents")
print(f"EFG Culture: {len(efg_data)} documents")
print(f"LFG Culture: {len(lfg_data)} documents")

# Show sample data
print("\nSample CW data (first 3 documents):")
for i in range(min(3, len(cw_data))):
    doc = cw_data[i]
    print(f"Document {doc['line_number']}: {doc['num_triplets']} triplets")
    print(f"  Triplets: {doc['triplets'][:3]}...")  # Show first 3 triplets