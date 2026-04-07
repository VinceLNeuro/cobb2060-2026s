#!/usr/bin/env python3
"""
Goal: 
    Preprocess the SMILES strings into indices based on pre-set dictionary -> save as 'npy' file (shorter set (with smaller molecules) for speed up and EDA, and full set for final training)

Rationale: 
    When creating the `Dataset` class (for later `DataLoader` input), the parsing and index conversion is the same across the training sessions. 
    Since we will do multiple training sessions (in hyperparam search), preprocessing can speed up greatly. 

Usage: 
    # full set
    python 1_preprocess.py pubchem_randomized.smi.gz pubchem_randomized.npy
    
    # shorter and smaller molecules set
    python 1_preprocess.py pubchem_randomized.smi.gz pubchem_randomized_small.npy --max_len 50 --max_mols 5000000

    ## The output is a uint8 numpy array of shape (N, max_length) where each row is a zero-padded (same as EOS token padding at the end), index-encoded SMILES string.
"""

import gzip
import argparse
import numpy as np

# From the provided pre-set dictionary
CHARTOINDEX = {
    '$': 0,'^': 1, 'C': 2, '(': 3,
    '=': 4, 'O': 5, ')': 6, '[': 7, '-': 8, ']': 9,
    'N': 10, '+': 11, '1': 12, 'P': 13, '2': 14,'3': 15,
    '4': 16, 'S': 17, '#': 18, '5': 19,'6': 20, '7': 21,
    'H': 22, 'I': 23, 'B': 24, 'F': 25, '8': 26, '9': 27
}


def preprocess(in_path, out_path, max_len = 150, max_mols = None):
    
    print(f"Reading {in_path} ...")

    # ======== count valid lines (or lines meeting requirement) ========
    n = 0
    with gzip.open(in_path, 'rt') as f:
        for line in f:
            smi = line.rstrip()
            if len(smi) == 0 or len(smi) >= max_len: #customizable
                continue
            if any(c not in CHARTOINDEX for c in smi):
                continue
            n += 1
            if max_mols and n >= max_mols:
                break
    print(f"  Valid strings/molecules: {n:,}")


    # ======== Preset & Fill the output array ========
    arr = np.zeros((n, max_len), dtype=np.uint8) #unsigned integer (since only 28 indices)
    
    idx = 0
    with gzip.open(in_path, 'rt') as f:
        for line in f:
            smi = line.rstrip()
            if len(smi) == 0 or len(smi) >= max_len:
                continue
            if any(c not in CHARTOINDEX for c in smi):
                continue
            
            # token encoding
            encoded = [CHARTOINDEX[c] for c in smi] + [CHARTOINDEX['$']]
            # fill
            arr[idx, :len(encoded)] = encoded

            # control max strings
            idx += 1
            if idx >= n:
                break
            if idx % 1_000_000 == 0:
                print(f"  Processed {idx:,} / {n:,}")

    np.save(out_path, arr)
    print(f"Saved {arr.shape} array to {out_path}")
    return arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input .smi.gz file')
    parser.add_argument('output', help='Output .npy file')
    parser.add_argument('--max_len', type=int, default=150)
    parser.add_argument('--max_mols', type=int, default=None)
    args = parser.parse_args()
    
    preprocess(args.input, args.output, args.max_len, args.max_mols)

