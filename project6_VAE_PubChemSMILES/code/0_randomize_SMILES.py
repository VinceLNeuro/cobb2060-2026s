#!/usr/bin/env python3
'''
Randomize smiles

Usage: 
    python3 0_randomize_SMILES.py pubchem_simplified.smi.gz > randomized.smi
'''

from openbabel import pybel #cheminformatics toolkit
import sys, gzip

with gzip.open(sys.argv[1],'rt') as f: #read + text mode
    for line in f:
        line = line.rstrip()
        m = pybel.readstring('smi',line)
        # opt: kekulized and non-canonical randomization
        random = m.write('smi',opt={'k':None,'C':True}).rstrip()
        if len(random) < 150 and '%' not in random: #randomizing can result in longer strings or have characters not in our reduced set
            print(random)
        elif len(line) < 150: # or keep if the canonical form has <150char -> so output canonical which should be right size
            print(line)

            