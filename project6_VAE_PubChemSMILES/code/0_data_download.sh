#!/bin/env bash

# Download canonical SMILES string file
wget -O /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/data/pubchem_simplified.smi.gz \
    https://bits.csb.pitt.edu/cobb2060/assign6/pubchem_simplified.smi.gz

# Download randomized
wget -O /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/data/pubchem_randomized.smi.gz \
    https://bits.csb.pitt.edu/cobb2060/assign6/pubchem_randomized.smi.gz

#### [NOTE] Those are gzip files (compressed, single file), so no need to untar -> can load directly using gzip.open in python ####

