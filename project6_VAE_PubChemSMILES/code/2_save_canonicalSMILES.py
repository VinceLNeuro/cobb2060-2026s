training_npy_path='/net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/data/pubchem_randomized_len50_Mol50k.npy'
output_pkl_path  ='/net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/run0_noSweep/train_smiles.pkl'

import numpy as np, pickle
from rdkit.Chem import AllChem as Chem

# Load your training .npy
examples = np.load(training_npy_path)

class Lang:
    '''Predefined mapping from characters to indices for our
    reduced alphabet of SMILES with methods for converting.
    You must use this mapping.'''
    
    def __init__(self):
        # $ is the end of sequence token (EOS)
        # ^ is the start of sequence token, which should never be generated
        self.chartoindex = {'$': 0,'^': 1, 'C': 2, '(': 3,
                '=': 4, 'O': 5, ')': 6, '[': 7, '-': 8, ']': 9,
                'N': 10, '+': 11, '1': 12, 'P': 13, '2': 14,'3': 15,
                '4': 16, 'S': 17, '#': 18, '5': 19,'6': 20, '7': 21,
                'H': 22, 'I': 23, 'B': 24, 'F': 25, '8': 26, '9': 27
                } 
        self.indextochar = {0: '$', 1: '^', 2: 'C', 3: '(',
                4: '=', 5: 'O', 6: ')', 7: '[', 8: '-', 9: ']',
                10: 'N', 11: '+', 12: '1', 13: 'P', 14: '2', 15: '3',
                16: '4', 17: 'S', 18: '#', 19: '5', 20: '6', 21: '7',
                22: 'H', 23: 'I', 24: 'B', 25: 'F', 26: '8', 27: '9'
                }
        self.nchars = 28 #total unique char-set in the data is 26   +   2 special tokens
        
    def indexesFromSMILES(self, smiles_str):
        '''convert smiles string into numpy array of integers'''
        index_list = [self.chartoindex[char] for char in smiles_str]
        index_list.append(self.chartoindex["$"])
        return np.array(index_list, dtype=np.uint8)
        
    def indexToSmiles(self,indices):
        '''convert list of indices into a smiles string'''
        smiles_str = ''.join(list(map(lambda x: self.indextochar[int(x)] if x != 0.0 else 'E',indices)))
        return smiles_str.split('E')[0] #Only want values before output $ end of sequence token

EOS_IDX = 0
SOS_IDX = 1
lang = Lang()  # use the Lang class from eval.py

train_smiles = set()
for row in examples:
    smi = lang.indexToSmiles(row)
    if smi:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_smiles.add(Chem.MolToSmiles(mol))  # canonical

with open(output_pkl_path, 'wb') as f:
    pickle.dump(train_smiles, f)
print(f"Saved {len(train_smiles):,} molecules")

