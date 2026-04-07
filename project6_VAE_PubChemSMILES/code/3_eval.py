#!/usr/bin/env python3

"""
(run `python 2_save_canonicalSMILES.py` first)

A script evaluating the generative capacity of a SMILES VAE 

  # Using a model from the local filesystem:
  python eval.py --model=model.pth TEST

"""

import argparse
import numpy as np
import sys, os, time, pickle
from sklearn import metrics
import torch
from rdkit.Chem import AllChem as Chem

def parse_args():
  """Parses arguments specified on the command-line."""

  argparser = argparse.ArgumentParser('Evaluate generative capacity of VAE decoder.')

  argparser.add_argument('--model', default='model.pth', help='PyTorch model file')
  argparser.add_argument('--train_pickle', default=None, help='Pickle of a set of all training set smiles')
  argparser.add_argument('--evals',default=1000, type=int, help='Number of evaluations')
  return argparser.parse_args()


class Lang:
    def __init__(self):
        self.chartoindex = {'SOS': 1, 'EOS': 0, 'C': 2, '(': 3,
                '=': 4, 'O': 5, ')': 6, '[': 7, '-': 8, ']': 9,
                'N': 10, '+': 11, '1': 12, 'P': 13, '2': 14,'3': 15,
                '4': 16, 'S': 17, '#': 18, '5': 19,'6': 20, '7': 21,
                'H': 22, 'I': 23, 'B': 24, 'F': 25, '8': 26, '9': 27
                } 
        self.indextochar = {1: 'SOS', 0: 'EOS', 2: 'C', 3: '(',
                4: '=', 5: 'O', 6: ')', 7: '[', 8: '-', 9: ']',
                10: 'N', 11: '+', 12: '1', 13: 'P', 14: '2', 15: '3',
                16: '4', 17: 'S', 18: '#', 19: '5', 20: '6', 21: '7',
                22: 'H', 23: 'I', 24: 'B', 25: 'F', 26: '8', 27: '9'
                }
        self.nchars = 28
        
    def indexesFromSMILES(self, smiles_str):
        index_list = [self.chartoindex[char] for char in smiles_str]
        index_list.append(self.chartoindex["EOS"])
        return np.array(index_list, dtype=np.uint8)
        
    def indexToSmiles(self,indices):
        '''convert list of indices into a smiles string'''
        #todo: be nice and ignore after first EOS before this check
        if 1 in indices:
            print("SOS character encountered in smile")
            return 'x'
        smiles_str = ''.join(list(map(lambda x: self.indextochar[int(x)] if x != 0.0 else 'E',indices)))
        return smiles_str.split('E')[0] #Only want values before output 'EOS' token        


def smilesToStatistics(list_of_smiles,trainsmi):
    '''Return number valid smiles, number of unique molecules, and average number of rings'''
    count_molecules = 0
    cannonical_smiles = set()
    ringcnt = 0
    novelcnt = 0
    for smiles in list_of_smiles:
        if smiles == '':
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                count_molecules += 1
                can = Chem.MolToSmiles(mol)
                if can not in cannonical_smiles:
                    #print(can)
                    cannonical_smiles.add(can)
                    r = mol.GetRingInfo()
                    ringcnt += r.NumRings()
                    if can not in trainsmi:
                        novelcnt += 1
                    
        except:
            continue
    N = len(cannonical_smiles)
    ringave = 0 if N == 0 else ringcnt/N
    return count_molecules, N, novelcnt, ringave

         

if __name__ == '__main__':
    LATENT_DIM = 1024
    args = parse_args()
    eval_quant = args.evals

    lang = Lang()
    print('about to load')
    model = torch.jit.load(args.model)
    print('loaded',flush=True)
    torch.manual_seed(0)
    trainsmi = set()
    if args.train_pickle:
        trainsmi = pickle.load(open(args.train_pickle,'rb'))
    print("read pickle",flush=True)
    start_time = time.time() 

    generated_smiles = []
    # with torch.no_grad():
    print('Creating z...', flush=True)
    z = torch.zeros((1,LATENT_DIM),device='cuda')
    print('z created on cuda -> Running model...', flush=True)
    _, output_smile = model(z)
    torch.cuda.synchronize() #show true GPU comp. time
    print(f'Done in {time.time()-start_time:.1f}s', flush=True)
    print(f'Output shape: {output_smile.shape}', flush=True)

    print('Converting zsmile...', flush=True)
    zsmile = lang.indexToSmiles(output_smile[0])
    print(f'zsmile: {zsmile}', flush=True)

    print('Starting loop...', flush=True)
    for i in range(eval_quant):
        print(f'Iter {i}: creating z...', flush=True)
        z_1 = torch.normal(0, 1, size=(1, LATENT_DIM),device='cuda')
        print(f'Iter {i}: running model...', flush=True)
        _, output_smile = model(z_1)
        torch.cuda.synchronize()
        print(f'Iter {i}: converting...', flush=True)
        generated_smiles.append(lang.indexToSmiles(output_smile[0]))
        print(i,flush=True)
        
    duration = (time.time()-start_time)
    valid_smiles, unique_mols, novelcnt, ringave = smilesToStatistics(generated_smiles,trainsmi)


    print("GenerateTime",duration)
    print("UniqueSmiles", len(set(generated_smiles)))
    print("ValidSmiles", valid_smiles)
    print("UniqueAndValidMols", unique_mols)
    print("NovelMols", novelcnt)
    print("AverageRings", ringave)
    print("SMILE",zsmile)

