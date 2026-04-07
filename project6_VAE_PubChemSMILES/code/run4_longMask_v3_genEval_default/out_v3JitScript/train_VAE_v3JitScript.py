#!/usr/bin/env python3
"""
VAE for SMILES string generation

VAE Architecture:
- Encoder: RNN (GRU or LSTM, optionally bidirectional)
- Reparameterization: z = mu + eps * std
- Decoder: Undirectional RNN (GRU or LSTM); teacher forcing during training/val -> turned off in trace

Training Notes:
- Sigmoid KL annealing schedule
- ReduceLROnPlateau scheduler
- Gradient clipping to prevent gradient explosion

Updates:
- Added random word-dropout (or masking) - so that the decoder will use z more than relying on teacher-forcing
- Modify the max KL weight to make it LESS focus on matching N(0,1)

- [BUG] Fixed mixed sequence of args in VAE/self.decoder
- [BUG] Fixed generatng SOS in inference mode
- [BUG] Fixed: create custom mask, so that first EOS is included in the CE loss calculation
- [!NOTE]Include a generation evaluation chunk -> update checkpoint selection rule
- [!NOTE]Change the LR scheduler update scheme to be based on `val/recon_loss`

Usage:
    python train_VAE.py -T /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/data/pubchem_randomized_len50_Mol50k.npy \
                    --out /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/run0_noSweep/vae_decoder_gen.pth \
                    --max_length 50 \
                    --bidirectional \
                    --epochs 10 \
                    --log_every 20 --val_every 92
"""

import gzip, time, pickle, math
import numpy as np
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.utils.data import DataLoader, Subset

from rdkit.Chem import AllChem as Chem
from typing import Tuple, List


#==================== Define the mapping/encoding class ====================

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


# ==================== Define the Dataset class ====================

class SmilesDataset(torch.utils.data.Dataset):
    '''Dataset that reads in a gzipped smiles file and converts to a
    numpy array representation.  Note we encountered memory usage issues
    when using variable sequence length batches and so use a fixed size.
    There are likely more memory efficient ways to store this data.
    Feel free to experiment with nested tensors (https://pytorch.org/docs/stable/nested.html),
    but you may assume strings will never be more than 150 characters.'''
    
    def __init__(self, data_path, max_length=150):
        self.max_length = max_length
        self.language = Lang()
        
        #TODO - for faster training you will want to preprocess
        #the training set and read in this processed file instead
        #for faster initialization
        if data_path.endswith('.npy'):
            self.examples = np.load(data_path)
            print(f"Loaded preprocessed array: {self.examples.shape}")

        else: # If not npy, process as gz files -> then convert to np array
            print("Reading raw .smi.gz (slow --> consider running 1_preprocess.py first)")
            
            # Same as the preprocess script
            rows = []
            with gzip.open(data_path, 'rt') as f:
                for line in f:
                    smi = line.rstrip()
                    # filter step
                    if len(smi) == 0 or len(smi) >= max_length:
                        continue
                    if any(c not in self.language.chartoindex for c in smi):
                        continue
                    enc = self.language.indexesFromSMILES(smi)
                    row = np.zeros(max_length, dtype=np.uint8)
                    # fill in (also, use 0 as padding)
                    row[:len(enc)] = enc
                    rows.append(row)
            self.examples = np.stack(rows)
            print(f"Loaded {len(self.examples):,} molecules")

            # with gzip.open(data_path,'rt') as f:
            #     N = sum(1 for line in f)
            
            # self.examples = np.zeros((N,max_length), dtype=np.uint8)
            
            # with gzip.open(data_path,'rt') as f:
            #     for i,line in enumerate(f):
            #         example = line.rstrip()
            #         ex = self.language.indexesFromSMILES(example)
            #         self.examples[i][:len(ex)] = ex
                
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)
    
    def getIndexToChar(self):
        return self.language.indextochar


# ==================== Helper function for train validation split ====================

def make_train_val_split(dataset: SmilesDataset, val_frac = 0.05, seed = 42):
    """
    Shuffle indices with a fixed seed so the split is reproducible across runs and independent of the order molecules appear in the .npy file.
    - default: 95/5 train/val split on indices
    - Returns: (train_subset_data, val_subset_data)
    """
    n = len(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n) #create randomized indices
    n_val = max(1, int(n * val_frac))

    # extract indices for train and val
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    
    print(f"Split: {len(train_idx):,} train | {len(val_idx):,} val  (val_frac={val_frac}, seed={seed})")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# *** ==================== Define the VAE model architecture ==================== ***

class Encoder(nn.Module):
    """
    RNN encoder (GRU or LSTM, optionally bidirectional)
    - Input  : dim=(batch, seq_len); padded
    - Process: SMILES string idx -> embeddings -> RNN encoder -> final hidden state -> two linear projections -> [mean, logv]
    - Output : mean dim=(batch, LATENT_DIM), logv dim=(batch, LATENT_DIM)
    """
    
    # Fixed hyperparam
    VOCAB_SIZE = 28    #fixed
    LATENT_DIM = 1024  #fixed - for nn.Linear

    def __init__(self, 
                 embedding_dim, # for nn.Embedding
                 hidden_size, num_layers, bidirectional = True, dropout = 0.1, # for nn.GRU
                 rnn_cell_type = 'GRU'): 
        
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cell_type = rnn_cell_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Define embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = self.VOCAB_SIZE, 
            embedding_dim = embedding_dim, 
            padding_idx = EOS_IDX
        )

        # Define nn.GRU/nn.LSTM
        rnn_cls = getattr(nn, rnn_cell_type) # -> nn.GRU or nn.LSTM (share the same input parameters)
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # "the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)"
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Define the linear fully-connected layer
        proj_linear = self.num_directions * hidden_size #bidir -> hidden states are connected and doubled
        self.fc_mean = nn.Linear(in_features = proj_linear, out_features = self.LATENT_DIM, bias=True)
        self.fc_logv = nn.Linear(in_features = proj_linear, out_features = self.LATENT_DIM, bias=True)


    def forward(self,input_seq):

        embedded = self.embedding(input_seq)
        output   = self.rnn(embedded)

        # Handle rnn type
        if self.rnn_cell_type == 'LSTM':
            _, (hidden, _) = output    # Outputs: output, (h_n, c_n)
                                       #    only hidden state is the final h, cell state is raw long-term memory
        else:
            _, hidden = output         # Outputs: output, h_n
                                       #                  h_n dim = (num_directions*num_layers, batch, hidden) - not affected by batch_first
        # Handle bidirectional hidden state extraction
        """
        https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.LSTM.html
            When bidirectional=True, h_n will contain a concatenation of the final forward and reverse hidden states, respectively.
        """
        if self.bidirectional:
            h = torch.cat([hidden[-2], hidden[-1]], dim=1) # concat last layer forward and reverse hidden states: (batch, 2*hidden)
        else:
            h = hidden[-1]                                 # (batch, hidden)

        # Calculate fc layer output
        mean = self.fc_mean(h)
        logv = self.fc_logv(h)

        # return a vector of means and log of variance
        # (doesn't have to be the log, but this interpretation can be convenient)
        return mean, logv


class Decoder(nn.Module):
    """
    Undirectional RNN decoder (GRU or LSTM) - always autoregressive generation (only left-to-right)
    """
    VOCAB_SIZE = 28
    LATENT_DIM = 1024

    def __init__(self, 
                 embedding_dim, # for nn.Embedding for x[t] input tokens
                 max_length,
                 hidden_size, num_layers, dropout = 0.1, rnn_cell_type = 'GRU'):  # for nn.GRU
        super(Decoder, self).__init__()
        # Decoder Setup
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cell_type = rnn_cell_type
        self.max_length = max_length

        self.embedding = nn.Embedding(self.VOCAB_SIZE, embedding_dim, padding_idx=EOS_IDX) # used in teacher forcing & input token
        rnn_cls = getattr(nn, rnn_cell_type)
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Only 1 fc layer due to 1 single output logit
        self.fc_out = nn.Linear(hidden_size, self.VOCAB_SIZE)


        # *** Define z-projection-to-hidden layer: project z to initial hidden state for each layer
        """
        Demo from claude: stacked GRU (num_layers=3)
                Layer 2:  h2_0 → h2_1 → h2_2 → output
                            ↑       ↑       ↑
                Layer 1:  h1_0 → h1_1 → h1_2        Combine through gating mechanism
                            ↑       ↑       ↑
                Layer 0:  h0_0 → h0_1 → h0_2
                            ↑       ↑       ↑
                Input:    tok_0   tok_1   tok_2
        """
        self.z_to_hidden = nn.Linear(self.LATENT_DIM, num_layers * hidden_size)
        
        # Also take care of the cell-state init hidden state
        if rnn_cell_type == 'LSTM':
            self.z_to_cell = nn.Linear(self.LATENT_DIM, num_layers * hidden_size)


        # # [NOTE] Pre-build SOS suppression bias as a persistent buffer
        # # [NOTE-BUG] Should remove SOS in generation
        # sos_bias = torch.zeros(self.VOCAB_SIZE)
        # sos_bias[SOS_IDX] = -1e9
        # self.register_buffer('sos_bias', sos_bias)


    def _init_hidden(self, z):
        """
        Project reparameterized z to the decoder hidden state (also cell state for LSTM)
            z (batch, LATENT_DIM) (same shape as mu and logvar) -> hidden state for the RNN.
        """
        batch_size = z.size(0)

        h = torch.tanh(self.z_to_hidden(z)) # sigma(h) for GRU and LSTM are all tanh, so also use tanh for init
        
        # RESHAPE the hidden layer tensor
        #   [NOTE] we use batch_first=True -> so h0 dim = (num_layers, N, Hout​)
        #   - view: interprets the flat memory (unroll by batch -> layer -> hidden)
        #   - permute: require this to switch to the right dim
        h = h.view(batch_size, self.num_layers, self.hidden_size)
        h = h.permute(1, 0, 2).contiguous()               # (num_layers, batch, hidden)

        if self.rnn_cell_type == 'LSTM':
            c = torch.tanh(self.z_to_cell(z))
            c = c.view(batch_size, self.num_layers, self.hidden_size)
            c = c.permute(1, 0, 2).contiguous()
            return (h, c)
 
        return h 

    def forward(self, z, teacherForcing_actual_input=None):

        #generated_sequence must be BATCH x SEQLENGTH, have type long, and contain
        #the index form of the generated sequences (smiles strings can be generated
        #by passing rows to Lang.indexToSmiles)
        # IMPORTANT:
        #  Because we are saving a trace, you must loop over the max SEQLENGTH and not
        #  terminate sequence generation early when actual_input=None

        batch_size = z.size(0)
        hidden = self._init_hidden(z)
 
        if teacherForcing_actual_input is not None:
            # ---- Teacher-forcing path (training & val) ----
            sos = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=z.device) #vector of 1 -> sos token
            ## concat the first sos token, so that the TF input would be one-step shift (one token earlier to the target seq)
            decoder_tf_input = torch.cat([sos, teacherForcing_actual_input[:, :-1]], dim=1) # dim =1 over the col
            embedded = self.embedding(decoder_tf_input)

            """
            h_0: tensor of shape (D*num_layers,Hout) or (D*num_layers,N,Hout)
                 containing the initial hidden state for the input sequence. 
                 ** Defaults to zeros if not provided. **
            """
            output, _ = self.rnn(embedded, hidden)
            logits = self.fc_out(output)
            generated = logits.argmax(dim=-1) #-1 = last dim of the matrix (should be seq_length)
            return logits, generated
 
        else:
            # ---- Autoregressive inference path (test phase) ----
            token = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=z.device)
            all_logits, all_tokens = [], []

            # Loop over the SEQLENGTH for generation
            for _ in range(self.max_length):
                embedded = self.embedding(token)
                output, hidden = self.rnn(embedded, hidden)
                logit = self.fc_out(output.squeeze(1)) #squeeze(1): removes dimension 1 if its size is 1 (which is seq_length=1 token) ---> (batch,vocab=28)
                all_logits.append(logit.unsqueeze(1))  #unsqueeze because we can later use `torch.cat` for merging at dim 1

                # We want to use multinomial sampling (based on probability of tokens) instead of deterministic 
                #   [NOTE] the whole point of a generative model is to sample diverse valid molecules from the latent space
                probs = F.softmax(logit, dim=-1)
                token = torch.multinomial(probs, num_samples=1) #(batch,1)
                all_tokens.append(token)
            
            return torch.cat(all_logits, dim=1), torch.cat(all_tokens, dim=1)

        #return decoder_output, generated_sequence 


class ExportDecoder(nn.Module):
    """
    Lightweight wrapper for scripting — only the autoregressive inference path.
    No teacher forcing, no LSTM branching, no Optional args.
    """
    def __init__(self, decoder: Decoder):
        super().__init__()
        self.embedding = decoder.embedding
        self.rnn = decoder.rnn
        self.fc_out = decoder.fc_out
        self.z_to_hidden = decoder.z_to_hidden
        self.hidden_size = decoder.hidden_size
        self.num_layers = decoder.num_layers
        self.max_length = decoder.max_length

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = z.size(0)

        h = torch.tanh(self.z_to_hidden(z))
        h = h.view(batch_size, self.num_layers, self.hidden_size)
        hidden = h.permute(1, 0, 2).contiguous()

        token = torch.full((batch_size, 1), 1, dtype=torch.long, device=z.device)  # SOS_IDX=1
        all_logits: List[torch.Tensor] = []
        all_tokens: List[torch.Tensor] = []

        for _ in range(self.max_length):
            embedded = self.embedding(token)
            output, hidden = self.rnn(embedded, hidden)
            logit = self.fc_out(output.squeeze(1))
            all_logits.append(logit.unsqueeze(1))

            probs = torch.softmax(logit, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            all_tokens.append(token)

        return torch.cat(all_logits, dim=1), torch.cat(all_tokens, dim=1)



class VAE(nn.Module):
    """
    Full VAE assembly = Encoder + reparameterisation + Decoder
    """
    VOCAB_SIZE = 28
    LATENT_DIM = 1024

    def __init__(self, 
                 embedding_dim = 128, #default 128 should be enough to differentiate for a small vocaburary set (28)
                 hidden_size = 512, num_layers = 2, max_length = 150, dropout = 0.1, bidirectional = True, rnn_cell_type = 'GRU',
                 word_dropout_rate=0.0, # for maksing in teacher-forcing
                 ): #all sorts of hyper parameters should be passed here
        super(VAE, self).__init__()        
       
        self.encoder = Encoder(embedding_dim, hidden_size, num_layers, bidirectional, dropout, rnn_cell_type)
        # [NOTE-BUG] fixed the wrong seq of args!
        self.decoder = Decoder(embedding_dim, max_length, hidden_size, num_layers, dropout, rnn_cell_type)
        self.word_dropout_rate = word_dropout_rate


    def reparameterization(self, mu, logvar):

        if self.training: #model.train() -> self.training = True
            std = torch.exp(0.5 * logvar)           #σ  = sqrt(exp(logvar))
            return mu + torch.randn_like(std) * std #error = random extraction from N(0,1)
        
        return mu #deterministic by using only the single point mean
    

    def forward(self, input_seq):

        mean, logv = self.encoder(input_seq)
        # calculate z from mean and logv
        z = self.reparameterization(mean, logv)

        # Word dropout: randomly mask teacher-forcing tokens to force decoder to use z
        #   - Masked positions become 0 (EOS/padding), so the decoder can't rely on them
        if self.training and self.word_dropout_rate > 0:
            keep_prob = 1.0 - self.word_dropout_rate
            mask = torch.bernoulli(
                torch.full_like(input_seq.float(), keep_prob) #a keep_prob matrix for bernoulli distirbution sampling (1/0)
            ).long()
            decoder_input = input_seq * mask
        else:
            decoder_input = input_seq

        decoder_output_logits, generated_tok = self.decoder(z, teacherForcing_actual_input=decoder_input) #masked
        
        return decoder_output_logits, generated_tok, (mean, logv, z)
    

# ==================== Define the loss function ====================

def vae_loss(logits, targets,  # target = batch
             mu, logvar,
             kl_weight = 1.0): # determine if the loss focus more on kl

    batch, seq_len, vocab = logits.shape #decoder_output_logits

    #### Reconstruction - token ####
    # [BUG] Cannot ignore EOS as it is the padding -> the decoder will be hard to end!
    #       Need a customized mask
    token_ce = F.cross_entropy( #shape: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
        logits.reshape(batch * seq_len, vocab),
        targets.reshape(batch * seq_len),
        # ignore_index=EOS_IDX,
        reduction='none',
    ).reshape(batch, seq_len)
    
    # keep positions up to and including first EOS; drop EOS-padding after that
    is_eos = (targets == EOS_IDX)               #marks EOS tokens
    eos_cum = torch.cumsum(is_eos.int(), dim=1) #running count to EOS per sequence
    valid_mask = (eos_cum == 0) | ((eos_cum == 1) & is_eos) # keep the first eos (cumsum=1)

    recon_loss = (token_ce * valid_mask).sum() / valid_mask.sum().clamp_min(1) #recalculate CE loss based on masked


    #### Regularization ####
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar), dim=1), #(batch, latent_dim) -> sum over latent, per sample kl
        dim=0
    ) #average over samples
    
    loss = recon_loss + kl_weight*kl_loss

    return loss, recon_loss, kl_loss


# ==================== KL sigmoid annealing schedule ====================

def kl_sigmoid_weight_schedule(step,        #global batch step
                               t0_steps,    #step number at which the KL weight reaches 50% of ramping up (determining sigmoid shape)
                               k,           #steepness of sigmoid
                               kl_max_weight=1.0, #the KL weight can never reach above this value
                            ) -> float:
    """
    Reference: https://arxiv.org/pdf/1511.06349.pdf
    - 'We add a variable weight to the kl term in the cost function at training time'
    - 'At the start of training, we set that weight to zero'
    - 'As training progressed, we gradually increase thsi weight, forcing the model to smooth out its encodings and pack them into the prior' (until reaches 1)
    - 'The rate of this increase is tuned as a hyperparameter'

    Sigmoid ramp from 0 to `kl_max_weight` over steps, so the model first learns to reconstruct before the KL term is fully applied.
    - [NOTE] We can estimate t0_steps and k from data itself
    """
    kl_weight = kl_max_weight / (1.0 + math.exp(-k * (step - t0_steps)))
    return kl_weight


# ==================== Validation loop ====================

@torch.no_grad()
def run_validation(model: VAE, val_loader: DataLoader,
                   device: torch.device, 
                   kl_weight=1.0,
                   val_batches: int = 200     #if 50k small set -> only 10 val batches; if 5M large set -> 975 val batches (can use this to increase speed)
                   ) -> dict:
    model.eval()

    total_overall_loss = total_recon = total_kl = 0.0
    n_batches = 0
    for batch in val_loader:
        batch = batch.to(device, non_blocking=True) #non-blocking between CPU & GPU, increase speed  
        # forward pass
        logits, _, (mu, logvar, _) = model(batch)
        # calculate validation loss
        overall_loss, recon_loss, kl_loss = vae_loss(logits, batch, mu, logvar, kl_weight=kl_weight)
        # logging 
        total_overall_loss += overall_loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1
        if n_batches >= val_batches:
            break

    model.train()

    return { #per-batch averaged
        'val/overall_loss':  total_overall_loss / n_batches,
        'val/recon_loss': total_recon / n_batches,
        'val/kl_loss':    total_kl / n_batches,
    }

#### Add a generaton evaluation step ####
@torch.no_grad()
def quick_generation_eval(model: VAE, device, n_samples=200):
    """Small generation sanity check for checkpointing."""
    model.eval()
    lang = Lang()

    z = torch.randn(n_samples, VAE.LATENT_DIM, device=device)
    _, generated = model.decoder(z)  # autoregressive path

    all_smiles = [lang.indexToSmiles(row.tolist()) for row in generated]

    n_valid = 0
    unique_valid = set()
    for smi in all_smiles:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            n_valid += 1
            unique_valid.add(Chem.MolToSmiles(mol, canonical=True))

    model.train()
    return {
        "gen/valid_rate": n_valid / n_samples,
        "gen/unique_valid": len(unique_valid),
        "gen/unique_gen_strings": len(set(all_smiles)),
    }


# ================================================================================
# Training loop (without wandb)
# ================================================================================

def run_training(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    #### Dataset & split ####
    full_dataset = SmilesDataset(args.train_data, max_length=args.max_length)
    train_set, val_set = make_train_val_split(
        full_dataset, val_frac=args.val_frac, seed=args.split_seed
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,                       # allows SGD
        num_workers=4,                      # 4 * num_GPUs
        pin_memory=(device.type == 'cuda'), # pairs with non_blocking=True
        drop_last=True,                     # drop the last incomplete batch
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,                      # No need to
        num_workers=4, pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )
    print(f"Train batches per epoch: {len(train_loader):,} | Val size: {len(val_set):,}")

    # # Optionally save training SMILES set for ** novelty evaluation **
    # if args.save_train_pickle:
    #     lang = Lang()
    #     train_smiles = set()
    #     # Get the training SMILES string
    #     for idx in train_set.indices:
    #         smi = lang.indexToSmiles(full_dataset.examples[idx])
    #         if smi:
    #             train_smiles.add(smi)
    #     # Save
    #     with open(args.save_train_pickle, 'wb') as f:
    #         pickle.dump(train_smiles, f)
    #     print(f"Saved {len(train_smiles):,} training SMILES to {args.save_train_pickle}")


    ############ Create an instance of the Model class ############
    model = VAE(
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        max_length=args.max_length,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        rnn_cell_type=args.rnn_cell_type,
        word_dropout_rate=args.word_dropout_rate,
    ).to(device)

    # Log num of parameters in each model && model architecture
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Model Architecture:")
    print(f"  RNN cell type:   {args.rnn_cell_type}")
    print(f"  Bidirectional:   {args.bidirectional}")
    print(f"  Embedding dim:   {args.embedding_dim}")
    print(f"  Hidden size:     {args.hidden_size}")
    print(f"  Num layers:      {args.num_layers}")
    print(f"  Dropout:         {args.dropout}")
    print(f"  Max length:      {args.max_length}")
    print(f"  Latent dim:      {VAE.LATENT_DIM} (fixed)")
    print(f"  Vocab size:      {VAE.VOCAB_SIZE} (fixed)")
    print(f"  Model parameters: {n_params:,}")
    print(f"Training:")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Max epochs:      {args.epochs}")
    print(f"  Grad clip:       1.0")
    print(f"  *Word dropout:    {args.word_dropout_rate}")
    print(f"  *KL max weight:   {args.kl_max_weight}")
    print(f"\n{'='*60}")


    ############ Setup  optimizer, LR scheduler, KL weight scheduler param ############
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(   # Scheduler is driven by val loss (keep high LR)
        optimizer, mode='min', #min = metric stop decreasing
        patience=3, factor=0.5
    )

    steps_per_epoch = len(train_set) // args.batch_size
    # weight = 0.5 at epoch 2, weight ≈ 0.99 at epoch 5
    kl_t0 = steps_per_epoch * 2
    t_99 = steps_per_epoch * 5
    kl_k  = math.log(99) / (t_99 - kl_t0)   # at step t_99 (epoch 5), we want the KL weight to be 0.99


    ############ Training loop ############
    global_step = 0
    best_val_loss = float('inf')
    best_gen_valid_rate = -1.0
    best_gen_unique_valid = -1
    best_val_recon = float("inf")

    for epoch in range(args.epochs):

        model.train()
        epoch_loss = epoch_recon = epoch_kl = 0.0
        t0 = time.time()

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)

            # Define kl weight per step
            kl_w = kl_sigmoid_weight_schedule(global_step, t0_steps=kl_t0, k=kl_k,
                                              kl_max_weight=args.kl_max_weight)

            optimizer.zero_grad() #IMPORTANT! reset gradient

            ## 1. forward pass
            logits, _, (mu, logvar, _) = model(batch)
            ## 2. calculate loss (mean loss here)
            loss, recon, kl = vae_loss(logits, batch, mu, logvar, kl_weight=kl_w)
            ## 3. backward prop
            loss.backward()
            ## 4. optimization, update param
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # default val; prevent gradient exploding - it computes the total norm across all parameters, and if it exceeds the threshold, scales all gradients down proportionally
            optimizer.step()

            ## Sum up epoch stats
            epoch_loss  += loss.item()
            epoch_recon += recon.item()
            epoch_kl    += kl.item()
            global_step += 1

            ## Periodic logging
            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Loss {loss.item():.4f} | Recon {recon.item():.4f} | "
                    f"KL {kl.item():.4f} | KL_w {kl_w:.4f} | {elapsed:.1f}s",
                )
                t0 = time.time()

            ## Periodic validation
            if global_step % args.val_every == 0:
                val_metrics = run_validation(
                    model=model, 
                    val_loader=val_loader, 
                    device=device, 
                    kl_weight=args.kl_max_weight,
                    val_batches=args.val_batches
                )

                # Modify LR platau scheduler based on recon_loss
                scheduler.step(val_metrics["val/recon_loss"])

                # [NOTE] Include quick generation check
                gen_metrics = quick_generation_eval(
                    model=model,
                    device=device,
                    n_samples=args.gen_eval_samples,
                )

                print(
                    f"  [VAL] Step {global_step} | Loss {val_metrics['val/overall_loss']:.4f} | *Recon {val_metrics['val/recon_loss']:.4f} | KL {val_metrics['val/kl_loss']:.4f}"
                )
                print(
                    f"  [GEN] Step {global_step} | Valid {gen_metrics['gen/valid_rate']:.1%} | "
                    f"UniqueValid {gen_metrics['gen/unique_valid']} | "
                    f"UniqueStr {gen_metrics['gen/unique_gen_strings']}"
                )

                # [NOTE] Checkpoint rule
                improved = False
                if gen_metrics["gen/valid_rate"] > best_gen_valid_rate:
                    improved = True
                elif gen_metrics["gen/valid_rate"] == best_gen_valid_rate:
                    if gen_metrics["gen/unique_valid"] > best_gen_unique_valid:
                        improved = True
                    elif gen_metrics["gen/unique_valid"] == best_gen_unique_valid:
                        if val_metrics["val/recon_loss"] < best_val_recon:
                            improved = True
                
                # Update best metrics
                if improved:
                    best_gen_valid_rate = gen_metrics["gen/valid_rate"]
                    best_gen_unique_valid = gen_metrics["gen/unique_valid"]
                    best_val_recon = val_metrics["val/recon_loss"]
                    best_val_loss = val_metrics["val/overall_loss"]
                    _save_model(model, args, device)
                    print(
                        f"  -> Saved best model (gen_valid_rate={best_gen_valid_rate:.1%}, gen_unique_valid={best_gen_unique_valid}, val_recon={best_val_recon:.4f})"
                    )

        # ---- End-of-epoch summary ----
        n_batches = len(train_loader)
        print(
            f"\n=== Epoch {epoch} done | "
            f"AvgTrainLoss {epoch_loss/n_batches:.4f} | "
            f"AvgRecon {epoch_recon/n_batches:.4f} | "
            f"AvgKL {epoch_kl/n_batches:.4f} ===\n",
            flush=True,
        )

    # Final log
    print(f"Training complete. Best val loss: {best_val_loss:.4f}. "
          f"Model saved to {args.out}", flush=True)


def _save_model(model: VAE, args, device: torch.device):
    """
    # This will create the file that you will submit to evaluate SMILES 
    # generated from a normal distribution. Note that vae.decoder must
    # be a Module 
    """
    # LATENT_DIM = 1024
    model.eval()

    # with torch.no_grad():
    #     z_dummy = torch.normal(0, 1, size=(1, VAE.LATENT_DIM), device=device)
    #     # Tracing runs the decoder once with z_dummy and **records every operation** that happen
    #     #   - torch.multinomial is not deterministic so we disable trace checking (check_trace will flag if outputs differs with the same input)
    #     #   - Traceing internally calls model.decoder(z_dummy) — with only z and no second argument -> So teacherForcing_actual_input=None
    #     traced = torch.jit.trace(model.decoder, z_dummy.to(device),
    #                              check_trace=False)
    # torch.jit.save(traced, args.out)

    export = ExportDecoder(model.decoder).to(device)
    export.eval()
    scripted = torch.jit.script(export)
    torch.jit.save(scripted, args.out)

    model.train()
    


if __name__ == '__main__':
    LATENT_DIM = 1024

    parser = argparse.ArgumentParser('Train a Variational Autoencoder')
    parser.add_argument('--train_data','-T',required=True,help='data to train the VAE with')
    parser.add_argument('--out',default='vae_decoder_gen.pth',help='File to save generate function to')
    parser.add_argument('--max_length', type=int, default=150)
    # parser.add_argument('--save_train_pickle', default=None, help='(Optional) save a pickle of training SMILES for novelty eval')

    # Train/val split
    parser.add_argument('--val_frac', type=float, default=0.05, help='Fraction of data held out for validation (default 5%%)')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed for train/val split (fixed for reproducibility)')
    
    # Model Architecture
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bidirectional', action='store_true', default=True)
    parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false')
    parser.add_argument('--rnn_cell_type', type=str, default='GRU', choices=['GRU', 'LSTM'])

    # Training
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--word_dropout_rate', type=float, default=0.5, help='Fraction of teacher-forcing tokens randomly masked to prevent posterior collapse')
    parser.add_argument('--kl_max_weight', type=float, default=0.01, help='Maximum KL weight (caps the sigmoid schedule); also try to solve posterior collapse')
    parser.add_argument("--gen_eval_samples", type=int, default=200)

    # Logging
    parser.add_argument('--log_every', type=int, default=500) #steps
    parser.add_argument('--val_every', type=int, default=5000, help='Run validation every N training steps')
    parser.add_argument('--val_batches', type=int, default=200,help='Number of val batches to evaluate per validation run')

    args = parser.parse_args()
    run_training(args)

