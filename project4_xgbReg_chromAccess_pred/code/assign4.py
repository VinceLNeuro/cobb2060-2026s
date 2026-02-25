#!/usr/bin/env python3
import Bio
import Bio.motifs as motifs
from Bio import SeqIO
import numpy as np
import xgboost as xgb
import argparse, sys,time,os
import multiprocessing
import pandas as pd


######## METHODS ########

#### 0. Set up global var for multiprocessing workers (Pool) <- will be passed by fun1 ####
MOTIFS_LOG_ODDS = None
MOTIFS_THRESHOLDS = None
MOTIFS_LENGTHS = None

def init_worker(log_odds_list, thresholds, lengths):
    """Initialize worker process with motif data."""
    global MOTIFS_LOG_ODDS, MOTIFS_THRESHOLDS, MOTIFS_LENGTHS
    MOTIFS_LOG_ODDS = log_odds_list
    MOTIFS_THRESHOLDS = thresholds
    MOTIFS_LENGTHS = lengths
    

#### 1. parse the motif file ####
"""
motif file (PFM): 4 columns - ACGT
- we want to parse for pfm for each TF (using the threshold and ACGT weight)
- then we want to convert it to PWM -> PSSM/logodds matrix

return: list of PSSM, list of threshold, list of length of motif
"""
def parse_motifs(path):

    with open(path) as f:
        parsed_motifs = motifs.parse(f, fmt='pfm-four-columns')
    # print(f'Parsed {len(parsed_motifs)} motifs...'

    log_odds_list = []
    thresholds = []
    lengths = []
    
    for m in parsed_motifs: # m is one motif
        # Extract threshold from name (will be the 3rd field)
        name_parts = m.name.split('\t')
        threshold = float(name_parts[-1])

        # m.counts = PFM
        # PFM -> PWM -> PSSM (logodds)
        pssm = m.counts.normalize(pseudocounts=0.01).log_odds()
        
        # Convert to numpy array format for Bio.motifs._pwm.calculate (same as what is indicated in "https://github.com/biopython/biopython/blob/master/Bio/motifs/matrix.py#L402")
        # Shape: (length, 4) with columns A, C, G, T
        log_odds = np.array(
            [ [pssm[letter][i] for letter in "ACGT"] for i in range(len(m)) ], # len(m) is the number of posistion
            dtype=np.float64
        )
        
        log_odds_list.append(log_odds)
        thresholds.append(threshold)
        lengths.append(len(m))
    
    return log_odds_list, np.array(thresholds, dtype=np.float32), np.array(lengths, dtype=np.int32)


#### 2. Extract sequences from genome (bed format) ####
def extract_sequences(bed, genome_fa):
    sequences = []
    for idx, row in bed.iterrows():
        chrom = row['chrom']
        start_pos = row['start']
        end_pos = row['end']
        try:
            seq = str(genome_fa[chrom][start_pos:end_pos]).upper()
            sequences.append(seq)
        except:
            sequences.append('')
    return sequences


#### 3. Featurize sequence ####
def featurize_single_sequence(seq):
    """
    Featurize single sequence -> 1. binary hit and 2. max_score per motif within the sequence (effect size)

    Return: per motif -> length of 2 vector; across the motifs
    """

    # Get global var in parallel
    global MOTIFS_LOG_ODDS, MOTIFS_THRESHOLDS, MOTIFS_LENGTHS
    num_motifs = len(MOTIFS_LOG_ODDS)
    
    if not seq: #empty
        return np.zeros(num_motifs * 2, dtype=np.float32) #2 features per motif

    
    seq_bytes = bytes(seq, 'ASCII') # specify ASCII encoding
    seq_len = len(seq_bytes)

    # Make reverse complement for the sequence --> so that it will match the __sequence for TF binding motif__
    rc_seq_bytes = seq_bytes.translate(
        bytes.maketrans(b'ACGTacgtNn', b'TGCAtgcaNn')
    )[::-1]
    
    features = np.zeros(num_motifs * 2, dtype=np.float32)

    # Across all the motifs for this sequence
    for m_idx in range(num_motifs):
        log_odds = MOTIFS_LOG_ODDS[m_idx]
        threshold = MOTIFS_THRESHOLDS[m_idx]
        m_len = MOTIFS_LENGTHS[m_idx]
        
        if seq_len < m_len: #only process sequence longer than the motif
            continue
        
        n_positions = seq_len - m_len + 1

        ## [SLIDE OVER BOTH STRAND - forward and reverse]
        scores_fwd = np.empty(n_positions, np.float32)
        motifs._pwm.calculate(seq_bytes, log_odds, scores_fwd) # use the PSSM to slide, store result to the np array
        scores_rc = np.empty(n_positions, np.float32)
        motifs._pwm.calculate(rc_seq_bytes, log_odds, scores_rc)

        # transformation
        max_score = max(np.max(scores_fwd), np.max(scores_rc))
        has_hit = 1.0 if max_score >= threshold else 0.0
        
        # store
        base_idx = m_idx * 2
        features[base_idx] = has_hit
        features[base_idx + 1] = max_score
    
    return features


#### 4. Multiprocessing setup ####
def featurize_batch_worker(batch):
    """Worker function to featurize a batch of sequences."""
    start_idx, sequences = batch
    
    num_motifs = len(MOTIFS_LOG_ODDS)
    n_seqs = len(sequences)
    features = np.zeros((n_seqs, num_motifs * 2), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        features[i] = featurize_single_sequence(seq)
    
    return (start_idx, features)


def featurize_sequences_parallel(sequences, log_odds_list, thresholds, lengths, n_workers=None):
    """Featurize all sequences in parallel using multiprocessing.Pool with initializer."""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    n_samples = len(sequences)
    num_motifs = len(log_odds_list)
    n_features = num_motifs * 2
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    
    # Create batches - more batches for better load balancing
    batch_size = max(1, n_samples // (n_workers * 4)) #allow workers picking up lower jobs (4x as sweet spot)
    batches = []
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batches.append((i, sequences[i:end_idx]))
    
    # Use Pool with initializer to pass motif data to workers
    with multiprocessing.Pool(n_workers, initializer=init_worker, initargs=(log_odds_list, thresholds, lengths)) as pool:
        results = pool.map(featurize_batch_worker, batches)
    
    for start_idx, features in results:
        end_idx = start_idx + features.shape[0]
        X[start_idx:end_idx] = features
    
    return X



######## MAIN ########
if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Predict ATAC-seq data from sequence motifs')
    parser.add_argument('-g','--genome',help='reference genome FASTA file',required=True)
    parser.add_argument('-t','--train',help='training bed file with chromosome, start, end and label',required=True)
    parser.add_argument('-m','--motifs',help='file of motifs to use as features',required=True)
    parser.add_argument('-p','--predict',help='training bed file for prediction with chromosome, start, end', required=True)
    parser.add_argument('-o','--output_file',help='Output predictions',required=True)

    args = parser.parse_args()

    start_time = time.time()
    
    #### Load data ####
    # Load motifs
    print(f"Loading motifs from {args.motifs}...")
    log_odds_list, thresholds, lengths = parse_motifs(args.motifs)
    print(f">> Loaded {len(log_odds_list)} motifs")
    
    # Load genome
    print(f"Loading genome from {args.genome}...")
    genome_fa = {}
    for seqr in SeqIO.parse(args.genome, 'fasta'):
        genome_fa[seqr.name] = seqr.seq    
    print(f">> Loaded {len(genome_fa.keys())} chromosomes")
    
    # Load training data
    print(f"Loading training data from {args.train}...")
    train_df = pd.read_csv(args.train, sep='\t', header=None, 
                           names=['chrom', 'start', 'end', 'label'])
    train_df = train_df.reset_index(drop=True)
    y_train = train_df['label'].values
    print(f">> Loaded {len(train_df)} training samples")
    
    # Load test data
    print(f"Loading test data from {args.predict}...")
    test_df = pd.read_csv(args.predict, sep='\t', header=None,
                          names=['chrom', 'start', 'end'])
    test_df = test_df.reset_index(drop=True)
    print(f">> Loaded {len(test_df)} test samples")
    
    # Extract sequences
    print("Extracting training sequences...")
    train_sequences = extract_sequences(train_df, genome_fa)
    print("Extracting test sequences...")
    test_sequences = extract_sequences(test_df, genome_fa)


    #### Featurizing ####
    # Featurize training data in parallel
    print("Featurizing training data...")
    feat_start = time.time()
    X_train = featurize_sequences_parallel(train_sequences, log_odds_list, thresholds, lengths)
    print(f">> Featurization took {time.time() - feat_start:.2f}s")
    print(f">> Feature matrix shape: {X_train.shape}")
    
    # Featurize test data in parallel
    print("Featurizing test data...")
    feat_start = time.time()
    X_test = featurize_sequences_parallel(test_sequences, log_odds_list, thresholds, lengths)
    print(f">> Featurization took {time.time() - feat_start:.2f}s")
    
    
    #### Train XGBoost model - post-hyperparam-search ####
    print("Training XGBoost model...")
    train_start = time.time()

    ## hyperparam
    n_estimators = 200
    max_depth = 7
    learning_rate = 0.1
    subsample = 0.8
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        n_estimators = n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"  Training took {time.time() - train_start:.2f}s")

    
    # Predict
    print("Making predictions...")
    predicts = model.predict(X_test)

    
    #### Save predictions ####
    np.save(args.output_file,predicts)
    print(f"Predictions saved to {args.output_file}")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s")

    
    