# Goal: Train a VAE to generate SMILES strings

Project website: https://bits.csb.pitt.edu/cobb2060/assign6/


## Data

Preprocessed SMILES (Simplified Molecular Input Line Entry System) strings

1. Prefiltered to reduced characters; (length < 150) ==> __62M valid SMILES strings__
2. Randomized ("randomization imporved the quality of molecular generative models") - `0_randomize_SMILES.py`

==> `project6_VAE_PubChemSMILES/data/pubchem_randomized.smi.gz`

<br>


## Training pipeline

### 1. Preprocessing SMILES string -> Encoding

- Preprocess the SMILES strings into indices based on pre-set dictionary -> save as 'npy' file (shorter set (with smaller molecules) for speed up and EDA, and full set for final training)

    - Rationale: When creating the `Dataset` class (for later `DataLoader` input), the parsing and index conversion is the same across the training sessions. Since we will do multiple training sessions (in hyperparam search), preprocessing can speed up greatly. 

    - `1_preprocess.py`
        
        - _<span style="color:gold">[NOTE]: From the EDA, we know that the `pubchem_randomized.smi.gz` file should be shuffled (- no clear pattern from first to the end) -> safe to select the first N molecules</span>_

        - Output: 

            - An unsigned int8 numpy array of shape __<span style="color:gold">(N, max_length)</span>__ where each row is a __zero-padded (same as EOS token padding at the end)__, index-encoded SMILES string.

            - We used this function to _<span style="color:gold">downsample</span>_ the 62M data to N molecules with max length (e.g., lenMax_MolN = len150_Mol5M)


### 2. General Model Architecture & Training

Final training PY script: /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/`run4_longMask_v3_genEval_default/out_v3JitScript/train_VAE_v3JitScript.py`

Final training SLURM: /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/`run4_longMask_v3_genEval_default/out_v3JitScript/run_v3JitScript_len150.sh`

Final training LOG: /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/`run4_longMask_v3_genEval_default/out_v3JitScript/run_v3JitScript_len150__defaultVAE_53441637.out`

__Final VAE: /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/`run4_longMask_v3_genEval_default/out_v3JitScript/vae_decoder_len150_Mol5M_defaultVAE_JitScript.pth`__
- model/vae_decoder_len150_Mol5M_defaultVAE_JitScript.pth

--- No sweep performed ---

```
============================================================
Model Architecture:
  RNN cell type:   GRU
  Bidirectional:   True
  Embedding dim:   128
  Hidden size:     512
  Num layers:      2
  Dropout:         0.1
  Max length:      150
  Latent dim:      1024 (fixed)
  Vocab size:      28 (fixed)
  Model parameters: 12,429,340
Training:
  Batch size:      512
  Learning rate:   0.001
  Max epochs:      10
  Grad clip:       1.0
  *Word dropout:    0.3
  *KL max weight:   0.02

============================================================
```

#### Additional notes recorded in final PY

VAE Architecture:
- Encoder: RNN (GRU or LSTM, optionally bidirectional)
- Reparameterization: z = mu + eps * std
- Decoder: Undirectional RNN (GRU or LSTM); teacher forcing during training/val (removed in export using `torch.jit.script`)

Training Notes:
- Added optional random word-dropout (or masking) on tearcher-forcing decoder input  -> so that the _decoder_ will use z more than relying on teacher-forcing
- Data-dependent Sigmoid KL annealing schedule (applying a max KL weight to make it LESS focus on matching ~N(0,1))
- ReduceLROnPlateau scheduler
- Gradient clipping to prevent gradient explosion

Updates:
- [BUG] Fixed mixed ordering of args in VAE/self.decoder
- [BUG] Fixed: create custom mask, so that first EOS is included in the CE loss calculation (fn:`vae_loss`)
- [BUG] Fixed generatng SOS in inference mode (_later removed again for simplicity_)
- [BUG] Fixed passing kl_weight to `run_validation/vae_loss`
- [IMPORTANT_NOTE] Include a generation evaluation chunk fn `quick_generation_eval` -> update checkpoint selection rule; __Game-changing__
- [NOTE] Change the LR scheduler update scheme to be based on `val/recon_loss`
- [IMPORTANT_NOTE] Change `_save_model` from `torch.jit.trace` to `torch.jit.script` for more stable model export

#### What is a healthy train/val signal

__<mark> Higher valid rate & num unique valid  +++  Lower val_recon_loss  +++  Moderate KL (no posterior collapse) </mark>__

<br>


## Decoder Generation Evaluation (output summary stats)

PY script: /net/dali/home/mscbio/til177/Github/cobb2060-2026s/project6_VAE_PubChemSMILES/code/`3_eval.py`

### NOTE

- Model saved using `torch.jit.trace` will have issue in for-loop generation if `with torch.no_grad():` is included in the code

- Everything's good for model saved using `torch.jit.script`

<br>


## TODO: Hyperparameter Sweep Results

