# CNN Multi-class Classifier for Cell Image

Project website: https://bits.csb.pitt.edu/cobb2060/assign5/

## Data

- Cells: MCF7 breast cancer cell line (labeled for for DNA, F-actin, and B-tubulin, imaged by fluorescent microscopy)

- Label: Mechanisms of action (MOA), or compound perturbation

- Original data: https://data.broadinstitute.org/bbbc/BBBC021/
    
    - Data for this project: https://bits.csb.pitt.edu/cobb2060/assign5/train.tar (22G)
        
        - Processed images (training set n=14,430; held-out test set n=1,718)
        
        - >Each training example consists of three 512x512 greyscale images (one image for each fluorescent probe - DNA/actin/tubulin)
            - greyscale = 1 channel
            - 3 dimensions (DNA/actin/tubulin) = __3 input channels__
            - <mark>(3, 512, 512) as model input


## Goal

- Build a Pytorch-based neural network (multi-class classifier) to predict MOA from fluorescent image. 

    - Rationale: different compounds perturb distinct molecular targets and pathways, which in turn produce __different morphological changes__ visible under microscopy


## Training Workflow

- Version 1: `train.py` --> `out_run1_default_epoch1_imgSize512`

    - The groundwork framework (before hyperparameter search)
    - Used default hyperparams with short training epoch (epoch=1)
    - Results:
        - Noticed great gap between validation acc and test acc --> Preprocessing mismatch

- Version 2: `train_ExportModel.py` --> `out_run2_default_epoch1_imgSize512_ExportModel`

    - Added in ExportModel Class so that same preprocessing (min-max norm + size reduction) can be done in the eval.py
    - Others kept the same as run 1
    - Results:
        - __(submitted for checkpoint assignment)__
        - Great improvement in test acc
        - Still underfitting --> require increased epochs

- Version 3: `train_ExportModel.py` --> `out_run3_default_epoch30_imgSize128_ExportModel`

    - Only changed epoch (1->30) & img_size (512->128) from run 2
    - Results:
        - best model @ <mark style="background:green">21st epoch: __val_acc=0.9040__</mark>
        - __Include wandb sweep hyperparameter search based on this script__

<br>

- [test_5_epochs] Version 4: `train_ExportModel_wandb_sweep/train_ExportModel_wandb_sweep.py` --> `train_ExportModel_wandb_sweep/out`

    - Notes about `wandb` sweep
        * build a config yaml file 
            * (start with random search, then grid search when have some clue)
        * init -> args. to cfg. & add log
        * get sweep id
        * start agents

    - Use random search for hyperparam sweep for 20 runs
    
    - Models in `train_ExportModel_wandb_sweep/out/model_ExportModel_sweep_{wandb.run.id}.best.pt`

    - <mark style="background:green">Conclusion about this sweep - __gained some info, but need longer run!__</mark>
        - Important figures: 
            - val/epoch_acc, val/epoch_loss: 
                - sweep11 has highest acc, but overfitting
                - sweep2 is the best
                - sweep16 is legit
                - sweep7,20,17,12 can be valid downstream options

            - https://wandb.ai/tvluo-university-of-pittsburgh/moa-cnn-classifier/sweeps/rxpxykb7/workspace/panel/ifu1gmyrd (filtered as below)

        - Table: 
            - Runs (__filter for completed (pass early stop) & top acc__)
                - 'Parameter importance with respect to' -> conclusion, rank by importance
                    - longer runtime (around max 30)
                    - smaller dropout_p
                    - smaller batch_size
                    - larger n_conv_blocks, weight_decay
                    - smaller lr
                    - larger fc_size
                - __img_size: 224__
                - __n_conv_blocks: 5__
                - fc_size: 256/512          (both has great acc instances)
                - base_ch: 32/16            (both has great acc instances)
                    - batch_size: 16/64     (see if 16 better)
                - lr: 0.0005, 0.001         (kind of distributed out)
                - dropout_p: 0.50, 0.28
                - weight_decay: 0.0001, 0.00001

<br>

- Version 5: `train_ExportModel_wandb_sweep2`

    - Random search for 20 epochs (20 runs parallel)
        - Updated sweep config from sweep1
    
    - __Conclusion__: should remove early-terminate & increase runs and epochs

<br>

- __<mark>[FINAL-30-epochs]</mark> Version 6: `train_ExportModel_wandb_sweep3_bayes`__

    - Major update:
        - Switch to __Bayesian search__ for better results
        - Removed early terminate (too aggressive for 20 runs) & Applied Longer/Full epochs -> Final analyses
        - Modified the max boundary to include some hyperparam values
        - Limited `base_ch` to 32 only based on previous two sweeps

    - Observations:
        1. Smooth training curve, but zigzag val curve: 
            - This is the result of `WeightedRandomSampler`, causing varied training batches. Each epoch, the sampler draws a different random mix of oversampled minority classes. 
            Some epochs the model gets more "helpful" examples, others less. This creates epoch-to-epoch variation in what the model learns, which shows up as validation fluctuation.
            - Also can be small validation set, given 13 labels
            - <mark>__NO OVERFITTING__ (given that the trend is good, the noise is a healthy trajectory)<mark>
        2. Top models has below CLEAR traits:
            - lr = 0.001
            - dropout_p = 0.1
            - img_size = 224
            - batch_size = 64
            - fc_size = 256
            - n_conv_blocks = 5
            - weight_decay little effect, but should be around 0.0001

    - <mark>Sweep link</mark>: https://wandb.ai/tvluo-university-of-pittsburgh/moa-cnn-classifier/sweeps/6axw7aap?nw=nwusertvluo

## Best model (built from scratch)

balmy-sweep-6 (tvluo-university-of-pittsburgh/moa-cnn-classifier/yjy73vh6)

- `/net/dali/home/mscbio/til177/Github/cobb2060-2026s/project5_CNN_cellImg_classifier/code/train_ExportModel_wandb_sweep3_bayes/out/model_ExportModel_sweep_yjy73vh6.best.pt`


