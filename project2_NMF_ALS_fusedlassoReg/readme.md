## Goal

- Input
    - bigwig-files, chr_number, start_pos, end_pos, outputName, **num_latent_features**
        
        - BW files: 
            - cell line: K562 cells; a human immortalized myelogenous leukemia line
            - ChIP-seq for multiple proteins
            - FAIRE-seq and DNaseI-seq for chromatin accesibiltiy
            - ```
                (start, end, value)
                (start, end, value)
                (start, end, value)
              ```
                        
        - matrix factorization structure:
            - row = genomic loc
            - col = bw file (different hits at the loc)

- Method
    - NMF using alternating least squares with a fused-lasso regularization to smooth the latent factors (hyperparameter tuning)
        - fused-lasso: penalizes chnages between adjacent genomic positions
            - y is unsmoothed vector, theta is smoothed y
            - Higher `fl_lambda` = cares more about penalty between adjacent genomic position - more similar nearby (smooth)
        
        - `l2_penalty` in diagonal matrix: for stabiltiy issue
