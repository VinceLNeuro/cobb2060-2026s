#!/usr/bin/env python3    
import numpy as np
import umap
import faiss
import scanpy as sc
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #ignore warning about files we are using
# from sklearn.metrics import adjusted_mutual_info_score, make_scorer
import scipy.sparse as sp
from collections import Counter
from sklearn.preprocessing import LabelEncoder
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV

parser = argparse.ArgumentParser(description='Unsupervised or Supervised Learning using UMAP, FAISS, and sklearn')
parser.add_argument('-d','--data',help='unlabeled anndata input file',required=True)
parser.add_argument('-k',type=int,help='Number of clusters to identify',required=False)
parser.add_argument('-t','--train_data',help='labeled anndata file for training',required=False)
parser.add_argument('-o','--output_file',help='Output file of cluster assignments (npy).',required=True)

args = parser.parse_args()


#### Patch -- resolve umap 0.5.11 / sklearn version conflict [remove in submission] ####
try:
    import umap.umap_ as _umap_module
    _orig_check = _umap_module.check_array
    def _patched_check(*args, ensure_all_finite=True, **kwargs):
        kwargs['force_all_finite'] = ensure_all_finite
        return _orig_check(*args, **kwargs)
    _umap_module.check_array = _patched_check
except Exception:
    pass
####


RANDOM_STATE = 42 # match the instruction
########## Hyperparam setup (check `1_testRun.ipynb` for grid search process) ##########
# Best params: {'n_components': 20, 'n_neighbors': 20, 'min_dist': 0.1} with AMI=0.6019
Kmeans_n_components = 20
Kmeans_n_neighbors  = 20
Kmeans_min_dist     = 0.1


########## Helper functions ##########
# normalize gene names (handle ENSMUSG format)
def unify_varnames(adata):
    adata.var_names = [v.split("_")[-1] for v in adata.var_names]
    adata.var_names_make_unique()

def preprocess_data(adata, min_cells=3, filtered_col_name = 'highly_variable'):    
    
    import scipy.sparse as sp
    # Check if data looks already log-normalized
    #     If max value is very large (>100), likely raw counts needing normalization
    max_val = adata.X.max() if not sp.issparse(adata.X) else adata.X.max()
    is_raw = max_val > 50

    if is_raw:
        # Save raw counts before normalization
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    ### Filter for HVGs ###
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if filtered_col_name in adata.var.columns: #use this
        adata = adata[:,adata.var[filtered_col_name]].copy()
        print('\n>> Used column exisiting filter')
    
    else: # use built in filter_genes
        if is_raw:
            # seurat_v3 needs raw counts — use the counts layer
            sc.pp.highly_variable_genes(
                adata, n_top_genes=2000, flavor='seurat_v3', layer='counts'
            )
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
        adata = adata[:, adata.var['highly_variable']].copy()
        print('\n>> Computed highly_variable_genes')

    return adata

def umap_embed(X, n_components, n_neighbors, min_dist,
              random_state = RANDOM_STATE):
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    return reducer.fit_transform(X).astype(np.float32)



######## Different tasks based on existence of train_data ########
if args.train_data is not None:
    # ---- Supervised Labeling -- prediction of `cell_ontology_class` ----

    train_adata = sc.read_h5ad(args.train_data)
    unify_varnames(train_adata)
    test_adata  = sc.read_h5ad(args.data)
    unify_varnames(test_adata)

    # align the gene names in train and test (before concat)
    common = np.intersect1d(train_adata.var_names, test_adata.var_names)
    print('Numb of common genes shared between two data:', len(common))
    train_adata = train_adata[:, common].copy() #filter row names
    test_adata = test_adata[:, common].copy()
    
    # preprocess
    train_adata = preprocess_data(train_adata)
    test_adata = preprocess_data(test_adata)
    
    # combine for UMAP
    combined = sc.concat([train_adata, test_adata], axis=0, #by obs/cells
                         join='inner')
    print(f'Combined data shape: {combined.shape}')
    # convert to array (handle sparse mat)
    X = combined.X.toarray() if hasattr(combined.X, 'toarray') else combined.X
    
    # compute UMAP embedding on combined data
    #       this allows a better manifold distribution in the test set by providing ground truth labels
    embedding = umap_embed(X, random_state=RANDOM_STATE, 
                            n_components=Kmeans_n_components, 
                            n_neighbors=Kmeans_n_neighbors, 
                            min_dist=Kmeans_min_dist)
    # split back into train and test
    n_train = train_adata.n_obs
    X_train_umap = embedding[:n_train]
    X_test_umap = embedding[n_train:]

    # get labels
    y = train_adata.obs["cell_ontology_class"].astype(str).to_numpy()
    print('Ground Truth Labels:', Counter(y))

    # normalize labels (numerical)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(y[:5])
    print(y_enc[:5])

    ### Best Classifier Identified from CV on labeled data only ###
    # Best classifier:  RF_200 with AMI=0.6241
    # Grid search res:  Best params: {'max_depth': 10, 'max_features': 0.3, 'n_estimators': 500}
    #                   Best AMI:    0.6279
    clf = RandomForestClassifier(n_estimators=500, max_depth=10, max_features=0.3, 
                                 class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_umap, y_enc)

    # Predict
    I = le.inverse_transform( #get the label (cellontology) back
        clf.predict(X_test_umap)
    )
    print(Counter(I))


else:
    # --- Clustering ----
    
    # read
    adata = sc.read_h5ad(args.data)
    # deal with the ENSG prefix within some dataset var_names -> get the gene name only
    unify_varnames(adata)
    # preprocess
    adata_hvg = preprocess_data(adata)
    # compute umap embedding
    X = adata_hvg.X.toarray() if hasattr(adata_hvg.X, 'toarray') else adata_hvg.X
    print(X.shape)
    embedding = umap_embed(X, random_state=RANDOM_STATE, 
                           n_components=Kmeans_n_components, 
                           n_neighbors=Kmeans_n_neighbors, 
                           min_dist=Kmeans_min_dist)
    # train the kmeans
    kmeans = faiss.Kmeans(d=Kmeans_n_components, #umap space
                           k=args.k, niter=50, seed=RANDOM_STATE)
    kmeans.train(embedding)
    # mapping from embedding to the centroids (first arg is distance)
    _, cluster_ids = kmeans.index.search(embedding, 1)
    # extract cluster ids into I, a flat one-dimensional array 
    I = cluster_ids.flatten()



########## OUTPUT ##########
np.save(args.output_file, I, allow_pickle=True)

