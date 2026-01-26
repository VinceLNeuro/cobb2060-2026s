import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import argparse, time
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif # a bit slow
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer


parser = argparse.ArgumentParser(description='Predict if genes are mutated based on mRNA expression data.')
parser.add_argument('-e','--train_expr',nargs='+',help='Expression data file(s)',required=True) #one or more
parser.add_argument('-m','--train_mut',nargs='+',help='Mutation data file(s)',required=True)
parser.add_argument('-p','--test_expr',help='Expression data of patients to predict',required=True)
parser.add_argument('-g','--gene',help='Hugo symbol of gene(s) to predict if mutated',nargs='+',required=True)
parser.add_argument('-o','--output_prefix',help='Output prefix of predictions. Each gene\'s predictions are output to PREFIX_GENE.txt',required=True)

args = parser.parse_args()
# data_dir='/ihome/hpark/til177/GitHub/cobb2060-2026s/Data_cobb2060/proj1'



#### Load data - assume all tsv data ####
def read_expr(fname, skiprows=0):
    """fname: file path (tested, all are TSV with first line as header)"""

    import warnings
    # Suppress only DtypeWarning
    warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

    expr = pd.read_table(fname, sep = '\t', header=0, skiprows=skiprows)
    print('==== Loaded %s ====' %(fname))
    
    return expr



#### read in expression training data (X) ####
ls_expr_df = defaultdict(list)
ls_num_missing_hugo = defaultdict(list)
for subpath in args.train_expr:
    path=subpath
    # load data
    df = read_expr(path)
    # check_missingness(df)
    # store to a dict, using subpath as key
    ls_expr_df[subpath].append(df)
    ls_num_missing_hugo[subpath] = df['Hugo_Symbol'].isna().sum()
# print(ls_expr_df.keys())
# print(ls_num_missing_hugo)

d = ls_num_missing_hugo
min_key = min(d, key=d.get)
min_val = d[min_key]
print("min key:", min_key, "min value:", min_val)

ref_hugo_entrez = ls_expr_df[min_key][0][['Hugo_Symbol','Entrez_Gene_Id']].copy()
ref_hugo_entrez = ref_hugo_entrez.dropna(subset=['Entrez_Gene_Id','Hugo_Symbol'])
ref_hugo_entrez = ref_hugo_entrez.drop_duplicates(subset=['Entrez_Gene_Id'], keep='first')

def sortByEntrez(data, entrez_col = 'Entrez_Gene_Id'):
    
    # remove the hard assert -> drop the missing 
    n_missing = int(data[entrez_col].isna().sum()) if entrez_col in data.columns else 0
    if n_missing > 0:
        print(f"Dropping {n_missing} rows with missing {entrez_col}")
        data = data.dropna(subset=[entrez_col]).copy()
    
    # Sort by Entrez_Gene_Id and reset index
    data = data.sort_values(entrez_col).reset_index(drop=True)
    return data

################################################################# Too Slow
# # reference (min_key) with Entrez -> Hugo mapping
# ls_expr_df_clean = {}
# for key in ls_expr_df.keys():
#     # print(key)
#     df = ls_expr_df[key][0].copy()
#     # check_missingness(df)
#     # sort by entrez id
#     df_sorted = sortByEntrez(df, 'Entrez_Gene_Id')
    
#     # Merge on Entrez_Gene_Id, keep new Hugo symbol as a temporary column
#     df_merged = df_sorted.merge(
#         ref_hugo_entrez,
#         how='left', on='Entrez_Gene_Id',
#         suffixes=('', '_New')  # Adds '_New' to duplicate columns
#     )
#     # Drop old Hugo_Symbol
#     df_merged = df_merged.drop(columns=['Hugo_Symbol'])
#     # Rename Hugo_Symbol_New to Hugo_Symbol
#     df_merged = df_merged.rename(columns={'Hugo_Symbol_New': 'Hugo_Symbol'})
#     df_merged = df_merged[['Hugo_Symbol'] + [c for c in df_sorted.columns if c != 'Hugo_Symbol']]
#     # display(df_merged.head())
#     # check_missingness(df_merged)
    
#     # Store to a new dict
#     ls_expr_df_clean[key] = df_merged
# # print(ls_expr_df_clean.keys())
################################################################# 

# create a lookuptable
entrez_to_hugo = ref_hugo_entrez.set_index('Entrez_Gene_Id')['Hugo_Symbol']

ls_expr_df_clean = {}
for key in ls_expr_df.keys():
    df = ls_expr_df[key][0].copy()
    df_sorted = sortByEntrez(df, 'Entrez_Gene_Id')

    # FAST: first use the lookup table to fill, then use original hugo to fill
    mapped = df_sorted['Entrez_Gene_Id'].map(entrez_to_hugo)
    df_sorted['Hugo_Symbol'] = mapped.fillna(df_sorted['Hugo_Symbol'])

    ls_expr_df_clean[key] = df_sorted


#### Aggregate expression data (In each data, imputation or removal, normalization/transformation, feature selection, scale, merge)
### Clean & normalize each dataset separately
# def log2cpm_for_counts(X):
#     X = X.astype(float)
#     # negatives or small max => already log/normalized
#     xmin = float(np.nanmin(X.to_numpy()))
#     xmax = float(np.nanmax(X.to_numpy()))
#     if xmin < 0 or xmax < 50:
#         return X

#     libsize = X.sum(axis=0)
#     libsize = libsize.replace(0, np.nan)  # avoid divide-by-zero
#     X_cpm = X.div(libsize, axis=1) * 1e6
#     return np.log2(X_cpm + 1)
def log2p1_if_counts(X):
    """Much faster than CPM; good enough for this assignment."""
    X = X.astype(np.float32)
    xmin = float(np.nanmin(X.to_numpy()))
    xmax = float(np.nanmax(X.to_numpy()))
    if xmin < 0 or xmax < 50:
        return X
    return np.log2(X + 1.0)


ls_expr_clean_norm = {}

for key, df in ls_expr_df_clean.items():
    print(f"\n=== {key} ===")
    data = df.copy()
    data = data.set_index('Hugo_Symbol')

    # Remove duplicate genes (keep first)
    if data.index.duplicated().any():
        print(f"Removing {data.index.duplicated().sum()} duplicate genes")
        data = data[~data.index.duplicated(keep='first')]
    
    # Drop metadata
    meta_cols = ['Entrez_Gene_Id']
    X = data.drop(columns=[c for c in meta_cols if c in data.columns])
    X = X.apply(pd.to_numeric, errors="coerce")
    
    # Drop genes with all-NaN
    all_nan = X.isna().all(axis=1)
    X = X.loc[~all_nan]
    print(f"Dropped {all_nan.sum()} all-NaN genes")
    
    # Drop genes with >50% missing
    miss_rate = X.isna().mean(axis=1)
    drop_mask = miss_rate > 0.5
    X = X.loc[~drop_mask]
    print(f"Dropped {drop_mask.sum()} genes (>50% missing)")
    
    # Impute with gene *median* -> more robust
    X_np = X.to_numpy()
    row_median = np.nanmedian(X_np, axis=1, keepdims=True)
    X_np = np.where(np.isnan(X_np), row_median, X_np)
    X = pd.DataFrame(X_np, index=X.index, columns=X.columns)

    # Check if already log-scaled
    max_val = X.max().max()
    print(f"Max value before norm: {max_val:.2f}")
    
    # Normalize: log2 CPM
    X = log2p1_if_counts(X)
    print(f"Max value after norm: {float(np.nanmax(X.to_numpy())):.2f}")
    print(f"Shape: {X.shape}")
    
    ls_expr_clean_norm[key] = X

### Concat normalized datasets
common_genes = set.intersection(*[set(df.index) for df in ls_expr_clean_norm.values()])
print(f"\nCommon genes: {len(common_genes)}")

expr_list = []
for key, X in ls_expr_clean_norm.items():
    X_filtered = X.loc[list(common_genes)] #get the intersection
    expr_list.append(X_filtered)

merged_expr = pd.concat(expr_list, axis=1)
print(f"Merged expression: {merged_expr.shape}")

# Check for duplicate sample names
if merged_expr.columns.duplicated().any():
    print(f"NOTE: {merged_expr.columns.duplicated().sum()} duplicate sample names!")



##### read in mutation training data (Y) ####
def mut_skiprows_if_commented(fname):
    """Return 1 if first line starts with '#', else 0."""
    with open(fname, "rt", errors="ignore") as f: #read text mode
        first = f.readline()                      #first line
    return 1 if first.lstrip().startswith("#") else 0

ls_mut_df  = defaultdict(list)
for subpath in args.train_mut:
    # path=os.path.join(data_dir,subpath)
    path=subpath
    # load data
    skip = mut_skiprows_if_commented(path)  # METABRIC needs 1, TCGA typically 0
    df = read_expr(path, skiprows=skip)
    # store to a dict, using subpath as key
    ls_mut_df[subpath].append(df)
print(ls_mut_df.keys())

def get_mut_trainData(data, key, mut_type_col='Consequence', sample_col='Tumor_Sample_Barcode'):
    """
    After `for key, val in ls_mut_df.items(): get_mut_trainData(data = val[0], key)`,
    Returns gene x sample mutation matrix (True/False)
    """
    # Get all unique gene-sample pairs (both syn and non-syn)
    all_pairs = data[['Hugo_Symbol', sample_col]].drop_duplicates().copy()
    
    # Identify non-synonymous mutations
    is_nonsyn = ~data[mut_type_col].str.contains('synonymous_variant', na=False)
    nonsyn_pairs = data.loc[is_nonsyn, ['Hugo_Symbol', sample_col]].drop_duplicates()
    nonsyn_pairs['mutated'] = True
    
    # Merge: all pairs get False by default, non-syn pairs get True
    merged = all_pairs.merge(nonsyn_pairs, on=['Hugo_Symbol', sample_col], how='left')
    merged['mutated'] = merged['mutated'].fillna(False)
    # display(merged)
    
    # Pivot to matrix
    mut_matrix = merged.pivot(
        index='Hugo_Symbol',
        columns=sample_col,
        values='mutated'
    ).fillna(False) #those genes with no mutation in the samples
    
    return mut_matrix

# Store to a fresh dict
ls_mut_df_clean = {}
for key in ls_mut_df: 
    print(key)
    data = ls_mut_df[key][0].copy()
    cleaned = get_mut_trainData(data = data, key = key)
    # display(cleaned)
    ls_mut_df_clean[key] = cleaned


### Aggregate mutation data
mut_list = [mut for mut in ls_mut_df_clean.values()]
merged_mut = pd.concat(mut_list, axis=1).fillna(False) #in case some genes not in all samples
print(f"Merged mutation: {merged_mut.shape}")



#### Align both data ####
# display(merged_expr)
# display(merged_mut)
common_genes = merged_expr.index.intersection(merged_mut.index)
common_samples = merged_expr.columns.intersection(merged_mut.columns)
print(f"Common genes: {len(common_genes)}")
print(f"Common samples: {len(common_samples)}")

merged_expr = merged_expr.loc[common_genes, common_samples]
merged_mut = merged_mut.loc[common_genes, common_samples]

print(f"Aligned expression: {merged_expr.shape}")
print(f"Aligned mutation: {merged_mut.shape}")



#### Prepare test data ####
test = read_expr(args.test_expr)

test_expr = test.copy().set_index('Hugo_Symbol')
test_expr = test_expr.drop(columns=['Entrez_Gene_Id'], errors='ignore')

#### Same as processing training data ####
# Remove duplicates
if test_expr.index.duplicated().any():
    test_expr = test_expr[~test_expr.duplicated(keep='first')]

# Filter to common genes
common_genes_final = merged_expr.index.intersection(test_expr.index)
test_expr = test_expr.loc[common_genes_final]

# convert to numerical
test_expr = test_expr.apply(pd.to_numeric, errors="coerce")

# Impute
X_test_np = test_expr.values
row_median = np.nanmedian(X_test_np, axis=1, keepdims=True)
mask = np.isnan(X_test_np)
X_test_np = np.where(mask, row_median, X_test_np)
test_expr = pd.DataFrame(X_test_np, index=test_expr.index, columns=test_expr.columns)

# Normalize if needed
test_expr = log2p1_if_counts(test_expr)
test_X = test_expr.T  # samples × genes

# Remove duplicate columns
if test_X.columns.duplicated().any():
    print(f"Removing {test_X.columns.duplicated().sum()} duplicate columns in test data")
    test_X = test_X.loc[:, ~test_X.columns.duplicated()]

print(f"Test data shape: {test_X.shape}")



#### Feature selection (Regularization) && scale -> Logistic Regression with Elastic Net (w.GridSearch) ####
## within the loop -> too slow
X_train_full = merged_expr.T  # samples x genes

# Align train/test to same gene columns
common_cols = X_train_full.columns.intersection(test_X.columns)
X_train_aligned = X_train_full.loc[:, common_cols]
test_X_aligned = test_X.loc[:, common_cols]
print(f"Common features (train/test): {len(common_cols)}")

# Variance filter
var_selector = VarianceThreshold(threshold=0.1)
var_selector.fit(X_train_aligned)
kept_cols = X_train_aligned.columns[var_selector.get_support()]
print(f"After variance filter: {len(kept_cols)} features")

# Select the top 2000 var
max_features = 2000
if len(kept_cols) > max_features:
    # manually calcualte var and find top var
    v = X_train_aligned.loc[:, kept_cols].var(axis=0).to_numpy()
    top = np.argpartition(v, -max_features)[-max_features:] #cut at the last nth
    kept_cols = kept_cols[top].sort_values()
    print(f"Capped to top {max_features} features by variance")


# Impute + scale for train/test together
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_var = X_train_aligned.loc[:, kept_cols].to_numpy(dtype=np.float32) #faster than 64
X_train_imp = imputer.fit_transform(X_train_var)
X_train_scaled = scaler.fit_transform(X_train_imp)

X_test_var = test_X_aligned.loc[:, kept_cols].to_numpy(dtype=np.float32) #faster than 64
X_test_imp = imputer.transform(X_test_var)
X_test_scaled = scaler.transform(X_test_imp)


target_genes = list(args.gene)
for gene in target_genes:
    print(f"\n=== {gene} ===")
    
    # If gene not in mutation data, still produce an output file (constant prob)
    if gene not in merged_mut.index:
        print("Not in mutation data -> writing constant p=0.0")
        with open(f"{args.output_prefix}_{gene}.txt", "wt") as out:
            for name in sorted(test_X.index):
                out.write(f"{name} {0.0:.5f}\n")
        continue
    
    # Prepare
    Y_train = merged_mut.loc[gene].reindex(X_train_aligned.index).astype(int)


    # # Define models to compare
    # scale_pos_weight = (Y_train == 0).sum() / (Y_train == 1).sum()
    # models = {
    #     'ElasticNet': LogisticRegression(
    #         penalty='elasticnet', # deprecated in the newest ver, but required here
    #         solver='saga', l1_ratio=0.5, C=0.1,
    #         max_iter=3000, tol=1e-3, class_weight='balanced', random_state=1234
    #     ),
    #     'L1': LogisticRegression(
    #         penalty='l1', #same reason
    #         solver='saga', C=0.1, #l1_ratio=1,
    #         max_iter=3000, tol=1e-3, class_weight='balanced', random_state=1234
    #     ),
    #     'RandomForest': RandomForestClassifier(
    #         n_estimators=200, max_depth=10, min_samples_leaf=5,
    #         class_weight='balanced', random_state=1234, n_jobs=-1
    #     ),
    #     'XGBoost': XGBClassifier(
    #         n_estimators=100, max_depth=3, learning_rate=0.1,
    #         scale_pos_weight=scale_pos_weight,
    #         random_state=1234, n_jobs=-1, eval_metric='auc'
    #     )
    # }

    # # Quick CV to find best model
    # best_model_name = None
    # best_score = 0
    # for name, model in models.items():
    #     scores = cross_val_score(model, X_scaled, Y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    #     mean_score = scores.mean()
    #     print(f"  {name}: AUC = {mean_score:.3f} ± {scores.std():.3f}")
        
    #     if mean_score > best_score:
    #         best_score = mean_score
    #         best_model_name = name
    
    # print(f"  Best model: {best_model_name} (AUC={best_score:.3f})")
    
    # # Train best model on full data
    # best_model = models[best_model_name]
    # best_model.fit(X_scaled, Y_train)
    
    # # Store results
    # results[gene] = {
    #     'best_model': best_model_name,
    #     'best_auc': best_score
    # }
    

    # # Predict - apply same transformations to test
    # X_test_var = test_X_aligned.loc[:, kept_cols].to_numpy()
    # X_test_scaled = scaler.transform(X_test_var)
    # P_mutated = best_model.predict_proba(X_test_scaled)[:, 1] #P(class=1) = P(mutated)
    # # print(P_mutated)



    # # Grid search
    # model = LogisticRegression(
    #     penalty='elasticnet', # deprecated in the newest ver, but required here
    #     solver='saga',
    #     max_iter=3000, tol=1e-3, class_weight='balanced', random_state=1234, n_jobs=-1,
    # )

    # # imbalanced
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    
    # grid_search = GridSearchCV(
    #     model,
    #     param_grid,
    #     cv=cv,
    #     scoring='roc_auc',
    #     n_jobs=-1,
    #     refit=True  # refits best model on ALL training data
    # )
    # grid_search.fit(X_scaled, Y_train)
    
    # # Results
    # print(f"Best params: {grid_search.best_params_}")
    # print(f"Best AUC: {grid_search.best_score_:.3f}")
    
    # best_model = grid_search.best_estimator_
    # n_features = (best_model.coef_ != 0).sum()
    # print(f"Features used: {n_features}")
    
    # # Store results
    # results[gene] = {
    #     'best_params': grid_search.best_params_,
    #     'best_auc': grid_search.best_score_,
    #     'n_features': n_features
    # }
    
    # # Predict - apply same transformations to test
    # X_test_var = test_X_aligned.loc[:, kept_cols].to_numpy()
    # X_test_scaled = scaler.transform(X_test_var)
    # P_mutated = best_model.predict_proba(X_test_scaled)[:, 1] #P(class=1) = P(mutated)
    # # print(P_mutated)


    # Final model (skip GridSearchCV) using chosen hyperparameters
    C_best = 0.01 #based on gridSearch res
    l1_ratio_best = 0.5
    
    best_model = LogisticRegression(
        # penalty="elasticnet",
        solver="saga",
        C=C_best,
        l1_ratio=l1_ratio_best,
        max_iter=500,
        tol=1e-3,
        class_weight="balanced",
        random_state=1234,
        # n_jobs=-1,
    )
    best_model.fit(X_train_scaled, Y_train.to_numpy())
    
    # Predict - apply same transformations to test
    P_mutated = best_model.predict_proba(X_test_scaled)[:, 1]


    # Output
    out = open('%s_%s.txt'%(args.output_prefix,gene),'wt')
    for (name,p) in sorted(zip(test_X_aligned.index, P_mutated)):
        out.write('%s %.5f\n'%(name,p))
    out.close()


# # Summary
# print(f"\n{'='*50}")
# print("Summary:")
# # for gene, res in results.items():
# #     print(f"{gene}: {res['best_model']}, AUC={res['best_auc']:.3f}")
# for gene, res in results.items():
#     print(f"{gene}: AUC={res['best_auc']:.3f}, C={res['best_params']['C']}, l1_ratio={res['best_params']['l1_ratio']}, features={res['n_features']}")

