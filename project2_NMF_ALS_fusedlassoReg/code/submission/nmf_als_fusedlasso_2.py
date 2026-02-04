import numpy as np
import argparse
import pyBigWig
from sklearn.utils.extmath import randomized_svd
from glob import glob
from numba import njit #jit without python

parser = argparse.ArgumentParser(description='Compute NMF to find latent variables with high correlation to transcription start sites')
parser.add_argument('-b','--bw_dir',help='Directory containing bigWig file(s)',required=True)
parser.add_argument('-c','--chromosome_number',help='Which chromosome to get the values for',required=True)
parser.add_argument('-s','--start_pos',type=int,help='Position to start reading from')
parser.add_argument('-e','--end_pos',type=int,help='Position to stop reading at')
parser.add_argument('-k',type=int,default=10,help='Number of latent vectors')
parser.add_argument('-o','--output_file',help='Output file of latent factors matrix.',required=True)

args = parser.parse_args()

files = sorted(glob(f'{args.bw_dir}/*.bw'))

# Place holder for the matrix
row = args.end_pos - args.start_pos
col = len(files)
Y = np.empty((row, col), dtype=np.float32)
# print(Y)
########### Build the matrix ###########
for idx, fname in enumerate(files):
    print('\n', idx, fname)
    # IMPLEMENT -- use pyBigWig to access the .bw files
    	# use args.chromosome_number to access the correct chromosome
    	# use args.start_pos and args.end_pos for the start and end position of the chromosome
    with pyBigWig.open(fname) as bw:
        f_val = bw.values(args.chromosome_number, args.start_pos, args.end_pos, numpy=True)
        f_val = f_val.astype(np.float32, copy=False) #change type
        #deal with NaN, apply any other transformations
        np.nan_to_num(f_val, copy=False, #in place
                      nan=0.0, posinf=0.0, neginf=0.0)
        #non-negative transformation
        f_val[f_val < 0] = 0
        #assign in place
        Y[:, idx] = f_val
print(Y.shape)

########### preprocessing to improve correlation ###########
Y = np.log1p(Y).astype(np.float32, copy=False)  # compress huge peaks
col_norm = np.linalg.norm(Y, axis=0) + 1e-12    # normalize different tracks
Y = (Y / col_norm).astype(np.float32, copy=False)


########### setup proximity operator using the provided code ###########
# Implement numba
@njit(cache=True)  # nopython mode by default; cache=True avoids recompiling between runs
def pyprox_dp(y, lam): #return theta - the smoothed input vector
    n = len(y)
    if n == 0:
        return y.copy()

    theta = np.zeros_like(y)    
    # Take care of a few trivial cases
    if n == 1 or lam == 0:
        for i in range(n):
            theta[i] = y[i]
        return theta
            
  # These are used to store the derivative of the
  # piecewise quadratic function of interest
    afirst = 0.0
    alast = 0.0
    bfirst = 0.0
    blast = 0.0
    
    x = np.zeros(2*n, dtype=y.dtype)
    a = np.zeros(2*n, dtype=y.dtype)
    b = np.zeros(2*n, dtype=y.dtype)
  
    l = 0
    r = 0

  # These are the knots of the back-pointers
    tm = np.zeros(n-1, dtype=y.dtype)
    tp = np.zeros(n-1, dtype=y.dtype)

  # We step through the first iteration manually
    tm[0] = -lam+y[0]
    tp[0] = lam+y[0]
    l = n-1
    r = n
    x[l] = tm[0]
    x[r] = tp[0]
    a[l] = 1
    b[l] = -y[0]+lam
    a[r] = -1
    b[r] = y[0]+lam
    afirst = 1
    bfirst = -lam-y[1]
    alast = -1
    blast = -lam+y[1]

  # Now iterations 2 through n-1
    lo = 0
    hi = 0
    alo = 0.0
    blo = 0.0
    ahi = 0.0
    bhi = 0.0
    
    for k in range(1,n-1):
        # Compute lo: step up from l until the
        # derivative is greater than -lam
        alo = afirst
        blo = bfirst
        for lo in range(l,r+1):            
            if alo*x[lo]+blo > -lam: break

            alo += a[lo]
            blo += b[lo]
        else:
            lo = r+1
        
        # Compute the negative knot

        tm[k] = (-lam-blo)/alo
        l = lo-1
        x[l] = tm[k]

        # Compute hi: step down from r until the
        # derivative is less than lam
        ahi = alast
        bhi = blast
        for hi in range(r,l-1,-1):
            if -ahi*x[hi]-bhi < lam: break
            ahi += a[hi]
            bhi += b[hi]
        else:
            hi = l-1        

        # Compute the positive knot
        tp[k] = (lam+bhi)/(-ahi)
        r = hi+1
        x[r] = tp[k]

        # Update a and b
        a[l] = alo
        b[l] = blo+lam
        a[r] = ahi
        b[r] = bhi+lam

        afirst = 1
        bfirst = -lam-y[k+1]
        alast = -1
        blast = -lam+y[k+1]
        
  # Compute the last coefficient: this is where 
  # the function has zero derivative

    alo = afirst
    blo = bfirst
    for lo in range(l, r+1):
        if alo*x[lo]+blo > 0: break
        alo += a[lo]
        blo += b[lo]
  
    theta[n-1] = -blo/alo

  # Compute the rest of the coefficients, by the
  # back-pointers
    for k in range(n-2,-1,-1):
        if theta[k+1]>tp[k]:
            theta[k] = tp[k]
        elif theta[k+1]<tm[k]:
            theta[k] = tm[k]
        else:
            theta[k] = theta[k+1]
  

    return theta

# Ensure numba is applied to contiguous array
def fused_lasso(y, lam):
    y = np.ascontiguousarray(y, dtype=np.float32)
    return pyprox_dp(y, np.float32(lam))

# Use bins to the rows or the genomic regions to speed up -> later recover the dimension by replication the same value within each bin
def fused_lasso_binned(y, lam, bin_size=20):
    if lam <= 0.0:
        return y
    if bin_size <= 1 or y.size < bin_size:
        return fused_lasso(y, lam)

    y = np.ascontiguousarray(y, dtype=np.float32)
    # length of original vector
    n = y.size
    # number of bins
    m = (n + bin_size - 1) // bin_size
    # padding to correspond to the binned range
    pad = m * bin_size - n

    if pad > 0:
        yp = np.empty(m * bin_size, dtype=np.float32)
        yp[:n] = y
        yp[n:] = y[-1]  # edge pad - last element value
    else:
        yp = y

    # reshape and get the mean over the bin
    yb = yp.reshape(m, bin_size).mean(axis=1).astype(np.float32, copy=False)
    # compute over the bin
    tb = fused_lasso(yb, lam)
    return np.repeat(tb, bin_size)[:n] # trim the extra values


########### NMF pipeline ###########
# Initialize
def init_H(Y,k,SEED=1221):
	# initialize H - nneg!
	# can be a random initialization or using the randomized_svd from sklearn
    _, D, Vt = randomized_svd(M=Y, n_components=k, random_state=SEED)
    
    # Use scaled version of loading H (give better initialization signal)
    #   convert D from (k,) to (k,1) -> strech to match Vt (k, n_features) -> (k, n_feature) -> element-wise
    H = (D[:,None] * Vt).astype(np.float32, copy=False)
    
    # nneg transformation
    H[H<0] = 0.0
    H += np.float32(1e-6) # random number to prevent exact zero

    return H

# NMF
def NMF_FL(Y, k, num_iter=50, l2penalty=1, fl_lambda=1, tol=1e-4):
    H = init_H(Y,k,SEED=1221)
    print('\n','Completed randomized_svd H initialization...')

    # Create diagonal offset D
    #   if l2penalty is small all this does is make the matrix invertible
    D = np.eye(k, dtype=np.float32) * np.float32(l2penalty)
    Y = np.asarray(Y, dtype=np.float32)

    # warm up JIT once (avoids compile cost inside iteration 0)
    _ = fused_lasso(np.zeros(16, dtype=np.float32), np.float32(fl_lambda))

    # store error and improvment for early stop
    ynorm = np.linalg.norm(Y, 'fro') + 1e-12
    prev_rel_err = None
    ## store bad impr patience/steps before break
    patience = 10
    bad = 0

    ## speed up - binning & less usage of fl
    L = Y.shape[0]
    if L >= 1000000:
        bin_size = 50
        fl_every = 5
    else:
        bin_size = 20
        fl_every = 3
    
    for n in range(num_iter):
        if n % 50 == 0:
            print('\n', f'==== Starting iteration {n} ====')
        # Update W
        # $W \leftarrow Y H^T (H H^T + D)^{-1}$
        A = (H @ H.T + D) #(k,k)
        B = (Y @ H.T)     #(nrow,k)
        
        # W = B @ np.linalg.inv(A) # slow indicated by GPT
        W = np.linalg.solve(A.T, B.T).T.astype(np.float32, copy=False)
        # np.linalg.solve() above is the same as A^T @ X = B^T 
        # -> X = (A^T)^(-1) @ B^T 
        # -> Xt= ((A^T)^(-1) @ B^T)^T
        #      = B @ inv(A.T).T
        #      = B @ inv(A)

        # Set negative elements of W to 0
        W[W < 0.0] = 0.0

        # apply fused lasso
        if fl_lambda > 0.0 and (n % fl_every == 0):
            for j in range(k): #cols of W
                # W[:, j] = fused_lasso(W[:, j], fl_lambda)
                W[:, j] = fused_lasso_binned(W[:, j], fl_lambda, bin_size=bin_size)
            W[W < 0.0] = 0.0
        
        # Update H
        C = (W.T@W) + D
        E = (W.T@Y)
        # H = inv(C)@E
        H = np.linalg.solve(C, E).astype(np.float32, copy=False)
        # C @ X = E
        # X = inv(C)@E

        # Set negative elements of H to 0
        H[H < 0.0] = 0.0

        #early stopping?
        rel_err = np.linalg.norm(Y - W@H, 'fro') / ynorm #Frobenius norm
        if prev_rel_err is None:
            impr = np.nan
        else:
            impr = (prev_rel_err-rel_err) / max(prev_rel_err, 1e-12) #prevent 0 div; we want impr > 0

        #log
        if n % 50 == 0:
            print(f"rel_err={rel_err:.6f} impr={impr:.3e}")

        #eval
        if prev_rel_err is not None and impr <= 1e-6:
            bad += 1
        else:
            bad = 0
        
        if bad >= patience:
            print(f"Stop: no improvement for {patience} iterations (iter {n})")
            break

        if prev_rel_err is not None and impr > 0.0 and impr < tol:
            print(f"Converged at iteration {n}")
            break

        #update error
        prev_rel_err = rel_err

    return W, H


# ########### RUN1 - Hyperparameter Tuning Using Grid Search (comment out later) ###########
# # k, (num_iter,) l2penalty, and fl_lambda are all hyperparameters that should be tuned to maximize correlation with genes

# def load_realAnnot(annotation_bw_path, chrom, start, end): 
#     with pyBigWig.open(annotation_bw_path) as bw:
#         g = bw.values(chrom, start, end, numpy=True)
#     g = g.astype(np.float32, copy=False)
#     np.nan_to_num(g, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
#     return g

# def eval_pearson_corr(a, b):
#     a = np.asarray(a, dtype=np.float32)
#     b = np.asarray(b, dtype=np.float32)
#     if a.shape[0] != b.shape[0]:
#         raise ValueError("Vectors must have the same length")
    
#     a_bar = a - a.mean()
#     b_bar = b - b.mean()
#     sasb = (np.linalg.norm(a_bar) * np.linalg.norm(b_bar)) + 1e-12
#     r = float(np.dot(a_bar, b_bar) / sasb)
#     return r

# def find_maxAbs_pearson_corr(W, g):
#     best = -1.0
#     best_feature = -1
#     for j in range(W.shape[1]):
#         r = eval_pearson_corr(W[:, j], g)
#         if abs(r) > best:
#             best = abs(r)
#             best_j = j
#     return best, best_j


# ##### Grid Search + Eval #####
# g = load_realAnnot('/ihome/hpark/til177/GitHub/cobb2060-2026s/Data_cobb2060/proj2/Annotation_hg18.Gene.bw', args.chromosome_number, args.start_pos, args.end_pos) 

# l2_grid = [1e-6, 1e-4, 1e-2, 1e-1, 1.0]
# fl_grid = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]

# best = (-1.0, None, None)

# for l2 in l2_grid:
#     for fl in fl_grid:
#         W, H = NMF_FL(Y, args.k, num_iter=100, l2penalty=l2, fl_lambda=fl, tol=1e-4)
#         score, best_j = find_maxAbs_pearson_corr(W, g)
#         print(f"l2={l2:g} fl={fl:g} score={score:.4f} best_j={best_j}")
#         if score > best[0]:
#             best = (score, l2, fl)

# print("BEST:", best)

#   BEST: (0.6943908209525683, 0.01, 0.3)

########### RUN2 - Production ###########
W, H = NMF_FL(Y, args.k, num_iter=100, l2penalty=0.01, fl_lambda=0.3, tol=1e-4)

np.save(args.output_file, W, allow_pickle=True)

# You already have a job in the queue that was submitted 2026-02-03 23:57:56