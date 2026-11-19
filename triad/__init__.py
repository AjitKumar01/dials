"""TRIAD: bundle completion from three gated relations (content, labels, co-occurrence graph)."""
import os
import sys
import warnings

# One BLAS thread, set before numpy loads: the truncated SVDs of the item features are bit-reproducible only at
# a fixed thread count, and Apple Accelerate cannot be limited after it starts.
if 'numpy' in sys.modules and os.environ.get('VECLIB_MAXIMUM_THREADS')!='1':
    warnings.warn('numpy was imported before triad without VECLIB_MAXIMUM_THREADS=1; SVD caches may not be bit-reproducible')
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key,'1')
