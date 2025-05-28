#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 12:47:31 2025

@author: samy
"""

import numpy as np
from scipy.stats import zscore, rankdata
from numba import njit, prange

# =============================================================================
# =============================================================================
# # Optimization functions
# =============================================================================
# =============================================================================


@njit(parallel=True)
def fast_corrcoef_numba_parallel(X):
    """
    Parallel Numba version of Pearson correlation matrix.
    X shape: (T, N) → T timepoints, N features
    Returns: (N, N) correlation matrix
    """
    T, N = X.shape
    means = np.empty(N)
    stds = np.empty(N)
    corr = np.empty((N, N))

    # Compute means (parallel)
    for i in prange(N):
        s = 0.0
        for t in range(T):
            s += X[t, i]
        means[i] = s / T

    # Compute stds (parallel)
    for i in prange(N):
        s = 0.0
        for t in range(T):
            diff = X[t, i] - means[i]
            s += diff * diff
        stds[i] = (s / (T - 1)) ** 0.5

    # Compute correlation matrix (parallel upper triangle)
    for i in prange(N):
        for j in range(i, N):
            s = 0.0
            for t in range(T):
                s += (X[t, i] - means[i]) * (X[t, j] - means[j])
            cov = s / (T - 1)
            c = cov / (stds[i] * stds[j])
            corr[i, j] = c
            corr[j, i] = c

    return corr


@njit(fastmath=True)
def fast_corrcoef_numba(X):
    """
    Numba-accelerated Pearson correlation matrix for 2D array (observations x features).
    Equivalent to fast_corrcoef(X), assumes columns are variables (features).
    """
    T, N = X.shape
    out = np.empty((N, N))
    
    # Compute means and stds manually
    means = np.empty(N)
    stds = np.empty(N)
    for i in range(N):
        s = 0.0
        for t in range(T):
            s += X[t, i]
        means[i] = s / T

    for i in range(N):
        s = 0.0
        for t in range(T):
            diff = X[t, i] - means[i]
            s += diff * diff
        stds[i] = (s / (T - 1)) ** 0.5

    # Compute correlation matrix
    for i in range(N):
        for j in range(i, N):
            s = 0.0
            for t in range(T):
                s += (X[t, i] - means[i]) * (X[t, j] - means[j])
            cov = s / (T - 1)
            corr = cov / (stds[i] * stds[j])
            out[i, j] = corr
            out[j, i] = corr

    return out

# @njit(fastmath=True)
def fast_corrcoef(ts):
    """
    Numba-accelerated Pearson correlation matrix using z-score and dot product.
    ts: np.ndarray (timepoints, features)
    """
    n_samples, n_features = ts.shape
    mean = np.mean(ts, axis=0)
    std = np.std(ts, axis=0, ddof=1)
    # Avoid division by zero for constant columns
    std[std == 0] = 1.0
    z = (ts - mean) / std
    return np.dot(z.T, z) / (n_samples - 1)

# =============================================================================
# Optimized speed computation functions
# =============================================================================

@njit(fastmath=True)
def pearson_speed(fc1, fc2):
    """
    Optimized Pearson correlation-based speed computation using Numba.
    
    Parameters:
        fc1, fc2: 2D arrays (n_pairs, n_frames)
    
    Returns:
        speeds: 1D array (n_frames,) of speed values
    """
    n_pairs, n_frames = fc1.shape
    speeds = np.empty(n_frames)
    
    for t in range(n_frames):
        fc1_t = fc1[:, t]
        fc2_t = fc2[:, t]
        
        # Compute means
        mean1 = np.mean(fc1_t)
        mean2 = np.mean(fc2_t)
        
        # Center the data
        fc1_centered = fc1_t - mean1
        fc2_centered = fc2_t - mean2
        
        # Compute correlation
        numerator = np.sum(fc1_centered * fc2_centered)
        denom1 = np.sum(fc1_centered * fc1_centered)
        denom2 = np.sum(fc2_centered * fc2_centered)
        
        denominator = np.sqrt(denom1 * denom2)
        
        if denominator > 1e-12:
            correlation = numerator / denominator
        else:
            correlation = 0.0
            
        speeds[t] = 1.0 - correlation
    
    return speeds

@njit(fastmath=True)
def cosine_speed(fc1, fc2):
    """
    Optimized cosine similarity-based speed computation using Numba.
    
    Parameters:
        fc1, fc2: 2D arrays (n_pairs, n_frames)
    
    Returns:
        speeds: 1D array (n_frames,) of speed values
    """
    n_pairs, n_frames = fc1.shape
    speeds = np.empty(n_frames)
    
    for t in range(n_frames):
        fc1_t = fc1[:, t]
        fc2_t = fc2[:, t]
        
        # Compute dot product and norms
        dot_product = np.sum(fc1_t * fc2_t)
        norm1 = np.sqrt(np.sum(fc1_t * fc1_t))
        norm2 = np.sqrt(np.sum(fc2_t * fc2_t))
        
        denominator = norm1 * norm2
        
        if denominator > 1e-12:
            cosine_sim = dot_product / denominator
        else:
            cosine_sim = 0.0
            
        speeds[t] = 1.0 - cosine_sim
    
    return speeds

def spearman_speed(fc1, fc2):
    """
    Spearman correlation-based speed computation.
    Note: Uses scipy.stats.rankdata which is not compatible with Numba.
    
    Parameters:
        fc1, fc2: 2D arrays (n_pairs, n_frames)
    
    Returns:
        speeds: 1D array (n_frames,) of speed values
    """
    n_pairs, n_frames = fc1.shape
    speeds = np.empty(n_frames)
    
    for t in range(n_frames):
        fc1_ranks = rankdata(fc1[:, t])
        fc2_ranks = rankdata(fc2[:, t])
        
        # Compute Pearson correlation of ranks
        mean1 = np.mean(fc1_ranks)
        mean2 = np.mean(fc2_ranks)
        
        fc1_centered = fc1_ranks - mean1
        fc2_centered = fc2_ranks - mean2
        
        numerator = np.sum(fc1_centered * fc2_centered)
        denom1 = np.sum(fc1_centered * fc1_centered)
        denom2 = np.sum(fc2_centered * fc2_centered)
        
        denominator = np.sqrt(denom1 * denom2)
        
        if denominator > 1e-12:
            correlation = numerator / denominator
        else:
            correlation = 0.0
            
        speeds[t] = 1.0 - correlation
    
    return speeds

# =============================================================================
# Vectorized einsum-based speed computation (most optimized)
# =============================================================================

def pearson_speed_vectorized(fc1, fc2):
    """
    Fully vectorized Pearson correlation speed computation using einsum.
    This is the most optimized version for large datasets.
    
    Parameters:
        fc1, fc2: 2D arrays (n_pairs, n_frames)
    
    Returns:
        speeds: 1D array (n_frames,) of speed values
    """
    # Center the data
    fc1_centered = fc1 - np.mean(fc1, axis=0, keepdims=True)
    fc2_centered = fc2 - np.mean(fc2, axis=0, keepdims=True)
    
    # Compute correlations using einsum (fully vectorized)
    numerator = np.einsum('ij,ij->j', fc1_centered, fc2_centered)
    fc1_ss = np.einsum('ij,ij->j', fc1_centered, fc1_centered)
    fc2_ss = np.einsum('ij,ij->j', fc2_centered, fc2_centered)
    
    denominator = np.sqrt(fc1_ss * fc2_ss)
    
    # Handle numerical edge cases
    valid_mask = denominator > np.finfo(float).eps
    correlations = np.zeros(fc1.shape[1])
    correlations[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    
    return 1.0 - correlations

def cosine_speed_vectorized(fc1, fc2):
    """
    Fully vectorized cosine similarity speed computation using einsum.
    
    Parameters:
        fc1, fc2: 2D arrays (n_pairs, n_frames)
    
    Returns:
        speeds: 1D array (n_frames,) of speed values
    """
    # Compute cosine similarity using einsum
    numerator = np.einsum('ij,ij->j', fc1, fc2)
    fc1_norms = np.sqrt(np.einsum('ij,ij->j', fc1, fc1))
    fc2_norms = np.sqrt(np.einsum('ij,ij->j', fc2, fc2))
    
    denominator = fc1_norms * fc2_norms
    
    # Handle numerical edge cases  
    valid_mask = denominator > np.finfo(float).eps
    cosine_similarities = np.zeros(fc1.shape[1])
    cosine_similarities[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    
    return 1.0 - cosine_similarities
