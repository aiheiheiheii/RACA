"""TNSE compact synthetic model descriptor data, TMC-compatible group API.

This module is read-only on disk. No algorithm should use labels or prototype
as oracle information while making verification decisions.
"""
from __future__ import annotations
import json
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
N_VALUES = tuple(range(1000, 5001, 500))
CR_VALUES = tuple(i / 2 for i in range(2, 11))  # 1.0 .. 5.0
D_VALUES = tuple(range(10, 51, 5))
H_VALUES = tuple(i / 10 for i in range(11))
SUBTYPES = ('common_normal', 'high_personalized_normal', 'severe_corrupt', 'mild_corrupt')
BANK_COUNTS = (4800, 1200, 2100, 900)
SUBTYPE_NAME_TNSE = ('common_normal', 'high_heterogeneity_normal', 'severe_corrupt', 'mild_corrupt')


def _hindex(h: float) -> int:
    i = round(float(h) * 10)
    if not (0 <= i <= 10 and np.isclose(float(h), i / 10, atol=1e-9)):
        raise ValueError('H must be one of {0, 0.1, ..., 1.0}')
    return i


def _round_count(x: float) -> int:
    return int(Decimal(str(x)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


@lru_cache(maxsize=4)
def _bank(root_text: str, hindex: int):
    path = Path(root_text) / 'banks' / f'bank_H_{hindex:02d}.npz'
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def _permutation(n: int, seed: int, subtype: int) -> np.ndarray:
    # Each subtype has a distinct deterministic permutation; paired across H,d,N,cr.
    return np.random.default_rng(np.random.SeedSequence([seed, 99173, subtype])).permutation(n)


def _to_dimension(X50: np.ndarray, theta: np.ndarray, prototype50: np.ndarray, d: int):
    """Same intrinsic angular severity across different descriptor dimensions.

    d-dimensional descriptor uses a deterministic subspace of the canonical
    50D attributes, with re-orthogonalization and norm conservation.
    This is descriptor construction, NOT a claim of direct truncation.
    """
    p = np.asarray(prototype50[:d], dtype=np.float64)
    p = p / np.linalg.norm(p)
    if d == 50:
        X = X50.copy()
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X.astype(np.float32), p.astype(np.float32)
    x = X50.astype(np.float64)
    p50 = prototype50.astype(np.float64)
    u = x[:, :d] - np.cos(theta)[:, None] * p50[None, :d]
    u -= (u @ p)[:, None] * p[None, :]
    norms = np.linalg.norm(u, axis=1)
    if np.any(norms < 1e-7):
        raise RuntimeError(f'Degenerate projected descriptor at d={d}')
    u /= norms[:, None]
    y = np.cos(theta)[:, None] * p[None, :] + np.sin(theta)[:, None] * u
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y.astype(np.float32), p.astype(np.float32)


def load_group(dataset_root=ROOT, n=3000, corruption_rate_percent=3.0,
               dim=30, H=0.5, seed=2026):
    """Return TMC-style dict, without materializing 729 redundant group NPZs.

    Every group is a reproducible sample from the materialized H-indexed banks.
    binary_label/subtype_label are EVALUATION LABELS ONLY.
    """
    root = Path(dataset_root).resolve()
    n = int(n)
    dim = int(dim)
    cr = float(corruption_rate_percent)
    if n < 1 or n > 5000:
        raise ValueError('n must be 1..5000 (published grid: 1000..5000 step 500)')
    if dim not in D_VALUES:
        raise ValueError(f'dim must be in {D_VALUES}')
    if not (0.0 <= cr <= 50.0):
        raise ValueError('corruption_rate_percent must be in [0, 50]')
    h = _hindex(H)
    bank = _bank(str(root), h)
    n_cor = _round_count(n * cr / 100.0)
    n_nor = n - n_cor
    n_high = _round_count(n_nor * 0.20)
    n_common = n_nor - n_high
    n_mild = _round_count(n_cor * 0.20)
    n_severe = n_cor - n_mild
    quotas = (n_common, n_high, n_severe, n_mild)
    if any(q > c for q,c in zip(quotas, BANK_COUNTS)):
        raise ValueError('Requested group exceeds available subtype bank')
    offsets = np.cumsum((0,)+BANK_COUNTS)
    ids = []
    for j, q in enumerate(quotas):
        indices = _permutation(BANK_COUNTS[j], int(seed), j)[:q]
        ids.append((indices + offsets[j]).astype(np.int32))
    ids = np.concatenate(ids)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7727, n, int(round(cr*10))]))
    ids = ids[rng.permutation(len(ids))]
    X, p = _to_dimension(bank['X_50'][ids], bank['theta'][ids].astype(np.float64),
                         bank['prototype_50'], dim)
    sim = (X.astype(np.float64) @ p.astype(np.float64)).astype(np.float32)
    y = bank['binary_label'][ids].copy()
    subtype = bank['subtype_label'][ids].copy()
    return {
        'X': X, 'binary_label': y, 'subtype_label': subtype,
        'cosine_to_prototype': sim, 'prototype': p,
        'tau_lower': 0.997, 'tau_upper': 1.0,
        'sample_id': ids, 'N': n, 'corruption_rate_percent': cr,
        'dimension': dim, 'heterogeneity_H': float(H), 'seed': int(seed),
        'counts': dict(zip(SUBTYPES, [int(z) for z in quotas])),
    }


def enumerate_configs():
    for n in N_VALUES:
        for cr in CR_VALUES:
            for d in D_VALUES:
                yield (n, cr, d)


def cosine_pairwise(X: np.ndarray, Y: np.ndarray | None = None):
    a = np.array(X, dtype=np.float64, copy=True)
    b = a if Y is None else np.array(Y, dtype=np.float64, copy=True)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.clip(a @ b.T, -1.0, 1.0)


def draw_candidate_set(group: dict, K=50, q_C=0.10, seed=2026,
                       use_severe_only=False, use_common_only=False):
    """Controlled ground-truth evaluation sampling, not a proposed selector.

    q_C sets composition using integer counts, which is not always an exact
    fraction for K=50; always report effective_ratio.
    """
    K = int(K)
    sub = group['subtype_label']
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1147]))
    # Exact percentages in 5% steps are impossible in a K=50 set at odd steps.
    # Stochastic rounding gives the nominal fraction in expectation across seeds.
    expected_bad = K * float(q_C)
    low = int(np.floor(expected_bad))
    n_bad = low + int(rng.random() < (expected_bad - low))
    nor = np.flatnonzero(sub == 0) if use_common_only else np.flatnonzero(group['binary_label'] == 0)
    bad = np.flatnonzero(sub == 2) if use_severe_only else np.flatnonzero(group['binary_label'] == 1)
    if len(nor) < K-n_bad or len(bad) < n_bad:
        raise ValueError('Insufficient candidate normal/corrupt pool')
    cand = np.concatenate((rng.choice(nor, K-n_bad, replace=False),
                           rng.choice(bad, n_bad, replace=False)))
    return rng.permutation(cand), n_bad/K


def generate_dynamic_trace(cr_percent=3, seed=2026, n=1000, rounds=200):
    """Exogenous impairment/onset/recovery schedule, common to all methods.

    cr_percent is the fraction of distinct identities experiencing one
    corruption episode, not per-round instantaneous corrupt prevalence.
    """
    n = int(n); rounds=int(rounds)
    if n > 5000 or n < 50 or rounds < 100:
        raise ValueError('n in [50,5000], rounds >= 100')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 863]))
    n_events = _round_count(n * float(cr_percent) / 100.0)
    targets = np.sort(rng.choice(n, n_events, replace=False))
    onsets = rng.integers(15, max(16,rounds-75), size=n_events, endpoint=False)
    dur = rng.integers(30, 71, size=n_events)
    recoveries = np.minimum(onsets+dur, rounds+1)
    t = np.arange(1, rounds+1, dtype=np.int32)[:, None]
    mask = np.zeros((rounds,n), dtype=np.bool_)
    mask[:,targets] = (t >= onsets[None,:]) & (t < recoveries[None,:])
    fail_uniform = rng.random((rounds,n), dtype=np.float32)
    return {
       'corrupted_mask': mask,
       'fail_uniform': fail_uniform,
       'event_model_ids': targets.astype(np.int32),
       'onset_round': onsets.astype(np.int32),
       'recovery_round': recoveries.astype(np.int32),
       'N':n, 'T':rounds, 'cr_event_percent':float(cr_percent), 'seed':int(seed)
    }


def resource_profiles(seed=2026, n=5000, rounds=200):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7281]))
    return {
       'node_z': rng.standard_normal((n,3)).astype(np.float32),
       'round_z': rng.standard_normal((rounds,3)).astype(np.float32),
       'seed': int(seed)
    }


def resource_costs(target_ids, verifier_ids, d=30, sigma=0.5, round_id=1,
                   profiles=None):
    """Deterministic pair-specific ms accounting; returns computation,transfer,delay.

    The function is stateless. Calling pairs in any order yields same result.
    Computation is serial per verifier; different verifiers run in parallel.
    """
    if profiles is None:
        profiles=resource_profiles()
    t=np.asarray(target_ids,dtype=np.int32)
    g=np.asarray(verifier_ids,dtype=np.int32)
    if t.shape != g.shape:
        raise ValueError('target_ids and verifier_ids must match shape')
    z=profiles['node_z']
    if not (1 <= int(round_id) <= len(profiles['round_z'])):
        raise ValueError('round_id must start at 1 and fit available resource trace')
    r=profiles['round_z'][int(round_id)-1]
    if np.any(t<0) or np.any(g<0) or np.any(t>=len(z)) or np.any(g>=len(z)):
        raise IndexError('resource IDs outside profiles')
    sig=float(sigma)
    z_comp = (0.9*z[g,0] + 0.1*r[0])/np.sqrt(0.82)
    z_tx = (0.6*z[t,1] + 0.6*z[g,1] + 0.2*r[1])/np.sqrt(0.76)
    z_lag= (0.6*z[t,2] + 0.6*z[g,2] + 0.2*r[2])/np.sqrt(0.76)
    factor=lambda v:np.exp(np.clip(sig*v-0.5*sig*sig,-20,20))
    comp_ms = 4.0*(float(d)/30.0)*factor(z_comp)
    transfer_ms = (4.0*float(d)*8.0/(50.0e6))*1000.0*factor(z_tx)
    delay_ms = 10.0*factor(z_lag)
    return {
       'compute_ms':comp_ms,
       'transfer_ms':transfer_ms,
       'network_ms':delay_ms,
       'total_ms':comp_ms+transfer_ms+delay_ms,
    }


def export_group(dataset_root=ROOT,n=3000,cr=3.0,d=30,H=0.5,seed=2026,
                 output_dir=None):
    """Optional physical TMC-style vectors.npz + metadata files for one group."""
    import csv
    obj=load_group(dataset_root,n,cr,d,H,seed)
    if output_dir is None:
        output_dir=Path(dataset_root)/'examples'/f'n_{n}'/f'corruption_{int(round(cr*10)):03d}'/f'dim_{d}'
    dest=Path(output_dir); dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'vectors.npz',
        X=obj['X'],binary_label=obj['binary_label'],subtype_label=obj['subtype_label'],
        cosine_to_prototype=obj['cosine_to_prototype'],prototype=obj['prototype'],
        tau_lower=np.array([0.997]),tau_upper=np.array([1.0]))
    with (dest/'row_metadata.csv').open('w',newline='',encoding='utf-8') as file:
        w=csv.writer(file)
        w.writerow(['row_index','device_id','binary_label','binary_label_name',
                    'subtype_label','subtype_name','cosine_to_prototype','sample_id'])
        for i, (y,s,cosv,sid) in enumerate(zip(obj['binary_label'],obj['subtype_label'],
                                              obj['cosine_to_prototype'],obj['sample_id'])):
            w.writerow([i,f'e_{i:05d}',int(y),('normal','corrupt')[int(y)],int(s),
                       SUBTYPES[int(s)],float(cosv),int(sid)])
    meta = dict(group_id=f'N{n}_CR{cr:.1f}_D{d}_H{H:.1f}_S{seed}',
                device_count=n,corruption_rate=cr,vector_dimension=d,
                heterogeneity_H=H,tau_lower=0.997,tau_upper=1.0,
                labels={str(i):name for i,name in enumerate(SUBTYPES)},counts=obj['counts'],
                generation='Deterministic sample from stored 50D angular descriptor banks')
    (dest/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    return dest


def load_dynamic_case(dataset_root=ROOT, cr_percent=3, seed=2026,
                      n=1000, d=30, H=0.5, rounds=200):
    """Paired descriptors+exogenous events for 200-round benchmark.

    No 3D (round,N,d) array is stored; consumer switches an event model's
    descriptor only while corrupted_mask[t, i] is true.
    """
    trace=generate_dynamic_trace(cr_percent,seed,n,rounds)
    normal=load_group(dataset_root,n,0,d,H,seed)
    source=load_group(dataset_root,n,5,d,H,seed+50000)
    corrupted=source['X'][source['binary_label']==1]
    m=len(trace['event_model_ids'])
    if len(corrupted)<m:
        raise ValueError('Insufficient corrupted descriptor pool')
    return {
        'normal_X':normal['X'],
        'corrupted_event_X':corrupted[:m],
        'event_model_ids':trace['event_model_ids'],
        'corrupted_mask':trace['corrupted_mask'],
        'fail_uniform':trace['fail_uniform'],
        'onset_round':trace['onset_round'],
        'recovery_round':trace['recovery_round'],
        'prototype':normal['prototype'],
        'seed':int(seed),
    }
