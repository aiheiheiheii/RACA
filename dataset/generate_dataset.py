"""Generate the entire compact TNSE dataset using a fixed, auditable DGP.

NOTES:
  - No genuine trained DNN weights are used; data are synthetic descriptors.
  - This code never uses TNSE/benchmark outcomes to change sampled descriptors.
  - For model scale/cr/d we expose 729 deterministic VIRTUAL configurations.
  - Eleven actual H-specific canonical banks + event and resource traces are saved.
"""
from pathlib import Path
import csv
import json
import numpy as np

from load_dataset import (ROOT, SUBTYPES, BANK_COUNTS, N_VALUES, CR_VALUES,
                          D_VALUES, H_VALUES, export_group,
                          generate_dynamic_trace, resource_profiles)

SEED = 2026
RNG = np.random.default_rng(SEED)
TOTAL = sum(BANK_COUNTS)


def rows_unit_orthogonal(g, p):
    g = np.asarray(g,dtype=np.float64)
    g = g-(g @ p)[:,None]*p[None,:]
    norms = np.linalg.norm(g,axis=1)
    if np.any(norms<1e-8):
        raise ValueError('Zero perturbation direction')
    return g/norms[:,None]


def make_correlated_normal_vectors(n, d=50):
    """Moderately block-correlated angular perturbation directions."""
    noise = RNG.normal(size=(n,d))
    block = RNG.normal(size=(n,d//5))
    for k in range(d//5):
        noise[:,5*k:5*k+5] += 0.45*block[:,k:k+1]
    # Tiny shared within-class direction makes legitimate changes correlated.
    common = RNG.normal(size=(1,d))
    return noise+0.15*common


def make_sparse_corruption_directions(n, frac_low, frac_high,d=50):
    result=np.zeros((n,d),dtype=np.float64)
    for j in range(n):
        count=max(2,int(round(d * RNG.uniform(frac_low,frac_high))))
        must0=int(RNG.integers(0,10))
        ids=RNG.choice(d, size=count-1, replace=False)
        ids=np.unique(np.r_[ids,must0])
        result[j,ids]=RNG.normal(size=len(ids))
    return result


def main(root=ROOT):
    root=Path(root)
    for name in ('banks','resources','dynamic','validation','examples'):
        (root/name).mkdir(parents=True,exist_ok=True)

    d=50
    p=RNG.normal(0.65,0.35,size=d)
    p/=np.linalg.norm(p)
    c,h,s,m=BANK_COUNTS
    total=c+h+s+m
    latent=np.vstack([
        make_correlated_normal_vectors(c,d),
        make_correlated_normal_vectors(h,d),
        make_sparse_corruption_directions(s,0.15,0.40,d),
        make_sparse_corruption_directions(m,0.04,0.15,d),
    ])
    U=rows_unit_orthogonal(latent,p)
    Utheta=RNG.uniform(size=total)
    y=np.r_[np.zeros(c+h,dtype=np.uint8),np.ones(s+m,dtype=np.uint8)]
    subtype=np.repeat(np.arange(4,dtype=np.uint8), BANK_COUNTS)
    assert len(y)==len(subtype)==total

    def angles(H):
        th=np.empty(total,dtype=np.float64)
        th[:c]=0.006+0.017*H+0.005*Utheta[:c]
        th[c:c+h]=0.014+0.073*H+0.022*Utheta[c:c+h]
        th[c+h:c+h+s]=0.13+0.18*Utheta[c+h:c+h+s]
        th[c+h+s:]=0.052+0.063*Utheta[c+h+s:]
        return th

    for idx,H in enumerate(H_VALUES):
        th=angles(H)
        Q=(np.cos(th)[:,None]*p+np.sin(th)[:,None]*U).astype(np.float32)
        Q=Q/np.linalg.norm(Q,axis=1,keepdims=True)
        path=root/'banks'/f'bank_H_{idx:02d}.npz'
        np.savez_compressed(path,
                 X_50=Q, theta=th.astype(np.float32),
                 prototype_50=p.astype(np.float32),
                 binary_label=y, subtype_label=subtype,
                 sample_id=np.arange(total,dtype=np.int32))
        print('generated',path.name,round(path.stat().st_size/1e6,2),'MB')

    # Fixed reference resource profiles; other seeds recomputed deterministically.
    profile=resource_profiles(seed=SEED,n=5000,rounds=200)
    np.savez_compressed(root/'resources'/'reference_profiles.npz',
        node_z=profile['node_z'],round_z=profile['round_z'],
        seed=np.array([SEED],dtype=np.int32))

    # Example exogenous dynamic episodes; dynamic loader supports arbitrary seeds.
    for cr in range(1,6):
        trace=generate_dynamic_trace(cr_percent=cr,seed=SEED,n=1000,rounds=200)
        np.savez_compressed(root/'dynamic'/f'reference_cr_{cr:02d}.npz',
             corrupted_mask=trace['corrupted_mask'],
             fail_uniform=trace['fail_uniform'],
             event_model_ids=trace['event_model_ids'],
             onset_round=trace['onset_round'],
             recovery_round=trace['recovery_round'],
             n=np.array([1000],dtype=np.int32),
             T=np.array([200],dtype=np.int32),
             event_cr_percent=np.array([cr],dtype=np.float32),
             seed=np.array([SEED],dtype=np.int32))

    # Materialized TMC-style example; full groups are deterministic views via loader.
    example=export_group(root,n=3000,cr=3.0,d=30,H=0.5,seed=SEED,
                         output_dir=root/'examples'/'n_3000'/'corruption_03'/'dim_30')
    print('materialized example',example)

    # Exact grid indices. Files for every combination are NOT duplicated on disk.
    summary=root/'dataset_summary.csv'
    with summary.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f)
        w.writerow(['group_id','n','corruption_rate_percent','d','H_default',
                    'n_normal','n_corrupt','status'])
        for n in N_VALUES:
            for cr in CR_VALUES:
                nc=int(np.floor(n*cr/100+0.5))
                for dim in D_VALUES:
                    w.writerow([f'n{n}_cr{cr:.1f}_d{dim}',n,cr,dim,0.5,n-nc,nc,'bank_view'])

    config={
        'dataset_name':'TNSE-SimSimple-v1',
        'provenance':'fully synthetic angular/static-attribute descriptors; not pretrained models',
        'seed':SEED,
        'n_values':list(N_VALUES),'cr_percent_values':list(CR_VALUES),
        'd_values':list(D_VALUES),'H_values':list(H_VALUES),
        'default':{'n':3000,'cr_percent':3.0,'d':30,'H':0.5,'tau_lower':0.997,
                   'K':50,'M':25,'alpha':0.5,'lambda':0.5,'rho':0.5,'eta':0.5,
                   'A_max':20,'T_round_max_ms':400,'beta_B':0.10,'sigma':0.5,
                   'T':200},
        'bank_subtype_counts':dict(zip(SUBTYPES, BANK_COUNTS)),
        'composition':{'normal_high_fraction':0.2,'corrupt_mild_fraction':0.2},
        'precision':'float32','payload_bytes':'4*d',
        'resource':{'compute_ms_at_d30':4.0,'bandwidth_Mbps':50.0,'oneway_network_ms':10.0},
        'dynamic':{'event_cr_is_ever_corrupted_fraction':True,
                   'onset_min_round':15,'onset_max_exclusive':'T-75',
                   'duration_min':30,'duration_max_inclusive':70},
        'expected_virtual_groups':len(N_VALUES)*len(CR_VALUES)*len(D_VALUES),
        'notes':'H=0.0..1.0 controls legitimate model variation only; corrupted banks stay fixed.',
    }
    (root/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    print('complete, virtual groups=',config['expected_virtual_groups'])

if __name__=='__main__':
    main()
