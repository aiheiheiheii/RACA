"""Self-contained full integrity and model-assumption validation."""
import csv, json, hashlib, math
from pathlib import Path
import numpy as np
from load_dataset import (ROOT, BANK_COUNTS, SUBTYPES, H_VALUES,
        N_VALUES, D_VALUES, CR_VALUES, enumerate_configs, load_group,
        cosine_pairwise, draw_candidate_set, generate_dynamic_trace,
        resource_profiles, resource_costs, load_dynamic_case)


def main(root=ROOT, full=True):
    root=Path(root)
    outcomes=[]
    def check(flag, what):
        if not bool(flag):
            raise AssertionError(what)
        outcomes.append('PASS '+what)
    with (root/'config.json').open() as f:
        cfg=json.load(f)
    check(cfg['expected_virtual_groups']==729,'9 x 9 x 9 = 729 virtual base configurations')
    with (root/'dataset_summary.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    check(len(rows)==729,'dataset_summary.csv 729 rows')
    check(len({r['group_id'] for r in rows})==729,'all group IDs unique')
    expected_sum=sum(n for n in N_VALUES)*len(CR_VALUES)*len(D_VALUES)
    check(sum(int(r['n']) for r in rows)==expected_sum,'virtual record totals correct')
    check(all(r['status']=='bank_view' for r in rows),'dataset_summary correctly advertises lazy bank views')

    for h in range(11):
        path=root/'banks'/f'bank_H_{h:02d}.npz'
        check(path.exists(),f'bank H={h/10:.1f} exists')
        with np.load(path, allow_pickle=False) as bank:
            q=bank['X_50']; theta=bank['theta']; p=bank['prototype_50'];
            y=bank['binary_label']; subtype=bank['subtype_label']
            check(q.shape==(9000,50) and q.dtype==np.float32,
                  f'bank {h}: 9000x50 float32')
            check(np.isfinite(q).all() and np.isfinite(theta).all(),
                  f'bank {h}: finite descriptors and angles')
            check(np.max(np.abs(np.linalg.norm(q,axis=1)-1))<4e-6,
                  f'bank {h}: descriptors unit normalized')
            check(np.max(np.abs(np.bincount(subtype,minlength=4)-np.array(BANK_COUNTS)))==0,
                  f'bank {h}: subtype counts match')
            check(np.all(y==(subtype>=2)), f'bank {h}: binary/subtype labels match')
            check(abs(np.linalg.norm(p)-1)<1e-6, f'bank {h}: prototype normalized')
            check(np.max(np.abs(q.astype('float64')@p.astype('float64')-np.cos(theta)))<1.1e-6,
                  f'bank {h}: cosine/angular representation match')

    # Important cross-H invariants: all variation is restricted to benign samples.
    with np.load(root/'banks'/'bank_H_00.npz') as z0, np.load(root/'banks'/'bank_H_10.npz') as z1:
        check(np.array_equal(z0['X_50'][6000:],z1['X_50'][6000:]),
              'corrupt bank invariant under normal-heterogeneity H')
        check(np.array_equal(z0['binary_label'],z1['binary_label']),
              'labels stable across heterogeneity H')
        check(np.mean(np.sum(z0['X_50'][4800:6000]*z0['prototype_50'],axis=1)) >
              np.mean(np.sum(z1['X_50'][4800:6000]*z1['prototype_50'],axis=1)),
              'high normal heterogeneity increases in angular deviation')

    if full:
        for n,cr,d in enumerate_configs():
            z=load_group(root,n,cr,d,0.5,seed=2026)
            wanted=int(math.floor(n*cr/100+.5))
            if z['X'].shape!=(n,d) or z['X'].dtype!=np.float32:
                raise AssertionError(f'{n} {cr} {d} unexpected shape/dtype')
            if z['binary_label'].sum()!=wanted:
                raise AssertionError(f'{n} {cr} {d} wrong corruption count')
            if not np.isfinite(z['X']).all():
                raise AssertionError(f'{n} {cr} {d} NaN/Inf')
            if max(abs(np.linalg.norm(z['X'],axis=1)-1))>5e-6:
                raise AssertionError(f'{n} {cr} {d} normalization')
            if max(abs(z['X'].astype(np.float64)@z['prototype'].astype(np.float64)-z['cosine_to_prototype']))>1e-6:
                raise AssertionError(f'{n} {cr} {d} cosine mismatch')
        check(True,'ALL 729 config load/shape/count/norm/cosine checks')

    a=load_group(root,3000,3,30,0.5,2026)
    b=load_group(root,3000,3,30,0.5,2026)
    check(np.array_equal(a['X'],b['X']) and np.array_equal(a['sample_id'],b['sample_id']),
          'same seed same group byte-identical')
    c=load_group(root,3000,3,30,0.5,2027)
    check(not np.array_equal(a['sample_id'],c['sample_id']),
          'independent seeds change sampling')
    e=load_group(root,3000,3,50,0.5,2026)
    check(np.array_equal(a['sample_id'],e['sample_id']),
          'same sample IDs paired across descriptor dimensions')
    f=load_group(root,3000,3,30,0.9,2026)
    check(np.array_equal(a['sample_id'],f['sample_id']),
          'same sample IDs paired across H')

    d=load_group(root,3000,3,30,0.5,2032)
    cand,q=draw_candidate_set(d,K=50,q_C=.48,seed=81,
                    use_severe_only=True,use_common_only=True)
    check(len(cand)==50 and int(d['binary_label'][cand].sum())==24,
          'GT theorem boundary K50 24 abnormal 26 normal')
    scores=cosine_pairwise(d['X'][cand]); supports=((scores>=.997)&(scores<=1+1e-8)).sum(axis=1)
    accept=supports>=26
    check(np.count_nonzero(accept)==26 and np.count_nonzero(d['binary_label'][cand][accept])==0,
          'GT majority theorem boundary pure 26 accepted normals')
    cand50,qe=draw_candidate_set(d,K=50,q_C=.50,seed=81,
                    use_severe_only=True,use_common_only=True)
    check(int(d['binary_label'][cand50].sum())==25,
          'GT 50% outside strict-majority guarantee')
    check(q==.48 and qe==.50,'candidate realized ratios recorded exactly')

    for cr in range(1,6):
        path=root/'dynamic'/f'reference_cr_{cr:02d}.npz'
        with np.load(path) as z:
            mask=z['corrupted_mask']; ids=z['event_model_ids'];
            ons=z['onset_round']; rec=z['recovery_round']; fail=z['fail_uniform']
            check(mask.shape==(200,1000) and fail.shape==(200,1000),
                  f'dynamic cr{cr} shapes')
            check(len(ids)==10*cr and len(set(ids.tolist()))==len(ids),
                  f'dynamic cr{cr} unique event count')
            check(np.all((ons>=15)&(ons<=125)&(rec>ons)&(rec<=201)),
                  f'dynamic cr{cr} onset/recovery well-defined')
            rebuilt=np.zeros_like(mask)
            for i,a,b in zip(ids,ons,rec):
                rebuilt[a-1:b-1,i]=True
            check(np.array_equal(mask,rebuilt),f'dynamic cr{cr} episodes match masks')
            check(np.all((fail>=0)&(fail<1)),f'dynamic cr{cr} valid failure draws')
            check(np.array_equal(mask,generate_dynamic_trace(cr,2026)['corrupted_mask']),
                  f'dynamic cr{cr} deterministic generator consistency')

    dyn=load_dynamic_case(root,cr_percent=5)
    check(dyn['normal_X'].shape==(1000,30) and dyn['corrupted_event_X'].shape==(50,30),
          'dynamic descriptor pairing 1000 normal / 50 event corruption states')
    with np.load(root/'resources'/'reference_profiles.npz') as z:
        prof=resource_profiles(2026,5000,200)
        check(np.array_equal(prof['node_z'],z['node_z']) and np.array_equal(prof['round_z'],z['round_z']),
              'reference resource profiles reproducible')
    pairs_i=np.arange(50,1050,dtype=np.int32)
    pairs_g=np.arange(1000,2000,dtype=np.int32)
    c0=resource_costs(pairs_i,pairs_g,d=30,sigma=0,round_id=13,profiles=prof)
    c5=resource_costs(pairs_i,pairs_g,d=30,sigma=.5,round_id=13,profiles=prof)
    check(np.allclose(c0['compute_ms'],4) and np.allclose(c0['network_ms'],10),
          'homogeneous resource baseline sigma=0')
    check(np.all(c5['total_ms']>0) and np.std(c5['total_ms'])>0.5,
          'heterogeneous resource costs finite positive and variable')
    rev=resource_costs(pairs_i[::-1],pairs_g[::-1],d=30,sigma=.5,round_id=13,profiles=prof)
    check(np.array_equal(c5['total_ms'],rev['total_ms'][::-1]),
          'resource lookup order-independent across algorithms')

    stat=[]
    for H in (0,0.3,0.5,0.7,1.0):
        group=load_group(root,3000,3,30,H)
        sim=group['cosine_to_prototype']; y=group['binary_label']; z=group['subtype_label']
        pred=(sim<.997)
        stat.append({'H':H,'FPR_vs_prototype':round(float(np.mean(pred[y==0])),4),
                     'FNR_vs_prototype':round(float(np.mean(~pred[y==1])),4)})
    # Not an evaluation result; purely a dataset-shape/overlap sanity diagnostic.
    check(stat[0]['FPR_vs_prototype'] <= stat[-1]['FPR_vs_prototype'],
          'increasing legitimate heterogeneity can raise baseline false positives')
    check(stat[2]['FNR_vs_prototype']>0,
          'mild corrupt samples retain nontrivial classification overlap')
    check(stat[2]['FNR_vs_prototype']<1,
          'severe corruption still distinguishable')
    report=['# TNSE-SimSimple Dataset Validation','',
      f'**Result:** PASS — {len(outcomes)} individual assertions.',
      f'**Full grid tested:** {full}.',
      f'**Materialized descriptor banks:** 11 × 9000 rows = 99,000 bank records; 9,000 source IDs (paired across H).',
      f'**Deterministic virtual configurations:** 729 (9N×9cr×9d); default H=0.5.',
      f'**Aggregate virtual group rows:** {expected_sum:,} (not independent samples).',
      '','## Sanity-check classification (prototype-based diagnostic; NOT Proposed results)',
      '', '| H | Diagnostic FPR | Diagnostic FNR |','|---:|---:|---:|']
    for row in stat:
        report.append(f"| {row['H']:.1f} | {row['FPR_vs_prototype']:.4f} | {row['FNR_vs_prototype']:.4f} |")
    report.extend(['','## Test assertions','','```text',*outcomes,'```','',
      '## Limitations','',
      '- Fully synthetic, not measured from actual pretrained/quantized models.',
      '- Theorem 1 requires stronger pairwise assumptions than a normal binary label alone.',
      '- Corruption episodes are exogenous and not oracle data for algorithms.',
      '- Only synthetic resource traces, not wall-clock measurements.',
      '- Same bank source identities appear across experiment H levels and virtual groups; paired samples must not be treated as independent observations.',
      '- Designed for reproducible experiments, not for forcing any algorithm to win.',''])
    (root/'validation'/'VALIDATION_REPORT.md').write_text('\n'.join(report),encoding='utf8')
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256')
    manifest=[]
    for path in files:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(f'{digest}  {path.relative_to(root).as_posix()}')
    (root/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n',encoding='utf8')
    return len(outcomes),stat

if __name__=='__main__':
    count,stat=main(full=True)
    print('VALIDATION PASSED',count,'checks')
    print(stat)
