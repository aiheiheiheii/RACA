# TNSE-SimSimple Dataset Validation

**Result:** PASS — 143 individual assertions.
**Full grid tested:** True.
**Materialized descriptor banks:** 11 × 9000 rows = 99,000 bank records; 9,000 source IDs (paired across H).
**Deterministic virtual configurations:** 729 (9N×9cr×9d); default H=0.5.
**Aggregate virtual group rows:** 2,187,000 (not independent samples).

## Sanity-check classification (prototype-based diagnostic; NOT Proposed results)

| H | Diagnostic FPR | Diagnostic FNR |
|---:|---:|---:|
| 0.0 | 0.0000 | 0.0667 |
| 0.3 | 0.0000 | 0.0667 |
| 0.5 | 0.0000 | 0.0667 |
| 0.7 | 0.0924 | 0.0667 |
| 1.0 | 0.2000 | 0.0667 |

## Test assertions

```text
PASS 9 x 9 x 9 = 729 virtual base configurations
PASS dataset_summary.csv 729 rows
PASS all group IDs unique
PASS virtual record totals correct
PASS dataset_summary correctly advertises lazy bank views
PASS bank H=0.0 exists
PASS bank 0: 9000x50 float32
PASS bank 0: finite descriptors and angles
PASS bank 0: descriptors unit normalized
PASS bank 0: subtype counts match
PASS bank 0: binary/subtype labels match
PASS bank 0: prototype normalized
PASS bank 0: cosine/angular representation match
PASS bank H=0.1 exists
PASS bank 1: 9000x50 float32
PASS bank 1: finite descriptors and angles
PASS bank 1: descriptors unit normalized
PASS bank 1: subtype counts match
PASS bank 1: binary/subtype labels match
PASS bank 1: prototype normalized
PASS bank 1: cosine/angular representation match
PASS bank H=0.2 exists
PASS bank 2: 9000x50 float32
PASS bank 2: finite descriptors and angles
PASS bank 2: descriptors unit normalized
PASS bank 2: subtype counts match
PASS bank 2: binary/subtype labels match
PASS bank 2: prototype normalized
PASS bank 2: cosine/angular representation match
PASS bank H=0.3 exists
PASS bank 3: 9000x50 float32
PASS bank 3: finite descriptors and angles
PASS bank 3: descriptors unit normalized
PASS bank 3: subtype counts match
PASS bank 3: binary/subtype labels match
PASS bank 3: prototype normalized
PASS bank 3: cosine/angular representation match
PASS bank H=0.4 exists
PASS bank 4: 9000x50 float32
PASS bank 4: finite descriptors and angles
PASS bank 4: descriptors unit normalized
PASS bank 4: subtype counts match
PASS bank 4: binary/subtype labels match
PASS bank 4: prototype normalized
PASS bank 4: cosine/angular representation match
PASS bank H=0.5 exists
PASS bank 5: 9000x50 float32
PASS bank 5: finite descriptors and angles
PASS bank 5: descriptors unit normalized
PASS bank 5: subtype counts match
PASS bank 5: binary/subtype labels match
PASS bank 5: prototype normalized
PASS bank 5: cosine/angular representation match
PASS bank H=0.6 exists
PASS bank 6: 9000x50 float32
PASS bank 6: finite descriptors and angles
PASS bank 6: descriptors unit normalized
PASS bank 6: subtype counts match
PASS bank 6: binary/subtype labels match
PASS bank 6: prototype normalized
PASS bank 6: cosine/angular representation match
PASS bank H=0.7 exists
PASS bank 7: 9000x50 float32
PASS bank 7: finite descriptors and angles
PASS bank 7: descriptors unit normalized
PASS bank 7: subtype counts match
PASS bank 7: binary/subtype labels match
PASS bank 7: prototype normalized
PASS bank 7: cosine/angular representation match
PASS bank H=0.8 exists
PASS bank 8: 9000x50 float32
PASS bank 8: finite descriptors and angles
PASS bank 8: descriptors unit normalized
PASS bank 8: subtype counts match
PASS bank 8: binary/subtype labels match
PASS bank 8: prototype normalized
PASS bank 8: cosine/angular representation match
PASS bank H=0.9 exists
PASS bank 9: 9000x50 float32
PASS bank 9: finite descriptors and angles
PASS bank 9: descriptors unit normalized
PASS bank 9: subtype counts match
PASS bank 9: binary/subtype labels match
PASS bank 9: prototype normalized
PASS bank 9: cosine/angular representation match
PASS bank H=1.0 exists
PASS bank 10: 9000x50 float32
PASS bank 10: finite descriptors and angles
PASS bank 10: descriptors unit normalized
PASS bank 10: subtype counts match
PASS bank 10: binary/subtype labels match
PASS bank 10: prototype normalized
PASS bank 10: cosine/angular representation match
PASS corrupt bank invariant under normal-heterogeneity H
PASS labels stable across heterogeneity H
PASS high normal heterogeneity increases in angular deviation
PASS ALL 729 config load/shape/count/norm/cosine checks
PASS same seed same group byte-identical
PASS independent seeds change sampling
PASS same sample IDs paired across descriptor dimensions
PASS same sample IDs paired across H
PASS GT theorem boundary K50 24 abnormal 26 normal
PASS GT majority theorem boundary pure 26 accepted normals
PASS GT 50% outside strict-majority guarantee
PASS candidate realized ratios recorded exactly
PASS dynamic cr1 shapes
PASS dynamic cr1 unique event count
PASS dynamic cr1 onset/recovery well-defined
PASS dynamic cr1 episodes match masks
PASS dynamic cr1 valid failure draws
PASS dynamic cr1 deterministic generator consistency
PASS dynamic cr2 shapes
PASS dynamic cr2 unique event count
PASS dynamic cr2 onset/recovery well-defined
PASS dynamic cr2 episodes match masks
PASS dynamic cr2 valid failure draws
PASS dynamic cr2 deterministic generator consistency
PASS dynamic cr3 shapes
PASS dynamic cr3 unique event count
PASS dynamic cr3 onset/recovery well-defined
PASS dynamic cr3 episodes match masks
PASS dynamic cr3 valid failure draws
PASS dynamic cr3 deterministic generator consistency
PASS dynamic cr4 shapes
PASS dynamic cr4 unique event count
PASS dynamic cr4 onset/recovery well-defined
PASS dynamic cr4 episodes match masks
PASS dynamic cr4 valid failure draws
PASS dynamic cr4 deterministic generator consistency
PASS dynamic cr5 shapes
PASS dynamic cr5 unique event count
PASS dynamic cr5 onset/recovery well-defined
PASS dynamic cr5 episodes match masks
PASS dynamic cr5 valid failure draws
PASS dynamic cr5 deterministic generator consistency
PASS dynamic descriptor pairing 1000 normal / 50 event corruption states
PASS reference resource profiles reproducible
PASS homogeneous resource baseline sigma=0
PASS heterogeneous resource costs finite positive and variable
PASS resource lookup order-independent across algorithms
PASS increasing legitimate heterogeneity can raise baseline false positives
PASS mild corrupt samples retain nontrivial classification overlap
PASS severe corruption still distinguishable
```

## Limitations

- Fully synthetic, not measured from actual pretrained/quantized models.
- Theorem 1 requires stronger pairwise assumptions than a normal binary label alone.
- Corruption episodes are exogenous and not oracle data for algorithms.
- Only synthetic resource traces, not wall-clock measurements.
- Same bank source identities appear across experiment H levels and virtual groups; paired samples must not be treated as independent observations.
- Designed for reproducible experiments, not for forcing any algorithm to win.
