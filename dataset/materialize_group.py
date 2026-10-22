"""Optional: write a physical TMC-style NPZ group to an output folder."""
from argparse import ArgumentParser
from pathlib import Path
from load_dataset import ROOT, export_group

if __name__=='__main__':
    p=ArgumentParser()
    p.add_argument('--dataset-root',default=str(ROOT))
    p.add_argument('--n',type=int,default=3000)
    p.add_argument('--cr',type=float,default=3.0)
    p.add_argument('--d',type=int,default=30)
    p.add_argument('--H',type=float,default=0.5)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    print(export_group(a.dataset_root,a.n,a.cr,a.d,a.H,a.seed,a.output_dir))
