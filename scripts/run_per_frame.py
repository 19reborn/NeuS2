import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default='.')
parser.add_argument('--output_dir', type=str, default='neus2_exp')
parser.add_argument('--config', type=str, default='base.json')

args = parser.parse_args()

frames = sorted(glob.glob(os.path.join(args.base_dir, '*.json')))
config = args.config

for scene in frames:
    num = os.path.basename(scene).split('.')[0]
    name = f"{args.output_dir}/{num}"
    
    os.system(f"python scripts/run_dynamic.py \
        --scene {scene} --mode nerf --name {name} --network {config} \
        --save_snapshot_per_frame")