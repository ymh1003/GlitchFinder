#!/usr/bin/env python3
'''
Integrate several heatmaps into a single one.
All need to share the same number of boxes when generating the combined pcd.
'''

import gcode_comp_Z as GZ
import argparse
import json
import numpy as np
import math
from tqdm import tqdm
import timerutils
from pathlib import Path
import zipfile

Timer = timerutils.Timer()

def read_json_file(filepath):
    try:
        with open(filepath, 'rt') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def load_npz_fast(npzfile, compare = False):
    z = zipfile.ZipFile(npzfile, "r")
    out = {}
    for i in tqdm(z.namelist(), f"reading npz"):
        with z.open(i) as d:
            # remove .npy
            out[i[:-4]] = np.lib.format.read_array(d)

    if compare:
        compare_npz(npzfile, out)

    return out

# for debugging
def compare_npz(npzfile, x):
    data = np.load(npzfile, allow_pickle=True)
    # could be assert(all(, but this is nicer for debugging.
    for k in x:
        print(k, np.array_equal(x[k], data[k]))

    return x

# for debugging
def decode(npzdata):
    out = {}

    for i in tqdm(npzdata, "decoding npz"):
        out[i] = npzdata[i]

    return out

# pcd_0_0, pcd_0_1, pcd_1_0, pcd_1_1, pcd_2_0, pcd_2_1, ...
def generate_pcd_from_box(cpcds, num_box, remove_ids):
    remove_ids = set(remove_ids)
    for i in tqdm(range(num_box), desc="Generating points from each box"):
        if i not in remove_ids:
            oa_pcd = []
            for j, cpcd in enumerate(cpcds):
                if j > 0:  # only append points from the second point cloud
                    oa_pcd.extend(cpcd[f'pcd_{i}_{1}'])
                else:  # this will append points from point cloud with original orientation
                    oa_pcd.extend(cpcd[f'pcd_{i}_{0}'])
                    oa_pcd.extend(cpcd[f'pcd_{i}_{1}'])
            yield oa_pcd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate several heatmaps for different rotations')
    parser.add_argument('json_files', type=str, nargs='+', help='JSON filepaths')
    parser.add_argument('-t', '--threshold', type=float, default=90.,
                        help='Threshold, in percentile')
    parser.add_argument('-n', '--name', type=str, help='Name for overall heatmap')
    parser.add_argument('--distribution', action="store_true", help='Output error distribution graph.')
    args = parser.parse_args()

    dir = Path(args.json_files[0]).parent

    num_box_list = []  # ensure that all heatmaps have the same number of boxes
    oa_hd_list, new_hd_list, none_ids = [], [], []
    cpcd_list = []

    for filepath in args.json_files:
        result = read_json_file(filepath)
        if result is not None:
            num_box_list.append(result['number of boxes'])
            oa_hd_list.append(result['hausdorff distance'])

            Timer.start("load combined pcd")
            #data = np.load(result['combined points'], allow_pickle=True)
            data = load_npz_fast(result['combined points'], compare=False)
            Timer.stop("load combined pcd")
            cpcd_list.append(data)
        else:
            exit(1)

    if len(set(num_box_list)) == 1:
        num_box = num_box_list[0]
    else:
        print("Invalid integration: different numbers of boxes")
        exit(1)

    oa_hd_list_T = np.array(oa_hd_list).T

    num_inf = 0
    for i, box in enumerate(oa_hd_list_T):
        if math.inf in box:
            num_inf += 1
            new_hd_list.append(math.inf)
        elif any(isinstance(_, float) for _ in box):
            num_hd = [hd for hd in box if isinstance(hd, float)]
            new_hd_list.append(sum(num_hd)/len(num_hd))
        else:
            new_hd_list.append(None)
            none_ids.append(i)
    
    filtered_hd = [i for i in new_hd_list if i != None and i != math.inf]
    
    if args.distribution:
        GZ.draw_distribution(filtered_hd, num_inf, filepath=dir/f"{args.name}_dist.pdf")
    
    cleaned_pcd = list(generate_pcd_from_box(cpcd_list, num_box, none_ids))
    
    GZ.heatmap_transparency(cleaned_PCD=cleaned_pcd, 
                            HD_list=new_hd_list, 
                            threshold_HD=np.percentile(filtered_hd, args.threshold),
                            filepath=dir/f"{args.name}_overall.pcd")
