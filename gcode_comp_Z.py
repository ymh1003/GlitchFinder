#!/usr/bin/env python3

from pathlib import Path
import argparse
import re
import math
import numpy as np
from numpy.linalg import inv
import open3d as o3d
import copy
# import shapely
from scipy.spatial import KDTree
import pandas as pd
# from descartes import PolygonPatch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# from mpl_toolkits.mplot3d import Axes3D
# import alphashape
from decimal import Decimal
from tqdm import tqdm
from time import perf_counter, monotonic_ns
from pyntcloud import PyntCloud
import cProfile
from memory_profiler import profile
import multiprocessing
import gc
import json
import timerutils

V = None
N = None

Timer = timerutils.Timer()

# global variable to pass combined PCD to child processes without
# serialization and without using mp shared memory (since this is
# readonly)
_combined_pcd_hack = None


def set_global_OP(bool):
    global V
    global N
    
    V = DecimalOrFloat_V(bool)
    N = DecimalOrFloat_N(bool)
    
    
def initialize_global():
    global V
    global N
    V = np.vectorize(lambda x:x)
    N = float


def DecimalOrFloat_V(bool):  # apply to lists
    if bool:
        return np.vectorize(Decimal)
    else:
        return np.vectorize(lambda x:x)


def DecimalOrFloat_N(bool):  # apply to numbers
    if bool:
        return Decimal
    else:
        return float
    

def SQRT(val):
    if isinstance(val, float):
        return math.sqrt(val)
    elif isinstance(val, Decimal):
        return val.sqrt()
    else:
        raise ValueError("Unsupported input type")


class Line:
    def __init__(self, startpoint, endpoint):
        self.start = startpoint
        self.end = endpoint


class printer3D:
    def __init__(self):
        self.curX = N('0')
        self.curY = N('0')
        self.curZ = N('0')
        self.absolute = True
        self.linear_paths = {}  # dictionary with key being z coordinate
        # self.wall = False
        self.MinX = N('0')
        self.MinY = N('0')
        self.MinZ = N('0')
        self.MaxX = N('0')
        self.MaxY = N('0')
        self.MaxZ = N('0')

    def G0(self, coordinates=None, E=None):
        if self.absolute:
            self.curX = coordinates[0] if coordinates[0] is not None else self.curX
            self.curY = coordinates[1] if coordinates[1] is not None else self.curY
            self.curZ = coordinates[2] if coordinates[2] is not None else self.curZ
        else:
            self.curX = self.curX + coordinates[0] if coordinates[0] is not None else self.curX
            self.curY = self.curY + coordinates[1] if coordinates[1] is not None else self.curY
            self.curZ = self.curZ + coordinates[2] if coordinates[2] is not None else self.curZ

        if self.curZ not in self.linear_paths:
            self.linear_paths[self.curZ] = []

    def G1(self, coordinates=None, E=None):
        if coordinates[2] is not None:  # only has z-coordinate
            self.curZ = self.curZ + coordinates[2] if not self.absolute else coordinates[2]
            if self.curZ not in self.linear_paths:
                self.linear_paths[self.curZ] = []
        elif coordinates[0] or coordinates[1] is not None:
            if self.absolute:
                new_X = coordinates[0] if coordinates[0] is not None else self.curX
                new_Y = coordinates[1] if coordinates[1] is not None else self.curY
            else:
                new_X = coordinates[0] + self.curX if coordinates[0] is not None else self.curX
                new_Y = coordinates[1] + self.curY if coordinates[1] is not None else self.curY
            # if self.wall:
            newline = Line([self.curX, self.curY], [new_X, new_Y])
            self.linear_paths[self.curZ].append(newline)
            self.curX = new_X
            self.curY = new_Y

    def G90(self):
        self.absolute = True

    def G91(self):
        self.absolute = False

    def MINX(self, num):
        self.MinX = num

    def MINY(self, num):
        self.MinY = num

    def MINZ(self, num):
        self.MinZ = num

    def MAXX(self, num):
        self.MaxX = num

    def MAXY(self, num):
        self.MaxY = num

    def MAXZ(self, num):
        self.MaxZ = num

    '''
    def TYPE(self, part):
        if part == 'WALL-INNER' or part == 'WALL-OUTER':
            self.wall = True
        else:
            self.wall = False
    '''


def extract_prusa_coord(input_line):
    COORD_RE = re.compile (r"G1\s(X(?P<x>[-+]?\d*(\.\d*)?))?(\s)?"
                            r"(Y(?P<y>[-+]?\d*(\.\d*)?))?(\s)?(Z(?P<z>[-+]?\d*(\.\d*)?))?(\s)?"
                             r"(E[-+]?\d*(\.\d*)?)?(\s)?(F\d*)?")
    m = COORD_RE.match(input_line)
    coord_x = N(m.group('x')) if m.group('x') is not None else None
    coord_y = N(m.group('y')) if m.group('y') is not None else None
    coord_z = N(m.group('z')) if m.group('z') is not None else None

    return coord_x, coord_y, coord_z


def extract_cura_coord(input_line):
    COORD_RE = re.compile (r"G[01]\s(F\d*(\.\d*)?)?(\s)?(X(?P<x>[-+]?\d*(\.\d*)?))?(\s)?"
                            r"(Y(?P<y>[-+]?\d*(\.\d*)?))?(\s)?(Z(?P<z>[-+]?\d*(\.\d*)?))?")
    m = COORD_RE.match(input_line)
    coord_x = N(m.group('x')) if m.group('x') is not None else None
    coord_y = N(m.group('y')) if m.group('y') is not None else None
    coord_z = N(m.group('z')) if m.group('z') is not None else None

    return coord_x, coord_y, coord_z


def extract_MinX(input_line):
    m = re.match(r";MINX:(?P<x>(\d*)?(\.)?(\d*)?)", input_line)
    minX = N(m.group('x'))
    return minX


def extract_MinY(input_line):
    m = re.match(r";MINY:(?P<x>(\d*)?(\.)?(\d*)?)", input_line)
    minY = N(m.group('x'))
    return minY


def extract_MinZ(input_line):
    # m = re.match(r";MINZ:(?P<x>(\d*)?(\.)?(\d*)?)", input_line)
    # minZ = N(m.group('x'))
    # return minZ
    return N('0')


def extract_MaxX(input_line):
    m = re.match(r";MAXX:(?P<x>(\d*)?(\.)?(\d*)?)", input_line)
    maxX = N(m.group('x'))
    return maxX


def extract_MaxY(input_line):
    m = re.match(r";MAXY:(?P<x>(\d*)?(\.)?(\d*)?)", input_line)
    maxY = N(m.group('x'))
    return maxY


def extract_MaxZ(input_line):
    m = re.match(r";MAXZ:(?P<x>(\d*)?(\.)?(\d*)?)", input_line)
    maxZ = N(m.group('x'))
    return maxZ


'''
# detect whether it's printing inner wall, outer wall, or sth else
def extract_part(input_line):
    m = re.match(r";TYPE:(?P<x>.*)", input_line)
    print_part = m.group('x')
    return print_part
'''

extrusion = r'\bE'
def read_gcode_from_prusa(filename):
    Timer.start("parsing")
    print(f"Start parsing G-Code file: {filename}")
    p = Path(filename)
    with p.open() as file:
        printer = printer3D()
        for instr in file:
            if instr.startswith('G1'):
                if re.search(extrusion, instr):
                    printer.G1(extract_prusa_coord(instr))
                else:  # no extrusion specified then it's equivalent to G0
                    printer.G0(extract_prusa_coord(instr))
            elif instr.startswith('G90'):
                printer.G90()
            elif instr.startswith('G91'):
                printer.G91()
    Timer.stop("parsing")
    print(f"Finish parsing G-Code file: {filename}")
    return dict2list(printer.linear_paths), None


def read_gcode_from_cura(filename):
    Timer.start("parsing")
    print(f"Start parsing G-Code file: {filename}")
    p = Path(filename)
    with p.open() as file:
        printer = printer3D()
        for instr in file:
            if instr.startswith('G0'):
                printer.G0(extract_cura_coord(instr))
            elif instr.startswith('G1'):
                printer.G1(extract_cura_coord(instr))
            elif instr.startswith('G90'):
                printer.G90()
            elif instr.startswith('G91'):
                printer.G91()
            # elif instr.startswith(';TYPE:'):
            #     printer.TYPE(extract_part(instr))
            elif instr.startswith(';MINX'):
                printer.MINX(extract_MinX(instr))
            elif instr.startswith(';MINY'):
                printer.MINY(extract_MinY(instr))
            elif instr.startswith(';MINZ'):
                printer.MINZ(extract_MinZ(instr))
            elif instr.startswith(';MAXX'):
                printer.MAXX(extract_MaxX(instr))
            elif instr.startswith(';MAXY'):
                printer.MAXY(extract_MaxY(instr))
            elif instr.startswith(';MAXZ'):
                printer.MAXZ(extract_MaxZ(instr))
    Timer.stop("parsing")
    print(f"Finish parsing G-Code file: {filename}")
    
    bbox = np.array([[printer.MinX, printer.MinY, printer.MinZ],
                     [printer.MaxX, printer.MaxY, printer.MaxZ]])
    # return a list of lines and the bounding box
    return dict2list(printer.linear_paths), bbox
    
# apply rotation in the order x->y->z
def rotation_matrix(agl):
    rot_x = np.asarray([[1, 0, 0], 
                        [0, math.cos(agl[0]), -math.sin(agl[0])], 
                        [0, math.sin(agl[0]), math.cos(agl[0])]])
    rot_y = np.asarray([[math.cos(agl[1]), 0, math.sin(agl[1])], 
                        [0, 1, 0], 
                        [-math.sin(agl[1]), 0, math.cos(agl[1])]])
    rot_z = np.asarray([[math.cos(agl[2]), -math.sin(agl[2]), 0], 
                        [math.sin(agl[2]), math.cos(agl[2]), 0], 
                        [0, 0, 1]])
    return np.matmul(V(rot_z), np.matmul(V(rot_y), V(rot_x)))


# compute overall bbox from a list of bounding boxes
def overall_bbox(bbox_list):
    coord_dict = {f'c{i}': [] for i in range(6)}
    for bbox in bbox_list:
        for i in range(6):
            coord_dict[f'c{i}'].append(bbox[i//3][i%3])
    return [[min(coord_dict[f'c{i}']) for i in range(3)],
            [max(coord_dict[f'c{i}']) for i in range(3, 6)]]


# remember that we are trying to compare two models obtained from 
# different rotations on the same original model
def align_bbox(bbox_1, rot_1, rot_2):
    ct_1 = [(bbox_1[0][0] + bbox_1[1][0]) / 2, 
            (bbox_1[0][1] + bbox_1[1][1]) / 2,
            (bbox_1[0][2] + bbox_1[1][2]) / 2]
    rotated = rotate_points(bbox_1, rot_1, ct_1, undo=1)
    rotated[:, 2] -= min(rotated[:, 2])
    ct_2 = [(rotated[0][0] + rotated[1][0]) / 2, 
            (rotated[0][1] + rotated[1][1]) / 2,
            (rotated[0][2] + rotated[1][2]) / 2]
    rotated = rotate_points(rotated, rot_2, ct_2, undo=0)
    rotated[:, 2] -= min(rotated[:, 2])

    bmin = rotated[0] if rotated[0][0] < rotated[1][0] else rotated[1]
    bmax = rotated[1] if rotated[0][0] < rotated[1][0] else rotated[0]
    
    return [bmin, bmax]


def dict2list(lines_dict):  # helper method: convert dictionary to list, and matrix for sampling
    Timer.start("dict2list")
    lines_list = []
    lines_matrix = np.array([])
    for z_coord, lines in lines_dict.items():
        for ln in lines:
            #newline = Line(ln.start + [z_coord], ln.end + [z_coord])
            #lines_list.extend([newline.start, newline.end])
            lines_list.append([ln.start[0], ln.start[1], z_coord])
            lines_list.append([ln.end[0], ln.end[1], z_coord])
            #lines_matrix = np.append(lines_matrix, [newline.start, newline.end])

    lines_matrix = np.append(lines_matrix, lines_list)
    Timer.stop("dict2list")
    Timer.start("dict2list:reshape")
    lines_matrix = np.reshape(lines_matrix, (-1, 2, 3))
    Timer.stop("dict2list:reshape")
    return lines_matrix  # only return N x 2 x 3 array


def max_length_of_paths(lines_matrix):  # input is an N x 2 x 3 array
    return np.max(np.sqrt(np.power(lines_matrix[:, 1, 0] - lines_matrix[:, 0, 0], 2) +
                          np.power(lines_matrix[:, 1, 1] - lines_matrix[:, 0, 1], 2) +
                          np.power(lines_matrix[:, 1, 2] - lines_matrix[:, 0, 2], 2)))


def length_of_paths(lines_matrix):
    with Timer.time("length_of_paths"):
        return np.sqrt(np.power(lines_matrix[:, 1, 0] - lines_matrix[:, 0, 0], 2) +
                       np.power(lines_matrix[:, 1, 1] - lines_matrix[:, 0, 1], 2) +
                       np.power(lines_matrix[:, 1, 2] - lines_matrix[:, 0, 2], 2))

'''
def rotate_paths_3d(lines_matrix, angle, ct):  # angle is a list of rotation angle about each axis, ct is the rotation
    # center coordinate
    agl = np.radians(np.array(angle))
    ct_decimal = V(ct)

    rot_x = np.asarray([[1, 0, 0], [0, math.cos(agl[0]), -math.sin(agl[0])], [0, math.sin(agl[0]), math.cos(agl[0])]])
    # negate the y angle
    rot_y = np.asarray([[math.cos(agl[1]), 0, math.sin(-agl[1])], [0, 1, 0], [-math.sin(-agl[1]), 0, math.cos(agl[1])]])
    rot_z = np.asarray([[math.cos(agl[2]), -math.sin(agl[2]), 0], [math.sin(agl[2]), math.cos(agl[2]), 0], [0, 0, 1]])
    translation = np.transpose(lines_matrix - ct_decimal, (0, 2, 1))
    rot_x, rot_y, rot_z = V(rot_x), V(rot_y), V(rot_z)
    rotated = np.matmul(rot_z, np.matmul(rot_y, np.matmul(rot_x, translation))).transpose((0, 2, 1)) + ct_decimal
    min_z = np.min(rotated[:, :, -1])
    
    # here we assume the lowest height is 0.2, but it should be set as a parameter == layer height
    rotated[:, :, -1] += (N('0.2') - min_z)

    # return value is a Nx2x3 array
    return rotated
'''

# @profile
# undo is a boolean indicating whether we are undoing the rotation
def rotate_points(pts, rot, ct, undo):
    agl = np.radians(rot)
    rot_mat = rotation_matrix(agl).transpose() if undo else rotation_matrix(agl)
    translation = np.transpose(pts - ct)
    rotated = np.matmul(rot_mat, translation).transpose() + ct
    return rotated


def align_pcd(pcd, d_1, rot_1, ct, rot_2, d_2):
    pcd[:, 2] -= d_1
    if rot_1 == [0., 0., 0.]:
        undo_rot_pcd = pcd
    else:
        undo_rot_pcd = rotate_points(pcd, rot_1, ct, undo=1)
    if rot_2 == [0., 0., 0.]:
        return undo_rot_pcd
    else:
        apply_rot_2_pcd = rotate_points(undo_rot_pcd, rot_2, ct, undo=0)
        apply_rot_2_pcd[:, 2] += d_2
        return apply_rot_2_pcd


# compute the distance to bring a rotated model back to the build plate (for reverse engineering)
def align_to_build_plate(dim, ot, orig_ct):
    x, y, z = dim[0], dim[1], dim[2]
    bbox_v = [[-x/2, -y/2, -z/2], [-x/2, y/2, -z/2], [x/2, y/2, -z/2], [x/2, -y/2, -z/2],
              [-x/2, -y/2, z/2], [-x/2, y/2, z/2], [x/2, y/2, z/2], [x/2, -y/2, z/2]]
    rot_z = rotate_points(bbox_v, ot, orig_ct, undo=0)[:, 2]
    rot_h = max(rot_z) - min(rot_z)
    return (rot_h - z) / 2


def print_cuboids(cuboids):
    for row in cuboids:
        print(f"{{"
            f"vA: [{row[0]}, {row[1]}], "
            f"vB: [{row[2]}, {row[3]}], "
            f"vC: [{row[4]}, {row[5]}], "
            f"vD: [{row[6]}, {row[7]}], "
            f"z: {row[8]}"
            f"}}\n")


# @profile
def reconstruct_cuboids(lines_matrix, radius):  # Nozzle 0.4mm, radius 0.2mm
    Timer.start("reconstruct_cuboids")
    theta = np.arctan2((lines_matrix[:, 1, 1] - lines_matrix[:, 0, 1]).astype(float),
                       (lines_matrix[:, 1, 0] - lines_matrix[:, 0, 0]).astype(float))  # no arctan2 for Decimal
    l = radius * SQRT(N('2'))  # r*sqrt(2)
    vA = lines_matrix[:, 0, 0:2] + np.column_stack((-l * V(np.cos(theta + math.radians(45))),
                                                    -l * V(np.sin(theta + math.radians(45)))))
    vB = lines_matrix[:, 0, 0:2] + np.column_stack((-l * V(np.cos(math.radians(45) - theta)),
                                                    l * V(np.sin(math.radians(45) - theta))))
    vC = lines_matrix[:, 1, 0:2] + np.column_stack((l * V(np.cos(math.radians(45) - theta)),
                                                    -l * V(np.sin(math.radians(45) - theta))))
    vD = lines_matrix[:, 1, 0:2] + np.column_stack((l * V(np.cos(theta + math.radians(45))),
                                                    l * V(np.sin(theta + math.radians(45)))))
    cuboids = np.column_stack((vA, vB, vC, vD, lines_matrix[:, 0, 2]))
    # print_cuboids(cuboids)
    Timer.stop("reconstruct_cuboids")
    return cuboids


def t_generator(l):
    for num in l:
        yield np.linspace(0.0, 1.0, num)

def simple_unique_elmwise(ar):
    seen = set()
    indices  = []
    last = None
    same_as_last = 0

    for ndx, el in enumerate(ar):
        elt = tuple(el)
        if elt == last: same_as_last += 1
        last = elt
        if elt in seen:
            continue
        indices.append(ndx)
        seen.add(elt)

    print("same_as_last", same_as_last)

    return ar[indices]

def simple_unique_elements(ar):
    return np.array(list(set([tuple(x) for x in ar])))

def original_unique(ar):
    _, indices = np.unique(ar, axis=0, return_index=True)
    return ar[indices]

# aka dup_elim
def sparsify(sample, max_num_l, t_l, t_h, t_w, run_elmwise = False):
    ptx = len(t_h) * len(t_w)

    res = np.concatenate([sample[x] for x in
                          [slice(i*max_num_l*ptx, (i*max_num_l+len(l))*ptx)
                           for i, l in enumerate(t_l)
                           ]
                          ])

    if run_elmwise:
        return simple_unique_elmwise(res) # should be exactly the same result as elmwise, but faster
    else:
        return res

def proportional_sampling_cuboids(cuboids, num_l_list, num_w, num_h, dedup_mode = 'sparsify'):
    print('Start sampling.')
    Timer.start("proportional_sampling_cuboids")

    max_num_l = max(num_l_list)
    t_l = list(t_generator(num_l_list))
    t_l_p = np.zeros([len(t_l), max_num_l])  # pad t_l to become a numpy array
    for i,j in enumerate(t_l):
        t_l_p[i][0:len(j)] = j
    t_l_p = V(t_l_p)

    # print out sparsity level
    #print(sum([len(x) for x in t_l]), t_l_p.shape)

    t_w = V(np.linspace(0.0, 1.0, num_w))
    # to work with the functionality of rotation (adjusting z value at the end)
    if num_h == 1:
        t_h = V(np.array([0.2]))  # layer height is 0.2
    else:
        t_h = V(np.linspace(0.0, 0.2, num_h))
    sample = cuboids[:, 0:2][:, np.newaxis, :] + t_l_p[:, :, np.newaxis] * \
             (cuboids[:, 4:6] - cuboids[:, 0:2])[:, np.newaxis, :]  # shape : N x max_num_l x 2

    tmp = t_w[:, np.newaxis] * (cuboids[:, 2:4] - cuboids[:, 0:2])[:, np.newaxis, :]
    sample = sample[:, :, np.newaxis, :] + np.repeat(tmp[:, np.newaxis, :, :], max_num_l, axis=1)

    z_coord = np.repeat(cuboids[:, 8][:, np.newaxis, np.newaxis, np.newaxis], max_num_l * num_w, axis=1) \
        .reshape((sample.shape[0], max_num_l, num_w, 1))
    sample = np.concatenate((sample, z_coord), axis=3).reshape(-1, 3)  # shape : (N x max_num_l x num_w) x 3
    sample = np.repeat(sample[:, np.newaxis, :], num_h, axis=1)
    sample[:, :, 2] -= t_h  # note here, it may result in -0.0000000001 but is supposed to be 0
    sample = np.reshape(sample, (-1, 3))

    # Timer.start("opsbefore")
    # sample = sample.astype(float)
    # Timer.stop("opsbefore")

    print("Before removing replicates: ", sample.shape[0])

    # remove duplicates: very slow
    t_start = perf_counter()
    # sample = original_unique(sample)

    if dedup_mode == 'compare':
        # use this branch to test correctness
        sample1 = simple_unique_elmwise(sample)
        print("elmwise done")
        sample2 = sparsify(sample, max_num_l, t_l, t_h, t_w, run_elmwise = True)
        print("sparsify done")

        assert sample1.shape == sample2.shape, f"{sample1.shape} != {sample2.shape}"
        k = 0
        for i, (x, y) in enumerate(zip(sample1, sample2)):
            if not (x == y).all():
                if k < 10:
                    print(i, x, y, (x == y).all())
                k += 1

        print("mismatches", k) # approximate

        #assert all([(x == y).all() for x, y in zip(sample1, sample2)])
        print("sparsify matches elmwise")
        sample = sample2
    elif dedup_mode == 'elmwise':
        sample = simple_unique_elmwise(sample)
    elif dedup_mode == 'sparsify':
        print("using sparsify")
        # this produces slightly more points, but is orders of magnitude faster
        sample = sparsify(sample, max_num_l, t_l, t_h, t_w, run_elmwise = False)
    else:
        raise NotImplementedError(f"Unknown dedup mode: {dedup_mode}")

    t_end = perf_counter()
    print("Time for removing duplicates: ", t_end - t_start)
    print("After removing replicates: ", sample.shape[0])

    print('Finish sampling.')
    Timer.stop("proportional_sampling_cuboids")
    return sample

'''
def uniform_sampling_cuboids(cuboids, num_l, num_w, num_h):
    print('Start sampling.')
    t_l = V(np.linspace(0.0, 1.0, num_l))  # shape:(num_l,)
    t_w = V(np.linspace(0.0, 1.0, num_w))
    # to work with the functionality of rotation (adjusting z value at the end)
    if num_h == 1:
        t_h = np.array([0.2])  # layer height is 0.2
    else:
        t_h = V(np.linspace(0.0, 0.2, num_h))
    sample = cuboids[:, 0:2][:, np.newaxis, :] + t_l[:, np.newaxis] * \
             (cuboids[:, 4:6] - cuboids[:, 0:2])[:, np.newaxis, :]  # shape : N x num_l x 2
    tmp = t_w[:, np.newaxis] * (cuboids[:, 2:4] - cuboids[:, 0:2])[:, np.newaxis, :]
    sample = sample[:, :, np.newaxis, :] + np.repeat(tmp[:, np.newaxis, :, :], num_l, axis=1)

    z_coord = np.repeat(cuboids[:, 8][:, np.newaxis, np.newaxis, np.newaxis], num_l * num_w, axis=1) \
        .reshape((sample.shape[0], num_l, num_w, 1))
    sample = np.concatenate((sample, z_coord), axis=3).reshape(-1, 3)  # shape : (N x num_l x num_w) x 3
    sample = np.repeat(sample[:, np.newaxis, :], num_h, axis=1)
    sample[:, :, 2] -= t_h  # note here, it may result in -0.0000000001 but is supposed to be 0
    print('Finish sampling.')

    return np.reshape(sample, (-1, 3))
'''

# cuboid_vt indicates whether it also returns the vertices of all cuboids
def sample_points_cuboid(lines_matrix, num_l_list, num_w, num_h, radius, dedup_mode = 'sparsify'):
    with Timer.time("sample_points_cuboid"):
        cuboids = reconstruct_cuboids(lines_matrix, radius)
        pcd = proportional_sampling_cuboids(cuboids, num_l_list, num_w, num_h, dedup_mode = dedup_mode)
        return pcd


def divide_bounding_box(bbox, pcd1, pcd2, length, width, height):
    num_l = math.ceil((bbox[1][0] + N('1.0') - bbox[0][0]) / length)
    num_w = math.ceil((bbox[1][1] + N('1.0') - bbox[0][1]) / width)
    num_h = math.ceil((bbox[1][2] + N('0.2') - bbox[0][2]) / height)
    total_num_box = num_l * num_w * num_h

    # sorted_pcd1, sorted_pcd2 = [[] for _ in range(total_num_cuboid)], [[] for _ in range(total_num_cuboid)]
    sorted_pcd1, sorted_pcd2 = zip(*([[], []] for _ in range(total_num_box)))

    print("Total # of points in pcd1:", len(pcd1))
    print("Total # of points in pcd2:", len(pcd2))

    x_group = (pcd1[:, 0] + N('0.5') - bbox[0][0]) // length
    y_group = (pcd1[:, 1] + N('0.5') - bbox[0][1]) // width
    z_group = np.absolute((pcd1[:, 2] - bbox[0][2])) // height

    offset = x_group * num_w * num_h + y_group * num_h + z_group

    for i in tqdm(range(pcd1.shape[0]), desc='Sorting the first point cloud'):
        sorted_pcd1[int(offset[i])].append(pcd1[i].tolist())

    x_group = (pcd2[:, 0] + N('0.5') - bbox[0][0]) // length
    y_group = (pcd2[:, 1] + N('0.5') - bbox[0][1]) // width
    z_group = np.absolute((pcd2[:, 2] - bbox[0][2])) // height

    offset = x_group * num_w * num_h + y_group * num_h + z_group
    for i in tqdm(range(pcd2.shape[0]), desc='Sorting the second point cloud'):
        sorted_pcd2[int(offset[i])].append(pcd2[i].tolist())
    combined_pcd = [[a, b] for a, b in zip(sorted_pcd1, sorted_pcd2)]
    
    gc.collect()

    return combined_pcd, num_l, num_w, num_h


def visualize_divided_pcd(combined_pcd, index):
    tmp_pcd = combined_pcd.copy()
    trb_cube = tmp_pcd.pop(index)
    tmp_pcd = [pt for cube in tmp_pcd for pcd in cube for pt in pcd]
    draw_divided_pcd(trb_cube[0], trb_cube[1], tmp_pcd)

def compute_HD_single_box_pargen(i, pcd_1, pcd_2, aug_pcd_1, aug_pcd_2):
    if isinstance(aug_pcd_1, tuple):
        assert aug_pcd_1[0] == aug_pcd_2[0] and aug_pcd_1[0] is None
        aug_pcd_1 = list(generate_neighborhood(_combined_pcd_hack, *aug_pcd_1[1:]))
        aug_pcd_2 = list(generate_neighborhood(_combined_pcd_hack, *aug_pcd_2[1:]))

    return compute_HD_single_box(i, pcd_1, pcd_2, aug_pcd_1, aug_pcd_2)

# @profile
def compute_HD_single_box(i, pcd_1, pcd_2, aug_pcd_1, aug_pcd_2):
    # print(i)
    # print("Shape of pcd_1:", len(pcd_1))
    # print("Shape of pcd_2:", len(pcd_2))
    # print("Shape of aug_pcd_1:", len(aug_pcd_1))
    # print("Shape of aug_pcd_2:", len(aug_pcd_2))
    tree_1, tree_2 = KDTree(aug_pcd_1), KDTree(aug_pcd_2)
    dist_1, _ = tree_2.query(pcd_1, k=1)
    dist_2, _ = tree_1.query(pcd_2, k=1)
    HD = max(max(dist_1), max(dist_2))
    return (i, HD)

def compute_HD_aug_box_pargen(i, pcd_1, aug_pcd_2):
    if isinstance(aug_pcd_2, tuple):
        assert aug_pcd_2[0] is None
        aug_pcd_2 = list(generate_neighborhood(_combined_pcd_hack, *aug_pcd_2[1:]))

    return compute_HD_aug_box(i, pcd_1, aug_pcd_2)

def compute_HD_aug_box(i, pcd_1, aug_pcd_2):
    if len(aug_pcd_2) == 0:
        return (i, math.inf)
    else:
        tree = KDTree(aug_pcd_2)
        dist, _ = tree.query(pcd_1, k=1)
        HD = max(dist)
        return (i, HD)

def recover_3d_group(i, num_w, num_h):
    x_group = i // (num_w * num_h)
    offset_in_face = i % (num_w * num_h)
    y_group = offset_in_face // num_h
    z_group = offset_in_face % num_h
    return x_group, y_group, z_group

def construct_surrounding_list(i, num_l, num_w, num_h):
    neighbors = []
    (x_group, y_group, z_group) = recover_3d_group(i, num_w, num_h)
    possible_neighbors = [(x_group + dx, y_group + dy, z_group + dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]]
    for (x, y, z) in possible_neighbors:
        if 0 <= x < num_l and 0 <= y < num_w and 0 <= z < num_h:
            neighbors.append(x * num_w * num_h + y * num_h + z)
    return neighbors

def generate_neighborhood(combined_pcd, index_list, total_num_cuboid, group):
    for j in index_list:
        if 0 <= j <= total_num_cuboid - 1:
            for p in combined_pcd[j][group]:
                yield p


def compute_hausdorff_distance(combined_pcd, num_l, num_w, num_h):
    N = num_w * num_h
    total_num_cuboid = num_l * N
    print(f"Total number of boxes: {total_num_cuboid}")
    
    orig_HD = [None] * total_num_cuboid
    total_num_points = 0

    single_box_list = []  # boxes that both point clouds have points inside
    aug_box_list = []  # boxes that only one point cloud has points inside
    removed_box_list = []  # boxes that no point clouds have points inside
    
    for i in range(total_num_cuboid):
        len_1, len_2 = len(combined_pcd[i][0]), len(combined_pcd[i][1])
        total_num_points += (len_1 + len_2)
        
        if len_1 == 0 and len_2 == 0:
            removed_box_list.append(i)
        elif len_1 == 0:
            aug_box_list.append((i, 0))
        elif len_2 == 0:
            aug_box_list.append((i, 1))
        else:
            single_box_list.append(i)

    global _combined_pcd_hack
    _combined_pcd_hack = combined_pcd
    _mp_start_method = multiprocessing.get_start_method()
    _use_pargen = True

    def neigh_gen(*args):
        if _use_pargen and _mp_start_method == 'fork':

            # neighborhood will be generated in worker and
            # combined_pcd will be accessed through _combined_pcd_hack instead of as an arg to prevent serialization and
            # data transfer of combined_pcd

            return (None, *args[1:])
        else:
            return list(generate_neighborhood(*args))

    with multiprocessing.Pool(processes=32, maxtasksperchild=1) as pool:  # can adjust maxtaskperchild
        print("Start doing tasks ...")
        with Timer.time("items_single"):
            items_single = [(i, combined_pcd[i][0], combined_pcd[i][1], 
                             neigh_gen(combined_pcd, construct_surrounding_list(i, num_l, num_w, num_h), total_num_cuboid, 0),
                             neigh_gen(combined_pcd, construct_surrounding_list(i, num_l, num_w, num_h), total_num_cuboid, 1)
                             )
                            for i in single_box_list]

        with Timer.time("single_box"):
            result_1 = pool.starmap(compute_HD_single_box_pargen, items_single, chunksize=None)
            for pair in result_1:
                orig_HD[pair[0]] = pair[1]

        with Timer.time("items_aug"):
            items_aug = [(i, combined_pcd[i][1 - pid], 
                          neigh_gen(combined_pcd, construct_surrounding_list(i, num_l, num_w, num_h), total_num_cuboid, pid)
                          )
                         for (i, pid) in aug_box_list]

        with Timer.time("aug_box"):
            result_2 = pool.starmap(compute_HD_aug_box_pargen, items_aug, chunksize=None)
            for pair in result_2:
                orig_HD[pair[0]] = pair[1]

    _combined_pcd_hack = None

    with Timer.time("average_hd"):
        averaged_HD, inf_count = average_hd(orig_HD, num_l, num_w, num_h, total_num_cuboid)

    if inf_count > 0:
        print(f"Note: there are {inf_count} cubes with infinite hausdorff distance.")
    
    return combined_pcd, total_num_cuboid, orig_HD, averaged_HD, inf_count, removed_box_list


def draw_hd_distribution(orig_hd, avg_hd, inf_count, odir, fn):
    filtered_orig_hd = [i for i in orig_hd if i != None and i != math.inf]
    draw_distribution(filtered_orig_hd, inf_count, odir/f"{fn}_orig_dist.pdf")
    
    filtered_avg_hd = [i for i in avg_hd if i != None and i != math.inf]
    draw_distribution(filtered_avg_hd, inf_count, odir/f"{fn}_avg_dist.pdf")
    

def draw_hd_heatmap(combined_pcd, remove_ids, hd_list, percentile, odir, fn):
    cleaned_pcd = list(clean_PCD(combined_pcd, set(remove_ids)))  # exclude boxes with no points inside
    filtered_hd_list = [i for i in hd_list if i != None and i != math.inf]
    threshold_hd = np.percentile(filtered_hd_list, percentile)
    heatmap_transparency(cleaned_pcd, hd_list, threshold_hd, odir/f"{fn}_avg_thresh.pcd")
    

def bounding_box_dist(bbox, pcd1, pcd2, length, width, height, percentile, odir, fn=""):
    PCD, num_l, num_w, num_h = divide_bounding_box(bbox, pcd1, pcd2, length, width, height)
    combined_pcd, num_cuboid, orig_hd, avg_hd, inf_count, remove_ids = compute_hausdorff_distance(PCD, num_l, num_w, num_h)
    with Timer.time("draw_hd_distribution"):
        draw_hd_distribution(orig_hd, avg_hd, inf_count, odir, fn)
    with Timer.time("draw_hd_heatmap"):
        draw_hd_heatmap(combined_pcd, remove_ids, avg_hd, percentile, odir, fn)
    return combined_pcd, num_cuboid, avg_hd


def average_hd(dist_list, num_l, num_w, num_h, total_cuboids):
    '''
    Average out the distances to reduce discretization/quantization errors.
    If any nearby cube has inf/None value, ignore those cubes.
    If the current cube has inf/None value, don't average.
    '''
    avg_list = [0.] * len(dist_list)  # preserve None/inf, average the rest
    count_inf = 0
    
    for (i, hd) in enumerate(dist_list):
        if hd == None:
            avg_list[i] = None
        elif hd == math.inf:
            avg_list[i] = math.inf
            count_inf += 1
        else:
            count, sum = 0, 0
            for j in construct_surrounding_list(i, num_l, num_w, num_h):
                if 0 <= j <= total_cuboids - 1:
                    if dist_list[j] != None and dist_list[j] != math.inf:
                        count += 1
                        sum += dist_list[j]
            avg_list[i] = sum / count
    return avg_list, count_inf

# no None/Inf in dist_list
def draw_distribution(dist_list, num_inf, filepath):
    plt.figure()

    try:
        _, bins, _ = plt.hist(dist_list, bins='auto', label='Finite values')
    except np.core._exceptions._ArrayMemoryError:
        print("Too small bin width, ran out of memory, using sturges instead of auto")
        _, bins, _ = plt.hist(dist_list, bins='sturges', label='Finite values')

    if num_inf > 0:
        max_finite = max(dist_list) if dist_list else 1
        bin_width = bins[1] - bins[0]
        inf_x = max_finite + bin_width          # place the ∞ bar just beyond the last finite bin
        plt.bar(inf_x, num_inf, width=bin_width, color='red', label='∞')

    plt.xlabel("Distance")          # x-axis label
    plt.ylabel("# Unit boxes")       # y-axis label
    plt.legend()
    plt.tight_layout()              # keep everything inside the figure bounds
    plt.savefig(filepath)


def clean_PCD(combined_pcd, removed_id):
    for i, pcd in enumerate(combined_pcd):
        if i not in removed_id:
            yield pcd[0] + pcd[1]


def heatmap_transparency(cleaned_PCD, HD_list, threshold_HD, filepath):
    HD_without_none = [hd for hd in HD_list if hd != None]
    count_valid = len(HD_without_none)
    
    color_list = []
    cmap = plt.cm.Reds
    HD_without_inf_none = [hd for hd in HD_without_none if hd != math.inf]
    norm = mcolors.Normalize(vmin=threshold_HD, 
                             vmax=max(HD_without_inf_none))
    # norm = mcolors.Normalize(vmin=min(HD_without_inf_none), 
    #                          vmax=max(HD_without_inf_none))  # no thresholding
    for hd in HD_without_none:
        if math.isinf(hd):
            color_list.append(cmap(1.0)[:3])
        elif hd <= threshold_HD:
            color_list.append(cmap(0.0)[:3])
        else:
            color = cmap(norm(hd))
            color_list.append(color[:3])
    
    color_pcd = np.repeat(color_list, 
                            [len(cleaned_PCD[i]) for i in range(count_valid)],
                            axis=0
                            )

    pcd = {'x': [p[0] for box in cleaned_PCD for p in box],
            'y': [p[1] for box in cleaned_PCD for p in box],
            'z': [p[2] for box in cleaned_PCD for p in box],
            'red': color_pcd[:, 0],
            'blue': color_pcd[:, 1],
            'green': color_pcd[:, 2],
            }

    with Timer.time("PyntCloud"):
        hmap = PyntCloud(pd.DataFrame(pcd))

    with Timer.time("hmap_to_instance_open3d"):
        tmp = hmap.to_instance("open3d", mesh=True)

    with Timer.time("write_point_cloud"):
        o3d.io.write_point_cloud(str(filepath), tmp)
    print(f"Write point cloud to {filepath}")


def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n]


def color_individual_part(i, HD_list, combined_pcd):
    colored = o3d.geometry.PointCloud()
    for index, _ in HD_list[i]:
        PCD = o3d.geometry.PointCloud()
        PCD.points = o3d.utility.Vector3dVector(np.array(combined_pcd[index][0]))
        if i == 0:
            PCD.paint_uniform_color([0.486, 0.988, 0.0])
        elif i == 1:
            PCD.paint_uniform_color([0.337, 0.51, 0.012])
        elif i == 2:
            PCD.paint_uniform_color([1, 1, 0])
        elif i == 3:
            PCD.paint_uniform_color([1, 0.5, 0])
        elif i == 4:
            PCD.paint_uniform_color([1, 0, 0])
        colored += PCD
    return colored


def draw_divided_pcd(pcd1, pcd2, pcd3):
    PCD1, PCD2, PCD3 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    PCD1.points = o3d.utility.Vector3dVector(np.array(pcd1))
    PCD2.points = o3d.utility.Vector3dVector(np.array(pcd2))
    PCD3.points = o3d.utility.Vector3dVector(np.array(pcd3))
    PCD1.paint_uniform_color([1, 0.706, 0])
    PCD2.paint_uniform_color([0, 0.651, 0.929])
    PCD3.paint_uniform_color([0.5, 0.5, 0.5])
    # o3d.visualization.draw_geometries([PCD1, PCD2, PCD3])
    o3d.io.write_point_cloud("Visualize_max_HD.pcd", PCD1+PCD2+PCD3)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def visualize(pcd, filename):
    PCD = o3d.geometry.PointCloud()
    PCD.points = o3d.utility.Vector3dVector(np.array(pcd))
    PCD.paint_uniform_color([0, 0.651, 0.929])
    o3d.io.write_point_cloud(filename, PCD)


def visualize_two_pcd(pcd1, pcd2, filename):
    PCD1, PCD2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    PCD1.points = o3d.utility.Vector3dVector(np.array(pcd1))
    PCD2.points = o3d.utility.Vector3dVector(np.array(pcd2))
    PCD1.paint_uniform_color([1, 0.706, 0])
    PCD2.paint_uniform_color([0, 0.651, 0.929])
    o3d.io.write_point_cloud(filename, PCD1+PCD2)
    print(f"Write point cloud to {filename}")


def save_combined_pcd(combined_PCD, file, dec_or_float):
    pcd_dict = {f'pcd_{i}_{j}': np.array(pcd, dtype=Decimal if dec_or_float else float) 
                    for i, box in enumerate(combined_PCD) 
                    for j, pcd in enumerate(box)}
    np.savez_compressed(file, **pcd_dict)


def parse_bbox(b):
    return [triple(coord) for coord in b]

def triple(t):
    return [float(n) for n in t.split(',')]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Compare two Gcode files.')
    
    parser.add_argument('--prusa', action='store_true', help='use this flag if the first G-code is from PrusaSlicer')
    parser.add_argument('imd_file', type=str, help='provide the filepath of the intermediate gcode')
    parser.add_argument('imd_ot', type=triple, help='orientation for the intermediate model')
    
    parser.add_argument('rot_file', type=str, help='provide the filepath of the rotated gcode')
    parser.add_argument('rot_ot', type=triple, help='orientation for the rotated model')
    
    parser.add_argument('orig_dim', type=triple, help='provide the dimension of the original model')
    parser.add_argument('-c', '--center', nargs=2, type=float, default=[150., 150.],
                        help='provide the xy coordinates of the rotation center. Default is (150,150)')
    
    parser.add_argument('-s', '--sampling', type=float, default=0.1, 
                        help='provide the sampling interval')
    parser.add_argument('-b', '--boxsize', nargs=3, default=[1., 1., 1.], 
                        help='provide the length, width, and height of the box used in generating heatmap')
    parser.add_argument('-t', '--threshold', type=float, default=90., 
                        help='provide the threshold (in percentile) used in heatmap generation')
    parser.add_argument('-d', '--decimal', type=int, default=0, help='0/1 for float/Decimal computation')
    parser.add_argument('--bound', nargs=2, help='provide bounding box for the two gcodes in the format of'
                        'bmin_x,bmin_y,bmin_z bmax_x,bmax_y,bmax_z')
    parser.add_argument('--name', type=str, help='provide name for generated heatmap and distributions')
    parser.add_argument('--collect', help='JSON file to collect results')
    parser.add_argument('--rerun', help='provide the JSON file storing the point clouds for rerunning')
    parser.add_argument('--dedup-mode', help="Specify PCD dedup mode", choices=['sparsify', 'elmwise', 'compare'], default='sparsify') # compare is for correctness debugging

    args = parser.parse_args()
    set_global_OP(args.decimal)

    # profiler = cProfile.Profile()
    # profiler.enable()

    # operate on the same point clouds but with different parameters
    # e.g., boxsize, points density, threshold percentile
    # json file needs to contain the point clouds
    if args.rerun:
        with open(args.rerun, 'rt') as f:
            results = json.load(f)
        
        if (args.orig_dim != results['original model dimension'] or
            args.imd_file != results['intermediate g-code']['filepath'] or
            args.imd_ot != results['intermediate g-code']['rotation degrees'] or
            args.rot_file != results['rotated g-code']['filepath'] or
            args.rot_ot != results['rotated g-code']['rotation degrees'] or
            args.sampling != results['sampling interval']):
                exit("Cannot rerun: experiment info does not match")
        
        if args.bound is not None:
            oa_bbox = parse_bbox(args.bound)
        else:
            oa_bbox = results['bounding box']
        
        # load point clouds
        with np.load(results['point clouds']) as pcds:
            pcd_1 = pcds['pcd1']
            pcd_2 = pcds['pcd2']
        
        # fn = Path(args.orig_file).stem
        # visualize_two_pcd(pcd_1, pcd_2, f"{fn}_{args.sampling}.pcd")

        t2_start = perf_counter()
        
        combined_pcd, num_boxes, hd_list = bounding_box_dist(
            bbox=oa_bbox, 
            pcd1=pcd_1, 
            pcd2=pcd_2, 
            length=N(args.boxsize[0]),
            width=N(args.boxsize[1]), 
            height=N(args.boxsize[2]),
            percentile=args.threshold,
            odir=Path(args.collect).parent if args.collect else Path(__file__).parent
        )
        
        t2_end = perf_counter()
        time_2 = t2_end - t2_start

        if args.collect:
            results['boxsize'] = args.boxsize
            results['threshold percentile'] = args.threshold
            results['bounding box'] = oa_bbox
            results['number of boxes'] = num_boxes
            results['hausdorff distance'] = hd_list
            save_combined_pcd(combined_pcd, results['combined points'], args.decimal)
            results['Time 2'] = time_2

            with open(args.collect, 'wt') as f:
                json.dump(results, f)

        exit("Finish rerun")

# Start with reading g-code files
    t1_start = perf_counter()
    if args.prusa:
        orig_lines, orig_bbox = read_gcode_from_prusa(args.imd_file)
    else:
        orig_lines, orig_bbox = read_gcode_from_cura(args.imd_file)
    rot_lines, rot_bbox = read_gcode_from_cura(args.rot_file)
    print("Total number of lines in orig_file: ", orig_lines.shape[0])
    print("Total number of lines in rot_file: ", rot_lines.shape[0])
    
    orig_h = args.orig_dim[2]
    orig_center = V([args.center[0], args.center[1], orig_h / 2])
    # print(f"original center: {orig_center}")

    if args.bound is not None:
        oa_bbox = parse_bbox(args.bound)
    elif args.prusa:
        oa_bbox = rot_bbox.tolist()  #TODO: compute bounding box for prusa's g-code
    else:
        undo_rot_bbox = align_bbox(rot_bbox, args.rot_ot, args.imd_ot)
        oa_bbox = overall_bbox([undo_rot_bbox, orig_bbox])

    ''' Uniform Sampling '''
    '''
    max_length_1 = max_length_of_paths(orig_lines)
    max_length_2 = max_length_of_paths(rot_lines)
    num_length_1 = math.ceil((float(max_length_1) + 0.4) / args.sampling) + 1
    num_length_2 = math.ceil((float(max_length_2) + 0.4) / args.sampling) + 1
    num_width = math.ceil(0.4 / args.sampling) + 1
    num_height = math.ceil(0.2 / args.sampling) + 1
    Points_1 = sample_points_cuboid(orig_lines, num_length_1, num_width, num_height, radius=N('0.2'))
    rot_Points_1 = rotate_points_3d(Points_1, args.angle, center_list)    
    Points_2 = sample_points_cuboid(rot_lines, num_length_2, num_width, num_height, radius=N('0.2'))
    '''
    
    ''' Proportional Sampling '''
    length_1 = length_of_paths(orig_lines)
    length_2 = length_of_paths(rot_lines)

    Timer.start("num_length")
    num_length_1 = np.ceil((length_1.astype(float) + 0.4) / args.sampling) + 1
    num_length_2 = np.ceil((length_2.astype(float) + 0.4) / args.sampling) + 1
    Timer.stop("num_length")

    num_width = math.ceil(0.4 / args.sampling) + 1
    num_height = math.ceil(0.2 / args.sampling) + 1
    Points_1 = sample_points_cuboid(orig_lines, num_length_1.astype(int), 
                                    num_width, num_height, radius=N('0.2'),
                                    dedup_mode=args.dedup_mode)
    Points_2 = sample_points_cuboid(rot_lines, num_length_2.astype(int), 
                                    num_width, num_height, radius=N('0.2'),
                                    dedup_mode=args.dedup_mode)
    
    Timer.start("rotation")
    # recover to original first then perform the same rotation as for the intermediate
    orig_imd_d = align_to_build_plate(args.orig_dim, args.imd_ot, orig_center)
    orig_rot_d = align_to_build_plate(args.orig_dim, args.rot_ot, orig_center)
    rot_Points_2 = align_pcd(Points_2, orig_rot_d, args.rot_ot, orig_center,
                             args.imd_ot, orig_imd_d)
    Timer.stop("rotation")
    
    # visualize_two_pcd(Points_1, rot_Points_2, "visualization_two_pcds.pcd")
    # exit(0)

    t1_end = perf_counter()
    time_1 = t1_end - t1_start
    
    gc.collect()
    
    t2_start = perf_counter()
    
    dir = Path(args.collect).parent
    # dir.mkdir(parents=True, exist_ok=True)
    combined_pcd, num_boxes, hd_list = bounding_box_dist(
        oa_bbox, 
        Points_1, rot_Points_2, 
        length=N(args.boxsize[0]), 
        width=N(args.boxsize[1]), 
        height=N(args.boxsize[2]),
        percentile=args.threshold,
        odir=dir if args.collect else Path(__file__).parent,
        fn=args.name
    )
    
    t2_end = perf_counter()
    time_2 = t2_end - t2_start
    
    print("Time Elapsed up to sampling and rotating:", time_1)
    print("Time Elapsed up to generating heatmap:", time_2)
    print("Total Time Elapsed:", time_1 + time_2)

    if args.collect:        
        Timer.start("collect json")
        
        dir = Path(args.collect).parent
        fn = args.name
        
        # Timer.start("saving pcds")
        # pcds_fn = dir/f"{fn}_pcds.npz"
        # np.savez_compressed(pcds_fn, pcd1=Points_1, pcd2=rot_Points_2)
        # Timer.stop("saving pcds")
        
        Timer.start("saving combined pcd")
        # saved combined pcd must NOT be flattened and cleaned
        # otherwise original pcd would be repeatedly drawn later
        # and each combined pcd won't align
        cpcd_fn = dir/f"{fn}_cpcd.npz"
        save_combined_pcd(combined_pcd, cpcd_fn, args.decimal)
        
        Timer.stop("saving combined pcd")
        
        results = {'original model dimension': args.orig_dim,
                   'intermediate g-code':
                    {'filepath': args.imd_file,
                     'rotation degrees': args.imd_ot,
                     'number of lines': orig_lines.shape[0],
                     'number of points': Points_1.shape[0]  # after removing duplicates
                     }, 
                   'rotated g-code':
                    {'filepath': args.rot_file,
                     'rotation degrees': args.rot_ot,
                     'number of lines': rot_lines.shape[0],
                     'number of points': Points_2.shape[0]
                     },
                   'sampling interval': args.sampling,
                   'boxsize': args.boxsize,
                   'threshold percentile': args.threshold,
                #    'point clouds': pcds_fn.as_posix(),
                   'bounding box': oa_bbox,
                   'number of boxes': num_boxes,
                   'hausdorff distance': hd_list,
                   'heatmap': (dir/f"{fn}_avg_thresh.pcd").as_posix(),
                   'original distribution': (dir/f"{fn}_orig_dist.pdf").as_posix(),
                   'averaged distribution': (dir/f"{fn}_avg_dist.pdf").as_posix(),
                   'combined points': cpcd_fn.as_posix(),
                   'Time 1': time_1,
                   'Time 2': time_2
                }
        
        with open(args.collect, 'wt') as f:
            json.dump(results, f, indent=4)
        
        Timer.stop("collect json")

    # profiler.disable()
    # profiler.print_stats(sort='cumulative')
