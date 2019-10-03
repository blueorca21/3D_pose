import os
import numpy as np
import sys
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/soyongshin/Desktop/Research/Codes/3D_pose/visualization')
from utils import open_json_file

def imu_anno_loader(file_name):
    event, start_time, end_time, value = [[], [], [], []]
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            event.append(row[2])
            start_time.append(row[4])
            end_time.append(row[5])
            value.append(row[6])
        line_count += 1
    
    return event, start_time, end_time, value
            

def imu_loader(file_name):
    t, x, y, z = [[],[],[], []]
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                t.append(float(row[0])/1000)
                x.append(float(row[1]))
                y.append(float(row[2]))
                z.append(float(row[3]))
            line_count += 1
    data = np.vstack([np.array(t), np.array(x), np.array(y), np.array(z)])
    return data
    

def check_new_id(org, new):
    if len(org) == 0:
        for i in new:
            if i != 0 and i !=1:
                new.remove(i)
        return new, new
    
    if len(new) > len(org):
        return org, []
    
    for i in org:
        if new.count(i) == 0:
            only_org = i
    
    for i in new:
        if org.count(i) == 0 and i != -1:
            only_new = i
    
    try:
        org[org.index(only_org)] = only_new
    except:
        pass
    return org, []

def kp_loader(path, start, end, all_files=False):
    if all_files:
        try:
            _, _, f = next(os.walk(path))
        except:
            import pdb; pdb.set_trace()
        sequence_length = len(f)
    else:
        sequence_length = int(end) - int(start) + 1
    file_name_default = 'body3DScene_'
    ids = []
    joints = []
    for i in range(sequence_length):
        name = str(int(start) + i)
        for o in range(len(end) - len(name)):
            name = '0' + name
        file_name = file_name_default + name + '.json'
        tmp_ids, tmp_joints = open_json_file(os.path.join(path, file_name))
        ids, new_index = check_new_id(ids, tmp_ids)
        for idx in ids:
            if ids == new_index:
                joints.append(tmp_joints[ids.index(idx)].reshape(-1,26,4))
            else:
                if tmp_ids.count(idx) != 0:
                    joints[ids.index(idx)] = np.vstack([
                        joints[ids.index(idx)], tmp_joints[tmp_ids.index(idx)].reshape(-1, 26, 4)
                    ])
                else:
                    joints[ids.index(idx)] = np.vstack([
                        joints[ids.index(idx)], np.zeros([1,26,4])
                    ])

    return joints, ids