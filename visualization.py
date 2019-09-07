import numpy as np
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def open_json_file(file_name):
    id = []
    joints = []
    with open(file_name) as json_file:
        data = json.load(json_file)
        for body in data["bodies"]:
            id.append(body['id'])
            joints.append(np.array(body['joints26']).reshape(1, -1,4))
    
    joints = np.vstack(joints)
    return id, joints

def plot_3d_line(ids, joints):
    part = {}
    part['face'] = [17, 15, 0, 16, 18]
    part['neck'] = [0, 1]
    part['larm'] = [4, 3, 2, 1]
    part['rarm'] = [1, 5, 6, 7]
    part['lleg'] = [11, 10, 9, 8]
    part['rleg'] = [8, 12, 13, 14]
    part['lfoot'] = [23, 22, 11, 24]
    part['rfoot'] = [20, 19, 14, 21]
    for i in range(len(ids)):
        globals()['fig%s' %i] = plt.figure()
        ax = plt.axes(projection='3d')
        x, y, z, conf = np.split(joints[i], 4, axis=1)
        x = x.reshape(26)
        y = y.reshape(26)
        z = z.reshape(26)
        ax.set_title('ids = %s' %ids[i])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        for name_part in part:
            part_x = []
            part_y = []
            part_z = []
            for index in part[name_part]:
                part_x.append(x[index])
                part_y.append(y[index])
                part_z.append(z[index])
            ax.plot3D(part_x, part_y, part_z)
    plt.show()

def plot_3d_Scatter(ids, joints):
    for i in range(len(ids)):
        globals()['fig%s' %i] = plt.figure()
        ax = plt.axes(projection='3d')
        x, y, z, conf = np.split(joints[i], 4, axis=1)
        ax.plot3D(x.reshape(26), y.reshape(26), z.reshape(26), 'ro')
        ax.set_title('ids = %s' %ids[0])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    plt.show()

def main():
    file_name = 'body3DScene_00009890.json'
    path = './imudome/190503_imu1/op25_body3DPSRecon_json_normCoord/0140'
    ids, joints = open_json_file(os.path.join(path, file_name))
    plot_3d_line(ids, joints)

if __name__ == '__main__':
    main()
