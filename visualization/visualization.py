import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import utils

def plot_3d_line(ids, joints):
    part = utils.make_part_dict()

    for i in range(len(ids)):
        globals()['fig%s' %i] = plt.figure()
        ax = plt.axes(projection='3d')
        x, y, z, conf = [array.reshape(26) for array in np.split(joints[i], 4, axis=1)]
        ax.set_title('id = %s' %ids[i])
        utils.set_range(x, y, z, ax)
        utils.draw_scatter(x, y, z, ax)
        utils.draw_line(x, y, z, ax, part)
        ax.view_init(azim = -90,elev = -50)

def main():
    file_name = 'body3DScene_00000100.json'
    path = '../Dataset/Dome_3D/190503_imu5/op25_body3DPSRecon_json_normCoord/0140'
    ids, joints = utils.open_json_file(os.path.join(path, file_name))
    plot_3d_line(ids, joints)
    plt.show()

if __name__ == '__main__':
    main()