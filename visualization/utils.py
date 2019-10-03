import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def make_part_dict():
    part = {}
    part['face'] = [18, 17, 1, 15, 16, 'crimson']
    part['neck'] = [0, 1, 'maroon']
    part['back'] = [2, 0, 'darkred']
    part['larm1'] = [5, 4, 'forestgreen']
    part['larm2'] = [4, 3, 'limegreen']
    part['larm3'] = [3, 0, 'springgreen']
    part['rarm1'] = [11, 10, 'gold']
    part['rarm2'] = [10, 9, 'orange']
    part['rarm3'] = [9, 0, 'darkorange']
    part['hip'] = [12, 2, 6, 'purple']
    part['rleg1'] = [14, 13, 'darkblue']
    part['rleg2'] = [13, 12, 'midnightblue']
    part['lleg1'] = [8, 7, 'seagreen']
    part['lleg2'] = [7, 6, 'darkgreen']
    part['rfoot'] = [22, 23, 14, 24, 'mediumblue']
    part['lfoot'] = [19, 20, 8, 21, 'mediumseagreen']

    return part

def open_json_file(file_name):
    id = []
    joints = []
    with open(file_name) as json_file:
        data = json.load(json_file)
        for body in data["bodies"]:
            id.append(body['id'])
            joints.append(np.array(body['joints26']).reshape(1, -1,4))
    
    try: 
        joints = np.vstack(joints)
    except:
        pass
    return id, joints

def set_range(x, y, z, ax, animation=False):
    if not animation:
        tmp_x, tmp_y, tmp_z = [[], [], []]
        for i in range(len(x)):
            if (x[i] != 0 and y[i] != 0 and z[i] != 0):
                tmp_x.append(x[i].item())
                tmp_y.append(y[i].item())
                tmp_z.append(z[i].item())
        x, y, z = [np.array(tmp_x), np.array(tmp_y), np.array(tmp_z)]
        max_range = 1.5 * np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max() + x.min())/2.0
        mid_y = (y.max() + y.min())/2.0
        mid_z = (z.max() + z.min())/2.0
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        ax.set_xlim(-200, 400)
        ax.set_ylim(-500, 000)
        ax.set_zlim(-450, -100)

def draw_scatter(x, y, z, ax):
    for j in range(len(x)):
        if (x[j] != 0 and y[j] != 0 and z[j] != 0):
            #ax.scatter(x[j], y[j], z[j], c='r', marker='^')
            ax.text(x[j], y[j], z[j], '%s' % (str(j)), size=5)

def draw_line(x, y, z, ax, part):
    for name_part in part:
        part_x, part_y, part_z = [[], [], []]
        for index in part[name_part][:-1]:
            if (x[index] != 0 and y[index] != 0 and z[index] != 0):
                part_x.append(x[index])
                part_y.append(y[index])
                part_z.append(z[index])
        ax.plot3D(part_x, part_y, part_z, part[name_part][-1])

def draw_ground(ax):
    X = np.arange(-1000, 1500, 500)
    Z = np.arange(-1000, 1500, 500)
    X, Z = np.meshgrid(X,Z)
    Y = np.zeros(X.shape)

    ax.plot_surface(X, Y, Z, alpha=0.1, linewidth=0, antialiased=False)