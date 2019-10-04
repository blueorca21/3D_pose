import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import utils
import cv2
sys.path.insert(0, '/Users/soyongshin/Desktop/Research/Codes/3D_pose/utils')
from dataloader import kp_loader


def single_person_plot(path, file_name, animation=False):
    if not animation:
        ids, joints = utils.open_json_file(os.path.join(path, file_name))
    part = utils.make_part_dict()
    for i in range(len(ids)):
        globals()['fig%s' %i] = plt.figure()
        ax = plt.axes(projection='3d')
        x, y, z, conf = [array.reshape(26) for array in np.split(joints[i], 4, axis=1)]
        #ax.set_title('id = %s' %ids[i])
        utils.set_range(x, y, z, ax, animation)
        utils.draw_scatter(x, y, z, ax)
        utils.draw_line(x, y, z, ax, part)
        ax.view_init(azim = -90,elev = -90)

def multi_people_plot(ids, joints, animation=False):    
    # Make the segments with neighbor joints
    part = utils.make_part_dict()
    
    plt.figure()
    ax = plt.axes(projection = '3d')    
    ax.set_axis_off()
    if not joints == []:
        tmp_joints = joints.reshape(-1,4)
        tmp_x, tmp_y, tmp_z, _ = [array.reshape(-1,) for array in np.split(tmp_joints, 4, axis=1)]
        utils.set_range(tmp_x, tmp_y, tmp_z, ax, animation)
    else:
        pass
    
    # Draw ground
    utils.draw_ground(ax)
    
    # Set viewpoint
    ax.view_init(azim = -90,elev = -50)
    for i in range(len(ids)):
        x, y, z, _ = [array.reshape(26) for array in np.split(joints[i], 4, axis=1)]
        utils.draw_line(x, y, z, ax, part)
        str_id = 'id : ' + str(i)
        ax.text(x[25],y[25]-10,z[25], str_id, color='red')
    

def animation(path, start, end, all_files=False):
    if all_files:
        start = '00000100'
    img = []
    joints, ids = kp_loader(path, start, end, all_files=all_files)    
    for i in range(joints[0].shape[0]):
        joint = np.array([joints[0][i], joints[1][i]])
        multi_people_plot(ids, joint, True)
        file_name = str(int(start) + i)
        plt.title(file_name)
        plt.savefig('output.png')
        plt.close()
        im = cv2.imread('output.png')
        img.append(im)
        if i%10 == 0:
            print('%d iteration has been passed' %i)
       
    height, width, l = im.shape
    video = cv2.VideoWriter('animation.avi', 0, 25, (width,height))

    for image in img:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
    
def main():
    # Path by the date and sequence of data
    date = '190503'
    path = '../Dataset/keypoints/' + date
    sequence_id = 1
    _, folders, _ = next(os.walk(path))
    folders.sort()
    path += '/'+folders[sequence_id] + '/op25_body3DPSRecon_json_normCoord/0140/'
    #Video generation
    animation(path, '00000100', '00000300', all_files=False)
    

if __name__ == '__main__':
    main()