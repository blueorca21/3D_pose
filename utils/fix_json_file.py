from Data import KPData
from dataloader import kp_loader
import numpy as np
import os
import json

def main():
    date = '190510'
    sequence = 2

    kp_path = '../Dataset/keypoints/' + date + '/'
    _, folders, _ = next(os.walk(kp_path))
    folders.sort()
    kp_path += folders[sequence] + '/op25_body3DPSRecon_json_normCoord/0140/'
    joints, ids = kp_loader(kp_path, '00000100', '00000100', all_files=True)
    
    '''
    numpy_file_name = date + '_sequence_' + str(sequence+1) + '_'
    np.save(numpy_file_name+'set1', joints[0])
    np.save(numpy_file_name+'set2', joints[1])
    import pdb; pdb.set_trace()
    '''

    
    write_path = '../../../Data/imudome_fixed/' + date +'/sequence_' + str(sequence+1) + '/'
    for i in range(joints[0].shape[0]):
        file_name = str(i+100)
        for _ in range(8-len(file_name)):
            file_name = '0' + file_name
        file_name = 'body3DScene_' + file_name + '.json'
        
        file_info = dict()
        file_info["version"] = 0.7
        file_info["univTime"] = -1.000
        file_info["fpsType"] = "vga_25"
        file_info["vgaVideoTime"] = 0.000
        file_info["bodies"] = []
        for j in range(len(joints)):
            ids = dict()
            ids["id"] = j
            joints26 = []
            for k in range(joints[0].shape[1]):
                for l in range(4):
                    joints26.append(joints[j][i][k][l])
            ids["joints26"] = joints26
            file_info["bodies"].append(ids)

        with open(write_path+file_name, 'w', encoding='utf-8') as make_file:
            json.dump(file_info, make_file, indent='\t')

if __name__ == '__main__':
    main()