from dataloader import *
import numpy as np
from scipy.signal import find_peaks

class KPData():
    def __init__(self, path, start, end, idx, all_files):
        joints, ids = kp_loader(path, start, end, all_files)
        self.joints = joints
        self.ids = ids
        self.height = self.calculate_height(idx)
        self.accel = self.calculate_accel(self.height)
    
    def calculate_accel(self, height):
        accel = [1]
        for i in range(1, len(height)-1):
            delta = height[i+1] + height[i-1] - 2 * height[i]
            accel_value = delta*25*25/100/9.81 + 1      # Accel data is unit of 'g'
            if abs(accel_value) > 20:                   # Outlier value when the position becomes (0, 0, 0)
                accel_value = 1                         # Set 1
            accel.append(accel_value)
        accel.append(1)
        return accel 

    def calculate_height(self, idx):
        height = []
        len_data = self.joints[0].shape[0]
        for frame in range(len_data):
            height.append(-1*self.joints[idx][frame][2][1])
        return height

    def peak(self, accel):
        idx = []
        idx = find_peaks(accel, height=2)[0]
        return idx



class IMUData():
    def __init__(self, path, imu_file, anno_file):
        self.data = imu_loader(path+imu_file)
        self.anno = imu_anno_loader(path+anno_file)
        self.data = self.init_time(self.segment_by_data_collection())

    def fit_time_to_anno(self, time_data, time_anno, idx):
        data_round = [int(time_data[i]/10) for i in range(len(time_data))]
        anno_round = int(time_anno/10)
        dif = []
        try:
            start_idx = data_round.index(anno_round)
        except:
            return False
        data_sim = time_data[start_idx:start_idx+data_round.count(anno_round)]
        for i in range(data_round.count(anno_round)):    
            dif.append(abs(data_sim[i] - time_anno))
        time_sim = data_sim[dif.index(min(dif))]
        idx = idx + np.where(time_data == time_sim)[0].item()
        return idx

    def segment_by_data_collection(self):
        time = self.data[0]
        idx = 0
        start, end, segm_data = [[], [], []]
        for i in range(len(self.anno[0])):
            if self.anno[0][i] == 'Data Collection':
                idx = self.fit_time_to_anno(time[idx:], int(self.anno[1][i]), idx)
                if not idx:
                    break
                else:
                    start.append(idx)
                idx = self.fit_time_to_anno(time[idx:], int(self.anno[2][i]), idx)
                end.append(idx)
                
        for i in range(len(start)):
            tmp = [self.data[j][start[i]:end[i]] for j in range(self.data.shape[0])]
            segm_data.append(np.array(tmp))
            
        return segm_data

    def init_time(self, data):
        for i in range(len(data)):
            time = data[i][0]
            time = (time - time[0])/1000
            data[i][0] = time
        return data

    def peak(self):
        idx = []
        for i in range(len(self.data)):
            idx.append(find_peaks(self.data[i][2], height=2.3)[0])
        return idx