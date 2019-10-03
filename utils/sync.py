import matplotlib.pyplot as plt
import numpy as np
from Data import KPData, IMUData
import os


def refine_peaks(kp_peaks, imu_peaks):
    kp_delete, imu_delete = [[], []]
    for i in range(kp_peaks.shape[0]):
        if i == 0:
            if kp_peaks[i+1] - kp_peaks[i] > 25:
                kp_delete.append(i)
        elif i == kp_peaks.shape[0]-1:
            if kp_peaks[i] - kp_peaks[i-1] > 25:
                kp_delete.append(i)
        else:
            if kp_peaks[i+1] - kp_peaks[i] > 25:
                if kp_peaks[i] - kp_peaks[i-1] > 25:
                    kp_delete.append(i)
    
    for i in range(imu_peaks.shape[0]):
        if i == 0:
            if imu_peaks[i+1] - imu_peaks[i] > 125:
                imu_delete.append(i)
        elif i == imu_peaks.shape[0]-1:
            if imu_peaks[i] - imu_peaks[i-1] > 125:
                imu_delete.append(i)
        else:
            if imu_peaks[i+1] - imu_peaks[i] > 125:
                if imu_peaks[i] - imu_peaks[i-1] > 125:
                    imu_delete.append(i)
    kp_peaks = np.delete(kp_peaks, kp_delete)
    imu_peaks = np.delete(imu_peaks, imu_delete)

    return kp_peaks, imu_peaks


def save_plot(kp,imu,sequence, zoom_time_front, zoom_time_end):
    x_label = [i/25 + imu.data[sequence][0][0] for i in range(len(kp.height))]
    plt.figure(figsize=(21,5))
    ax1 = plt.subplot(1,3,1)
    ax1.plot(x_label, kp.accel, label='Video data')
    ax1.plot(imu.data[sequence][0], imu.data[sequence][2], label='IMU data')
    ax1.legend()
    
    ax2 = plt.subplot(1,3,2)
    ax2.plot(x_label, kp.accel, label='Video data')
    ax2.plot(imu.data[sequence][0], imu.data[sequence][2], label='IMU data')
    ax2.set_xlim((zoom_time_front-1, zoom_time_front+3))
    ax2.set_ylim((-1, 8))

    ax3 = plt.subplot(1,3,3)
    ax3.plot(x_label, kp.accel, label='Video data')
    ax3.plot(imu.data[sequence][0], imu.data[sequence][2], label='IMU data')
    ax3.set_xlim((zoom_time_end-3, zoom_time_end+1))
    ax3.set_ylim((-1, 8))

    ax1.title.set_text('Full-size plot')
    ax2.title.set_text('Begin Jump')
    ax3.title.set_text('End Jump')
    file_name = 'Sequence_' + str(sequence) + '.png'
    plt.savefig(file_name)
    plt.show()


def validation_by_plot(kp, imu, sequence):
    x_label = [i/25 + imu.data[sequence][0][0] for i in range(len(kp.height))]
    plt.figure()
    plt.plot(x_label, kp.accel, label='Video data')
    plt.plot(imu.data[sequence][0], imu.data[sequence][2], label='IMU data')
    plt.legend()
    plt.show()


def peak_syncing(kp, imu, sequence = 0):
    kp_peaks = kp.peak(kp.accel)       # Keypoints peaks
    imu_peaks = imu.peak()
    kp_peaks, imu_peaks[sequence] = refine_peaks(kp_peaks, imu_peaks[sequence])
    begin_diff_list = [imu_peaks[sequence][i] - kp_peaks[i]*5 for i in range(1)]
    begin_diff = int(np.array(begin_diff_list).mean())
    tmp_imu = [imu.data[sequence][i][begin_diff:] for i in range(imu.data[sequence].shape[0])]
    imu.data[sequence] = np.vstack(tmp_imu)    
    end_diff = (-1)* (imu.data[sequence][0].shape[0] - 5 * (len(kp.height) -1) - 1)
    if end_diff > 0:
        end_diff = -1
    tmp_imu = [imu.data[sequence][i][:end_diff] for i in range(imu.data[0].shape[0])]
    imu.data[sequence] = np.vstack(tmp_imu)
    #validation_by_plot(kp, imu, sequence)
    imu_peaks = imu.peak()
    kp_peaks, imu_peaks[sequence] = refine_peaks(kp_peaks, imu_peaks[sequence])
    zoom_time_front = imu.data[sequence][0][imu_peaks[sequence][0]]
    zoom_time_end = imu.data[sequence][0][imu_peaks[sequence][-1]]
    save_plot(kp, imu, sequence, zoom_time_front, zoom_time_end)


def main():
    date = '190510'
    sequence = 7
    subject_id = 1

    kp_start, kp_end = ['00000100', '00000500']
    kp_path = '../Dataset/keypoints/' + date +'/'
    _, folders, _ = next(os.walk(kp_path))
    folders.sort()
    kp_path += folders[sequence] + '/op25_body3DPSRecon_json_normCoord/0140/'
    imu_path = '../Dataset/IMU/'
    _, folders, _ = next(os.walk(imu_path))
    folders.sort()
    imu_path += folders[subject_id] + '/'
    imu_file = date + '/sacrum/accel.csv'
    _, _, filename = next(os.walk(imu_path))
    imu_anno_file = filename[1]

    imu = IMUData(imu_path, imu_file, imu_anno_file)
    kp = KPData(kp_path, kp_start, kp_end, subject_id, True)
    peak_syncing(kp, imu, sequence=sequence)
    csv_file = date + '-sacrum-' + str(sequence) + '.csv'
    np.savetxt(csv_file, imu.data[sequence], delimiter=',')

if __name__ == '__main__':
    main()