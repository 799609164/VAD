# -*- coding: utf-8 -*-
import librosa
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import python_speech_features as psf

# 计算帧能量
def calEnergy(y_frames):
    energy = np.sum(np.power(y_frames,2),axis=0)
    energy_normal = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))

    return energy_normal

# 计算过零率
def calZeroCrossingRate(y_frames):
    crossings = librosa.zero_crossings(y_frames,axis=0)
    zcr = np.mean(crossings, axis=0, keepdims=True)

    return np.squeeze(zcr)

# 根据短时能量、短时过零率，进行端点检测
def endPointDetect(y, energy, zcr):
    energy_mean = np.mean(energy)  
    sum = np.sum(energy[-5:])   # 取后5帧能量

    ML = sum / 5           
    MH = energy_mean / 2 # 较高的能量阈值：总能量均值的1/2
    # MH = energy_mean  # 信噪比较小（snr_0/snr_-5/snr_-10）时
    ML = (ML + MH) / 2    # 较低的能量阈值：(后5帧能量均值 + 总能量均值)的1/2

    Zs = np.sum(zcr[:5])  / 5  # 过零率阈值：前5帧过零率均值

    A = []
    B = []
    C = []

    # 1. 使用较大能量阈值 MH 进行初步能量检测
    flag = 0
    for i in range(len(energy)):
        if len(A) == 0 and flag == 0 and energy[i] > MH :
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 21 > A[len(A) - 1]:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 21 <= A[len(A) - 1]:
            A = A[:len(A) - 1]
            flag = 1

        if flag == 1 and energy[i] < MH :
            A.append(i)
            flag = 0

    # 2. 使用较小能量阈值 ML 进行第二步能量检测
    for j in range(len(A)) :
        i = A[j]
        if j % 2 == 1 :
            while i < len(energy) and energy[i] > ML :
                i = i + 1
            B.append(i)
        else :
            while i > 0 and energy[i] > ML :
                i = i - 1
            B.append(i)
    
    # 3. 使用过零率阈值 Zs 进行最后一步检测
    for j in range(len(B)) :
        i = B[j]
        if j % 2 == 1 :
            while i < len(zcr) and zcr[i] >= 3 * Zs :
                i = i + 1
            C.append(i)
        else :
            while i > 0 and zcr[i] >= 3 * Zs :
                i = i - 1
            C.append(i)

    return C

# 将时间标签转换为帧标签
def make_label(json_file, y, sr, frame_len=400, frame_step=160):
    temp = np.zeros(len(y))
    with open(json_file) as f:
        segments = json.load(f)
    f.close()

    # 时间标签 -> 采样点标签temp
    for i in range(len(segments['speech_segments'])):
        start = int(segments['speech_segments'][i]['start_time'] * sr)
        end = int(segments['speech_segments'][i]['end_time'] * sr)

        temp[start:end] = 1
    
    # 采样点标签temp分帧
    temp_frames = psf.sigproc.framesig(temp, frame_len, frame_step)  # label (699,1)

    # 帧标签
    label = np.zeros(temp_frames.shape[0])  # (699,)
    for i in range(temp_frames.shape[0]):
        label[i] = np.where(temp_frames[i].sum() > 0, 1, 0)

    return label[:698]

# 比较检测结果与帧标签
def evaluate(wave_detect, label):
    # 将检测结果转换为帧结果(698,)
    pred = [0 for i in range(len(label))]

    for i in range(0,len(wave_detect),2):
        if(wave_detect[i+1]>len(label)):
            wave_detect[i+1]=len(label)
        for j in range(wave_detect[i], wave_detect[i+1]):
            pred[j] = 1
    
    acc_n = 0
    miss_n = 0
    false_n = 0

    for i in range(len(label)):
        if label[i] == pred[i]:
            acc_n += 1
        elif label[i]==1 and pred[i]==0:
            miss_n +=1
        elif label[i]==0 and pred[i]==1:
            false_n +=1 

    return acc_n/len(label), miss_n/len(label), false_n/len(label)

# 绘制结果
def result_plot(y, y_frames, sr, energy, zcr, wave_detect):
    
    # 列表形式的帧检测结果 转为 矩阵形式的frames检测结果
    n = np.zeros(y_frames.shape)
    for i in range(0,len(wave_detect),2):
        n[..., wave_detect[i]:wave_detect[i+1]+1] = 1

    # deframes
    n_deframes = psf.sigproc.deframesig(n.T, 0, 400, 160)

    x = np.linspace(0,7,len(y))
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(x,y)
    plt.plot(x,n_deframes[:len(y)])
    #plt.vlines(wave_detect,-1,1 ,colors = "r")
    plt.title("wave")

    plt.subplot(3, 1, 2)
    plt.plot(energy)
    plt.title("energy")
    plt.ylim(0,1)
    # 绘制短时过零率
    plt.subplot(3, 1, 3)
    plt.plot(zcr)
    plt.title("zero crossing rate")

    plt.tight_layout()
    plt.show()

# 评估不同信噪比的测试集
def main(mode):
    FRAME_LEN = 400
    FRAME_SETP = 160
    Acc = []
    FRR = []    # 检测为非语音帧的语音帧
    FAR = []    # 检测为语音帧的非语音帧

    TEST_SET = os.path.join("test", mode)
    test_files = [filename for filename in os.listdir(TEST_SET) if filename.endswith(".wav")]
    test_labels = [filename for filename in os.listdir(TEST_SET) if filename.endswith(".json")]

    for i in range(len(test_files)):
        y, sr = librosa.load(TEST_SET+"\\"+test_files[i],sr=16000)
        y = y / np.max(y)   # 归一化
        y_frames = psf.sigproc.framesig(y, FRAME_LEN, FRAME_SETP, winfunc=lambda x:np.hamming(x)).T  # (400,699)
        label = make_label(TEST_SET+"\\"+test_labels[i], y, sr)

        energy = calEnergy(y_frames)    # (699,)
        zcr = calZeroCrossingRate(y_frames)    # (699,)
        wave_detect = endPointDetect(y, energy, zcr)    # 端点检测
        if(len(wave_detect) % 2 ==1):
            wave_detect.append(len(label))
        #result_plot(y, y_frames, sr, energy, zcr, wave_detect) # 绘制结果

        acc_n, miss_n, false_n = evaluate(wave_detect, label)   # 评估
        print(test_files[i],test_labels[i])
        print("acc:{:.4f}, miss:{:.4f}, false:{:.4f}" .format(acc_n,miss_n,false_n))
        print("-----------------------------------------")
        Acc.append(acc_n)
        FRR.append(miss_n)
        FAR.append(false_n)

    Acc_average = np.mean(Acc)
    FRR_average = np.mean(FRR)
    FAR_average = np.mean(FAR)
    print("mode:{}\tAcc:{:.4f}, FRR:{:.4f}, FAR:{:.4f}" 
    .format(mode,Acc_average,FRR_average,FAR_average))
    print("==========================================")


if __name__ == '__main__':
    main("pure")
    #main("snr_10")
    #main("snr_5")
    
    # 将 MH = energy_mean / 2 改为 MH = energy_mean
    #main("snr_0")
    #main("snr_-5")
    #main("snr_-10")