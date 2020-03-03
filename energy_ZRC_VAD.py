# -*- coding: utf-8 -*-
import wave
import numpy as np
import matplotlib.pyplot as plt


def sgn(data):
    if data >= 0 :
        return 1
    else :
        return 0


def normalize(x):
    x = np.array(x)
    x_normed = x / x.max()
    
    return list(x_normed)


def read_wav(path):
    f = wave.open(path)

    str_data = f.readframes(f.getnframes()) # readframes() 按照采样点读取数据
    wave_data = np.fromstring(str_data, dtype = np.short)   # 转成二字节数组形式（每个采样点占两个字节）
    f.close()

    return wave_data, f.getframerate()


# 计算每一帧的能量
def calEnergy(wave_data) :
    energy = []
    sum = 0
    for i in range(len(wave_data)) :
        sum = sum + (int(wave_data[i]) * int(wave_data[i])) # 每一帧中采样点对应的幅值的平方和
        if (i + 1) % 256 == 0 : #  帧长：256
            energy.append(sum)
            sum = 0
        elif i == len(wave_data) - 1 :
            energy.append(sum)

    return normalize(energy)


# 计算过零率
def calZeroCrossingRate(wave_data) :
    zeroCrossingRate = []
    sum = 0
    for i in range(len(wave_data)) :
        if i % 256 == 0:
            continue
        sum = sum + np.abs(sgn(wave_data[i]) - sgn(wave_data[i - 1]))
        if (i + 1) % 256 == 0 :
            zeroCrossingRate.append(float(sum) / 255)
            sum = 0
        elif i == len(wave_data) - 1 :
            zeroCrossingRate.append(float(sum) / 255)
    
    return normalize(zeroCrossingRate)

# 利用短时能量，短时过零率，使用双门限法进行端点检测
def endPointDetect(wave_data, energy, zeroCrossingRate) :
    sum = 0
    energyAverage = 0
    for en in energy :
        sum = sum + en
    energyAverage = sum / len(energy)   # 计算能量均值

    sum = 0
    for en in energy[:5] :
        sum = sum + en
    ML = sum / 5                        
    MH = energyAverage / 4  # 较高的能量阈值：总能量均值的1/4
    ML = (ML + MH) / 4    # 较低的能量阈值：(前五帧能量均值 + 总能量均值)的1/4
    sum = 0
    for zcr in zeroCrossingRate[:5] :
        sum = float(sum) + zcr             
    Zs = sum / 5    # 过零率阈值：前五帧过零率均值

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
            while i < len(zeroCrossingRate) and zeroCrossingRate[i] >= 3 * Zs :
                i = i + 1
            C.append(i)
        else :
            while i > 0 and zeroCrossingRate[i] >= 3 * Zs :
                i = i - 1
            C.append(i)
    wave_detect = [i*256 for i in C]    # 将检测点由帧转换为对应的采样点
    
    return wave_detect


def result_plot(wave_data, sr, energy, zeroCrossingRate, wave_detect):
    plt.figure()
    
    # 绘制原始波形
    x1 = [i/sr for i in range(len(wave_data))]   # x轴尺度变换，将采样点转换为时间
    wave_detect = [i/sr for i in wave_detect]    # 检测点尺度变换
    
    plt.subplot(3, 1, 1)
    plt.plot(x1,normalize(wave_data))
    plt.vlines(wave_detect,-1,1 ,colors = "r")
    plt.title("wave")
    # 绘制短时能量
    x2 = [i*256/sr for i in range(len(energy))]  # x轴尺度变换，将帧转换为时间
    
    plt.subplot(3, 1, 2)
    plt.plot(x2,energy)
    plt.title("energy")
    plt.ylim(0,1)
    # 绘制短时过零率
    plt.subplot(3, 1, 3)
    plt.plot(x2,zeroCrossingRate)
    plt.title("zero crossing rate")
    plt.ylim(0,1)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    wave_data, sr = read_wav("test.wav")
    energy = calEnergy(wave_data)
    zeroCrossingRate = calZeroCrossingRate(wave_data)
    wave_detect = endPointDetect(wave_data, energy, zeroCrossingRate)
    result_plot(wave_data, sr, energy, zeroCrossingRate, wave_detect)
