# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pyAudioAnalysis import ShortTermFeatures as sF
from pyAudioAnalysis import audioTrainTest as aT


def read_wave(path):  
    audiofile = AudioSegment.from_file(path)
    signal = np.frombuffer(audiofile._data, np.int16)
    sr = audiofile.frame_rate

    return signal, sr


def smoothMovingAvg(signal, windowLen=11):
    windowLen = int(windowLen)
    if signal.ndim != 1:
        raise ValueError("")
    if signal.size < windowLen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if windowLen < 3:
        return signal
    s = np.r_[2*signal[0] - signal[windowLen-1::-1],
                 signal, 2*signal[-1] - signal[-1:-windowLen:-1]]
    w = np.ones(windowLen, 'd')
    y = np.convolve(w/w.sum(), s, mode='same')
    
    return y[windowLen:-windowLen+1]


def extract_features(y, sr, st_win, st_step):
    st_feats, _ = sF.feature_extraction(y, sr, st_win * sr, st_step * sr)    # st_feats：(68,966)——(特征个数,帧数)
    
    return st_feats


def train_SVM(st_feats):
    st_energy = st_feats[1, :]
    en = np.sort(st_energy)
    l1 = int(len(en) / 10)
    t1 = np.mean(en[0:l1]) + 0.000000000000001    # 计算10%较低能量的均值，作为低阈值
    t2 = np.mean(en[-l1:-1]) + 0.000000000000001    # 计算10%较高能量的均值，作为高阈值
    class1 = st_feats[:, np.where(st_energy <= t1)[0]]    # 将能量低于低阈值的帧，作为class1
    class2 = st_feats[:, np.where(st_energy >= t2)[0]]    # 将能量高于高阈值的帧，作为class2
    feats_s = [class1.T, class2.T]    # class1.T:(58,68)|class2.T:(38,68)

    [feats_s_norm, means_s, stds_s] = aT.normalizeFeatures(feats_s)    # 标准化：减均值除方差
    svm = aT.trainSVM(feats_s_norm, 1.0)

    return svm, means_s, stds_s


def VAD(st_feats, means_s, stds_s, st_step, smoothWindow=0.5, weight=0.5):
    # silenceRemoval(y, sr, st_win, st_step, smoothWindow=0.5, weight=0.5, plot=False)
    # 1. 使用SVM检测
    prob_on_set = []
    for i in range(st_feats.shape[1]):
        cur_fv = (st_feats[:, i] - means_s) / stds_s    # 对每一帧特征标准化
        prob_on_set.append(svm.predict_proba(cur_fv.reshape(1,-1))[0][1])    # 保存SVM对每帧的检测概率
    prob_on_set = np.array(prob_on_set)
    prob_on_set = smoothMovingAvg(prob_on_set, smoothWindow / st_step)    # 平滑输出概率曲线
    
    # 2. 检测起始帧索引
    # 将SVM输出概率排序，取前10%的较大值和后10%的较小值，加权平均得到阈值
    prog_on_set_sort = np.sort(prob_on_set) 
    Nt = int(prog_on_set_sort.shape[0] / 10)   
    T = (np.mean((1 - weight) * prog_on_set_sort[0:Nt]) + 
        weight * np.mean(prog_on_set_sort[-Nt::]))

    max_idx = np.where(prob_on_set > T)[0]    # 得到大于阈值的帧
    
    # 3. 转换检测结果为秒
    i = 0
    time_clusters = []
    seg_limits = []

    while i < len(max_idx):
        cur_cluster = [max_idx[i]]
        if i == len(max_idx)-1:
            break
        while max_idx[i+1] - cur_cluster[-1] <= 2:
            cur_cluster.append(max_idx[i+1])
            i += 1
            if i == len(max_idx)-1:
                break
        i += 1
        time_clusters.append(cur_cluster)
        seg_limits.append([cur_cluster[0] * st_step,
                           cur_cluster[-1] * st_step])

    # 4. 删除较小语音段
    min_dur = 0.2    # 检测到的起始点与结束点之间大于0.2s
    seg_limits_2 = []
    for s in seg_limits:
        if s[1] - s[0] > min_dur:
            seg_limits_2.append(s)
    seg_limits = seg_limits_2

    return seg_limits, prob_on_set


def plot_result_(y, sr, seg_limits, prob_on_set, st_step):
    plt.subplot(2, 1, 1)    # 绘制原始波形和检测结果
    x = np.arange(0, y.shape[0] / float(sr), 1.0 / sr)
    plt.plot(x, y/y.max())
    for s in seg_limits:
        plt.axvline(x=s[0], color='red')
        plt.axvline(x=s[1], color='red')
    plt.title('wave')

    plt.subplot(2, 1, 2)    # 绘制SVM输出概率曲线和检测结果
    plt.plot(np.arange(0, prob_on_set.shape[0] * st_step, st_step), 
                prob_on_set)
    for s in seg_limits:
        plt.axvline(x=s[0], color='red')
        plt.axvline(x=s[1], color='red')
    plt.title('SVM probability')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    st_win=0.02
    st_step=0.01
    y, sr = read_wave("test.wav")   

    # 1. 特征提取
    st_feats = extract_features(y, sr, st_win, st_step)
    # 2. 训练SVM
    svm , means_s, stds_s = train_SVM(st_feats)
    # 3. VAD检测
    seg_limits, prob_on_set = VAD(st_feats, means_s, stds_s, st_step, smoothWindow=0.5, weight = 0.5)
    # 4. 绘制结果
    plot_result_(y, sr, seg_limits, prob_on_set, st_step)

    # 或者直接调用
    '''
    from pyAudioAnalysis.audioSegmentation import silenceRemoval as sR
    seg_lims = sR(y, sr, st_win, st_step, 0.5, 0.4, True)
    '''
    