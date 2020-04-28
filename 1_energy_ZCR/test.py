import librosa
import numpy as np
import python_speech_features as psf
from energy_ZRC_VAD import *

FRAME_LEN = 400
FRAME_SETP = 160

# 1. 加载测试样本
y,sr = librosa.load("test.wav",sr=16000)
y = y / np.max(y)
y_frames = psf.sigproc.framesig(y, FRAME_LEN, FRAME_SETP, winfunc=lambda x:np.hamming(x)).T  # (400,699)

# 2. 加载标签
label = make_label("test.json", y, sr)

# 3. 计算特征值
energy = calEnergy(y_frames)    # (699,)
zcr = calZeroCrossingRate(y_frames)    # (699,)

# 4. 端点检测
wave_detect = endPointDetect(y, energy, zcr)

# 5. 评估
if(len(wave_detect) % 2 ==1):
    wave_detect.append(len(label))
acc, miss, false = evaluate(wave_detect, label)
print("acc:{:.4f}, miss:{:.4f}, false:{:.4f}" .format(acc,miss,false))

# 6. 绘制结果
result_plot(y, y_frames, sr, energy, zcr, wave_detect)
