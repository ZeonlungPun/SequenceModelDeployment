from read_data import get_train_data,get_test_data
from sklearn.metrics import f1_score
from scipy.stats import skew, kurtosis, iqr
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
from scipy.fftpack import fft
import m2cgen as m2c

def extract_features(X):
    n_samples = X.shape[0]
    all_features = []

    for i in range(n_samples):
        sample = X[i]  # (128, 9)
        sample_features = []

        for j in range(9):
            channel = sample[:, j]

            # --- A. 時域特徵  ---
            sample_features.extend([
                np.mean(channel), np.std(channel),
                np.max(channel), np.min(channel),
                np.mean(channel ** 2)  # Energy
            ])

            # --- B. 頻域特徵 (FFT) ---
            # 1. 執行 FFT 並取振幅 (Magnitude)
            # 使用 rfft 處理實數信號更高效
            fft_values = np.abs(np.fft.rfft(channel))

            # 2. 提取頻域統計量
            sample_features.append(np.mean(fft_values))  # 頻譜平均能量
            sample_features.append(np.std(fft_values))  # 頻譜波動
            sample_features.append(np.argmax(fft_values))  # 主頻位置 (哪種頻率最強)

            # 3. 頻譜能量 (Spectral Energy)
            spectral_energy = np.sum(fft_values ** 2) / len(fft_values)
            sample_features.append(spectral_energy)

            # 4. 頻譜熵 (Spectral Entropy) - 衡量信號複雜度
            psd = (fft_values ** 2) / np.sum(fft_values ** 2 + 1e-6)
            entropy = -np.sum(psd * np.log(psd + 1e-6))
            sample_features.append(entropy)

        all_features.append(sample_features)

    return np.array(all_features)
data_path = './HAR'
train_path = 'train'
test_path = 'test'

xtrain,ytrain = get_train_data(data_path,train_path)
xtest,ytest =get_test_data(data_path,test_path)
X_train_features = extract_features(xtrain)
X_test_features = extract_features(xtest)

# X_train_features = pd.read_csv('HAR/train/X_train.txt', delim_whitespace=True, header=None)
# ytrain = pd.read_csv('HAR/train/y_train.txt', delim_whitespace=True, header=None)
#
# X_test_features = pd.read_csv('HAR/test/X_test.txt', delim_whitespace=True, header=None)
# ytest = pd.read_csv('HAR/test/y_test.txt', delim_whitespace=True, header=None)
#
# ytrain = ytrain.values.ravel()
# ytest = ytest.values.ravel()

predictor = ExtraTreesClassifier(n_estimators=50, random_state=42)
predictor.fit(X_train_features, ytrain)
y_pred = predictor.predict(X_test_features)
macro_f1 = f1_score(ytest, y_pred, average='macro')
print("macro f1:",macro_f1)
code = m2c.export_to_c(predictor)
code = code.replace(
    "void score(double * input, double * output)",
    "void classify_action(double * input, double * output)"
)

with open("HARmodel.c", "w") as f:
    f.write(code)