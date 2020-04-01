import numpy as np
from math import pi
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

# 拟合 KernelRidge 比 SVR快; 然而，对于更大的训练集 SVR通常更好。
# 关于预测时间，由于学习的稀疏解， SVR 对于所有不同大小的训练集都比 KernelRidge 快
plt_x = np.linspace(0, 2*pi, 100)
x = np.array([[i] for i in plt_x])
y = []
cnt = 0
for i in range(0, len(x)):
    if cnt % 5 == 0:
        y.append(np.sin(x[i]) + np.random.random() / 5)
    else:
        y.append(np.sin(x[i]))
    cnt += 1
y = np.array(y)
plt.plot(plt_x, y)
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
svr.fit(x, y)
kr.fit(x, y)
y1 = svr.predict(x)
y2 = kr.predict(x)
plt.plot(plt_x, y1)
plt.plot(plt_x, y2)
plt.legend(['real', 'svr', 'kr'])
plt.show()
