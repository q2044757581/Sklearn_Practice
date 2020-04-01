from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
# 线性回归
reg = linear_model.LinearRegression(fit_intercept=True)
# 稳健回归
# reg = linear_model.RANSACRegressor()
x = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
y = [1, 2, 3, 10, 5]
print(x)
print(y)
reg.fit(x, y)
# print(reg.estimator_.coef_)
# print(reg.estimator_.intercept_)
print(reg.coef_)
print(reg.intercept_)
result = reg.predict(x)
print(result)
