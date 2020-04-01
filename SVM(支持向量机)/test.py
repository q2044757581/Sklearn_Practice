from sklearn import svm
import numpy as np
import time
x = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([0, 0, 1, 1, 2])
'''
‘linear’:线性核函数
‘poly’：多项式核函数
‘rbf’：径像核函数/高斯核
‘sigmod’:sigmod核函数
‘precomputed’:核矩阵
'''
# 用one-against-one方式把二分类器变成多分类器 这种方法虽然好,但是当类别很多的时候,model的个数是n*(n-1)/2,代价还是相当大的
model = svm.SVC(gamma='scale', decision_function_shape='ovo')
# LinearSVC 实现 “one-vs-the-rest” 多类别策略 这种方法有种缺陷,因为训练集是1:M,这种情况下存在biased.因而不是很实用
# model = svm.LinearSVC()  # 实现线性核函数的支持向量分类
# model = svm.NuSVC(kernel='linear')  # 和svc相似
model.fit(x, y)
print("类数", model.decision_function([[0, 1]]).shape[1])  # 3个类
# print(model.coef_.shape)
print(model.dual_coef_.shape)
time1 = time.time()
print(model.predict(x))
print(model.support_vectors_)  # 支持向量
print(model.support_)  # 支持向量的下标
print(model.n_support_)  # 每个类别的支持向量的数目
print(time.time() - time1)