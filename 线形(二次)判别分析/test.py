from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import datasets  # 引入数据集,sklearn包含众多数据集
from sklearn.model_selection import train_test_split  # 将数据分为测试集和训练集
from math import sqrt


def RMSE(target, y_p):
    """
    :param target:
    :param y_p:
    :return: 均方根误差
    """
    result = 0
    for i in range(len(target)):
        result += (target[i] - y_p[i]) ** 2
    return sqrt(result)


def error(target, y_p):
    """
    :param target:
    :param y_p:
    :return: 均方根误差
    """
    result = 0
    for i in range(len(target)):
        if target[i] != y_p[i]:
            result += 1
    return result, len(target)


# 读取数据
data = datasets.load_iris()
x = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#  利用train_test_split进行将训练集和测试集进行分开，test_size占30%
'''
收缩是一种在训练样本数量相比特征而言很小的情况下可以提升的协方差矩阵预测（准确性）的工具。 在这个情况下，
经验样本协方差是一个很差的预测器。收缩 LDA 可以通过设置 
discriminant_analysis.LinearDiscriminantAnalysis 类的 shrinkage 参数为 ‘auto’ 来实现。
shrinkage parameter （收缩参数）的值同样也可以手动被设置为 0-1 之间。
特别地，0 值对应着没有收缩（这意味着经验协方差矩阵将会被使用）， 
而1值则对应着完全使用收缩（意味着方差的对角矩阵将被当作协方差矩阵的估计）。
设置该参数在两个极端值之间会估计一个（特定的）协方差矩阵的收缩形式
solver：str，求解算法，
取值可以为：

    svd：使用奇异值分解求解，不用计算协方差矩阵，适用于特征数量很大的情形，无法使用参数收缩（shrinkage）
    lsqr：最小平方QR分解，可以结合shrinkage使用
    eigen：特征值分解，可以结合shrinkage使用

shrinkage：str or float，是否使用参数收缩
取值可以为：

    None：不适用参数收缩
    auto：str，使用Ledoit-Wolf lemma
    浮点数：自定义收缩比例

priors：array，用于LDA中贝叶斯规则的先验概率，当为None时，每个类priors为该类样本占总样本的比例；
当为自定义值时，如果概率之和不为1，会按照自定义值进行归一化
components：int，需要保留的特征个数，小于等于n-1
store_covariance：是否计算每个类的协方差矩阵，0.19版本删除

1、默认的 solver 是 ‘svd’。它可以进行classification (分类) 以及 transform (转换),而且它不会依赖于协方差矩阵的计算（结果）。
这在特征数量特别大的时候十分具有优势。然而，’svd’ solver 无法与 shrinkage （收缩）同时使用。

2、lsqr solver 则是一个高效的算法，它仅用于分类使用。它支持 shrinkage （收缩）。

3、eigen（特征） solver 是基于 class scatter （类散度）与 class scatter ratio （类内离散率）之间的优化。 它可以被用于 
classification （分类）以及 transform （转换），此外它还同时支持收缩。然而，该解决方案需要计算协方差矩阵，
因此它可能不适用于具有大量特征的情况。
'''
model = LinearDiscriminantAnalysis(n_components=1)
# model = LinearDiscriminantAnalysis(n_components=1, shrinkage='auto', solver='eigen')
# model = QuadraticDiscriminantAnalysis()
model.fit(X_train, y_train)
y_p = model.predict(X_test)
print('RMSE: ', RMSE(y_test, y_p))
num1, num2 = error(y_test, y_p)
print('ERROR: ', num1, '/', num2)

# 利用LDA降维
# print(model.transform(X_train))
# nx, ny = 200, 100
# x_min, x_max = plt.xlim()
# y_min, y_max = plt.ylim()
# xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
#                      np.linspace(y_min, y_max, ny))
# Z = reg.predict_proba(np.c_[xx.ravel(), yy.ravel()])
# Z = Z[:, 1].reshape(xx.shape)
# plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
#                norm=colors.Normalize(0., 1.), zorder=0)
# plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')
# plt.show()
