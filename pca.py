import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# # 加载鸢尾花数据集
# iris = load_iris()
# data = iris.data  # 特征数据
# target = iris.target  # 标签数据

data = pd.read_csv('D:\\白叶枯数据\\ZJ_LY_all_data_with_average_severity-带表头.csv', header=None)
data = data.iloc[:, :23].values

# print(type(data))
print(data.shape)
print(data)

# PCA降维
pca = PCA(n_components=5).fit(data)  # 利用PCA算法降成2维
new_data = pca.transform(data)
print(new_data.shape)
print(new_data)
result = pd.DataFrame(data=new_data, columns=[f"X{x}" for x in range(1, 6)])
result.to_csv('C:\\Users\\user\\Desktop\\Pca_result.csv', index=True)
# plt.title('Iris dimensions reduction: 4 to 2')
# plt.scatter(new_data[:, 0], new_data[:, 1], c=target)
# plt.show()
#
# model = KNeighborsClassifier(3)
# score = model.fit(data, target).score(data, target)
# print('4-dims:', score)
# score = model.fit(new_data, target).score(new_data, target)
# print('2-dims:', score)

# 数据从4维降到2维后，可以很方便地进行可视化。
# 从散点图中直观地看，降维后的数据较好地保留了原数据的分布信息.
# 另外可以看到，降维后的KNN分类模型准确性有所提升，这也是数据降维的目的之一。
