import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


def ML_perf(yc, y_r, yp, y_rp, str_n):
    plt.figure()
    RMSEC = np.sqrt(np.mean((yc-y_r)**2))
    R2C = 1-(np.sum((yc-y_r)**2))/(np.sum((yc-np.mean(yc))**2))
    RPDC = np.std(yc)/RMSEC

    plt.scatter(yc, y_r, c='g', zorder=1, edgecolors=(0, 0, 0))
    plt.plot([0, 5], [0, 5], 'r')

    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.title(str_n + ' ($\mathregular{R_c^2}$=%.2f, RMSEC=%.2f, RPDC=%.2f)' % (R2C, RMSEC, RPDC))
#     plt.title(str_n + ' ($\mathregular{R_c^2}$=%.2f, RMSEC=%.2f, RPDC=%.2f)' % (R2C, RMSEC, RPDC))
#     plt.savefig("result/SVR-CAL-5-1.png",dpi=600)#保存图片
    plt.show()

    plt.figure()
    RMSEP = np.sqrt(np.mean((yp-y_rp)**2))
    R2P = 1-(np.sum((yp-y_rp)**2))/(np.sum((yp-np.mean(yp))**2))
    RPDP = np.std(yp)/RMSEP

    plt.scatter(yp, y_rp, zorder=1, edgecolors=(0, 0, 0))
    plt.plot([0, 5], [0, 5], 'r')

    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.title(str_n + '($\mathregular{R_p^2}$=%.2f, RMSEP=%.2f, RPDP=%.2f)' % (R2P, RMSEP, RPDP))
#     plt.title(str_n + ' ($\mathregular{R_p^2}$=%.2f, RMSEP=%.2f, RPDP=%.2f)' % (R2P, RMSEP, RPDP))
#     plt.savefig("result/SVR-PRE-5-1.png",dpi=600)#保存图片
    plt.show()


def x_y_split(df_raw):
    df_x = df_raw.iloc[:, :5]  # iloc 根据行号来索引  loc 根据行index来索引
    df_y = df_raw.iloc[:, -1]
    return df_x, df_y


df_raw = pd.read_csv('C:\\Users\\user\\Desktop\\Pca_result.csv', header=None)
test_data = pd.read_csv('C:\\Users\\user\\Desktop\\2022.csv', header=None)
df_train, df_test = train_test_split(test_data, test_size=0.2, random_state=42)

df_train = pd.concat([df_train, df_raw], axis=0)

x_train, y_train = x_y_split(df_train)
x_test, y_test = x_y_split(df_test)

print(x_train)
print(x_test)


def SVM(X_train, y_train, X_test, y_test):
    # 定义SVM回归器
    svr = SVR()

    # 算不动
    # c_range = np.logspace(-2, 10, 13, base=10)
    # gamma_range = np.logspace(-9, 3, 13, base=10)

    # 定义超参数搜索空间
    param_grid = {
        # 'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range
        'kernel': ['rbf'],
        'C': [k for k in range(1000, 5000, 500)],
        'gamma': [0.01, 0.1, 1, 10]

    }

    # 使用网格搜索和交叉验证进行超参数优化
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # 获取最佳的SVM回归器
    optim_estimator = grid_search.best_estimator_

    # 在训练数据上拟合最佳回归器
    optim_estimator.fit(X_train, y_train)

    # 对训练数据进行交叉验证预测
    y_train_pred = optim_estimator.predict(X_train)

    # 对测试数据进行预测
    y_test_pred = optim_estimator.predict(X_test)

    # 计算评价指标并画图
    y_train_pred = np.array(y_train_pred).reshape(y_train.shape)
    y_test_pred = np.array(y_test_pred).reshape(y_test.shape)
    ML_perf(y_train, y_train_pred, y_test, y_test_pred, 'SVR')

    # 计算RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # 计算R2
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return grid_search.best_params_, rmse_train, rmse_test, r2_train, r2_test


best_params_, rmse_train,  rmse_test, r2_train, r2_test = SVM(x_train, y_train, x_test, y_test)
print(best_params_, rmse_train,  rmse_test, r2_train, r2_test)
