from numpy import *
from numpy import linalg as la
import pandas as pd




filename = 'D:/movies_data/ml-latest-small/movies/ratings.csv'


def loadEXData(dataPath):
    """
    获得评分矩阵
    :param dataPath: 文件路径
    :return: user-item评分矩阵
    """
    # 设置我们需要加载的字段
    dtype = {'userId': int32, 'movieId': int32, 'rating': float32}
    # 加载数据只选用前3列
    ratings = pd.read_csv(dataPath, dtype=dtype, usecols=['userId', 'movieId', 'rating'])
    # 通过透视表将电影id转换成列名称，生成user-item评分矩阵
    user_item_matrix = ratings.pivot_table(index=['userId'], columns=['movieId'], values=['rating'])
    user_item_matrix = user_item_matrix.fillna(0)
    return user_item_matrix


def ecludSim(inA, inB):
    """
    基于欧式距离计算相似度；
    :param inA: 列向量
    :param inB: 列向量
    :return:
    """
    return 1.0 / (1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    """
    计算皮尔逊相似度；
    :param inA: 列向量
    :param inB: 列向量
    :return:
    """
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    """
    计算余弦相似度；
    :param inA: 列向量
    :param inB: 列向量
    :return:
    """
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeans, item):
    """
    基于物品相似度的推荐引擎:计算某用户未评分物品中，以对该物品和其他物品评分的用户的物品相似度，然后进行综合评分；
    :param dataMat: 训练数据集
    :param user: 用户编号
    :param simMeans: 相似度计算方法
    :param item: 未评分的物品编号
    :return: ratSimTotal/simTotal 评分（0-5）
    """
    # 数据集中的物品数目
    n = shape(dataMat)[1]
    # 初始化两个评分值
    simTotal = 0.0
    ratSimTotal = 0.0
    # 遍历行中的每个物品（对用户评过分的物品进行遍历，并将它与其他物品进行比较）
    for j in range(n):
        # user对j物品的评分
        userRating = dataMat[user, j]
        # 如果某个物品的评分为0，表示该物品未被评分跳过
        if userRating == 0:
            continue
        # 找出要预测评分的物品列和当前取的物品j列里评分都不为0的下标（也就是所有评过这两个物品的用户对这两个物品的评分）
        # nonzero函数返回不为0元素的坐标
        # [].A：将矩阵化为Array数组
        # nonzero函数返回的坐标取[0]是用户编号
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        # A向量：预测评分列item，B向量：当前所取的j列，两组向量要么A有评分B没评分，要么A没评分B有评分
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeans(dataMat[overLap, item], dataMat[overLap, j])
        # print('the %d and %d similarity is %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def svdEst(dataMat, user, simMeans, item):
    """
    基于SVD的评分估计
    :param dataMat: 训练数据集
    :param user: 用户编号
    :param simMeans: 相似度计算方法
    :param item: 未评分的物品编号
    :param k: 奇异值个数
    :return: ratSimTotal/simTotal 评分（0-5）
    """
    n = shape(dataMat)[1]
    simTotal = 0
    ratSimTotal = 0
    # 对原数据矩阵进行SVD操作
    U, Sigma, VT = la.svd(dataMat)
    # 判断选取多少个奇异值合适
    k = analyse_data(Sigma, 0.9)
    # 取sigma矩阵中前k个奇异值构建成一个对角矩阵
    Sigk = mat(eye(k) * Sigma[:k])
    # 利用U矩阵将物品转换到低维空间，构建转换后的物品（物品+k个主要特征）
    # 计算的是V(nxk)
    xfromedItems = dataMat.T * U[:, :k] * Sigk.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeans(xfromedItems[item, :].T, xfromedItems[j, :].T)
        # print('the %d and %d similarity is %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N, simMeans, est_method):
    """
    调用推荐引擎，对指定用户，产生前N评分预测
    :param dataMat: 训练数据集
    :param user: 用户编号
    :param N:
    :param simMeans: 相似度计算方法
    :param est_method: 使用推荐算法
    :return: 返回最终前N个推荐结果
    """
    # 获取指定用户编号下未评分的物品集
    un_rate_items = nonzero(dataMat[user, :].A == 0)[1]
    if len(un_rate_items) == 0:
        return 'you rated everything'
    # 记录该用户未评分物品的评分
    item_scores = []
    for item in un_rate_items:
        estimate_score = est_method(dataMat, user, simMeans, item)
        item_scores.append((item, estimate_score))
    # 按每个元素下标为1的参数从大小排列，取前N个值
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[: N]


def analyse_data(S, a):
    """
    根据奇异值平方和的百分比确定维度降到多少合适
    :param S: 奇异值矩阵
    :param a: 奇异值平方和百分比
    :return:
    """
    S *= S
    threshold = sum(S) * a
    k = 0
    for i in range(S.shape[0]+1):
        if sum(S[:i]) >= threshold:
            k = i
            break
    return k


if __name__ == '__main__':
    myMat = mat(loadEXData(filename))
    print(myMat)
    user = 1
    N = 3
    topn1 = recommend(myMat, user, N, pearsSim, standEst)
    topn2 = recommend(myMat, user, N, pearsSim, svdEst)
    print('采用SVD推荐引擎对用户ID：%d,推荐前%d部电影是：' % (user, N))
    for i in topn2:
        print(i)





