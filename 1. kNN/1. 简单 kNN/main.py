# coding=utf-8
import numpy as np
import operator


def create_data_set():
    """
    创建一个简单的样本集
    :return: 样本集
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    """
    根据给定的数据集将输入的 in_x 分类
    :param in_x: 将要分类的数据
    :param data_set: 样本集
    :param labels: 样本集的 label
    :param k: kNN 作为分类基准的点的个数
    :return: in_x 的分类结果的 label
    """

    # numpy.ndarray.shape docs: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
    # 获取 array 的形状（在各个维度的长度）
    # 此处是获得输入样本集的点的个数
    data_set_size = data_set.shape[0]

    # numpy.tile docs: https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html
    # 将 array 在对应的维度上复制指定次数
    # 将 in_x 扩展到与样本集一样大，同时计算与样本集中的每一个点的距离
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set  # 每项求差值
    # 每项平方
    sq_diff_mat = diff_mat ** 2

    # numpy.ndarray.sum docs: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.sum.html
    # 返回指定维度加和后的 array
    sq_distances = sq_diff_mat.sum(axis=1)
    # 依次开方，值为 in_x 与样本集中每一项的欧氏距离
    distances = sq_distances ** 0.5

    # numpy.argsort docs: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    # array 排序
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    # 选取欧氏距离最近的 k 个点
    for i in range(k):
        # 获取每个点的 label
        vote_i_label = labels[sorted_dist_indicies[i]]
        # 计算每种 label 出现的次数
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    # 对结果的字典进行排序，返回的结果是元组的元组: ((key, value), (key, value), ...)
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回在距离前 k 近的样本中出现最多的 label
    return sorted_class_count[0][0]


if __name__ == '__main__':
    _group, _labels = create_data_set()
    _in_x = [0, 0]
    print(classify0(_in_x, _group, _labels, 3))
