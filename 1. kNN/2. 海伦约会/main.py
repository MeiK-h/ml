# coding=utf-8
import numpy as np
import operator


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


def file2matrix(filename):
    """
    读取海伦的约会数据并解析为 numpy 矩阵
    数据的格式为：
    每年获得的飞行常客里程数\t玩视频游戏所耗时间百分比\t每周消费的冰淇淋公升数\t满意指数\n
    :param filename: 存储海伦约会数据的文件
    :return: 解析出来的矩阵
    """
    with open(filename) as fr:
        array_o_lines = fr.readlines()
    number_of_lines = len(array_o_lines)

    # numpy.zeros docs: https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
    # 返回给定形状和类型的，由 0 填充的新矩阵
    return_mat = np.zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_o_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0: 3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    """
    将同一列的数据以同比例映射到 [0, 1] 之间
    :param data_set: 要映射的数据
    :return: 映射完成的数据，缩放的比例，数据的最小值
    """
    _min_vals = data_set.min(0)
    _max_vals = data_set.max(0)
    _ranges = _max_vals - _min_vals
    _m = data_set.shape[0]
    _norm_data_set = data_set - np.tile(_min_vals, (_m, 1))
    _norm_data_set = _norm_data_set / np.tile(_ranges, (_m, 1))
    return _norm_data_set, _ranges, _min_vals


def dating_class_test():
    """
    测试给定的数据
    :return:
    """
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        print('the classifier came back with: {0}, the real answer is: {1}'.format(classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1
    print('the total error rate is: {0}'.format(error_count / num_test_vecs))


def classify_person():
    """
    判断输入的数据会属于哪个分类
    :return:
    """
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input('每年获得的飞行常客里程数：'))
    ff_miles = float(input('玩视频游戏所耗时间百分比：'))
    ice_cream = float(input('每周消费的冰淇淋公升数：'))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    print('You will probably like this person:', result_list[classifier_result - 1])


if __name__ == '__main__':
    dating_class_test()
    classify_person()
