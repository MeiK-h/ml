# coding=utf-8
import math
import operator

import TreePlotter


def calc_shannon_ent(data_set):
    """
    计算指定数据集的香农熵（信息熵： https://zh.wikipedia.org/zh-hans/熵 (信息论)）
    :param data_set: 输入数据集
    :return: 数据集的香农熵
    """
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        label_counts[current_label] = label_counts.get(current_label, 0) + 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = label_counts[key] / num_entries
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent


def create_data_set():
    """
    创建一个测试数据集
    :return: 测试数据集
    """
    _data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    _labels = ['no surfacing', 'flippers']
    return _data_set, _labels


def split_data_set(data_set, axis, value):
    """
    按指定规则划分数据集，返回指定 axis 轴上值为 value 的数据
    :param data_set: 要划分的数据集
    :param axis: 指定的轴（维度）
    :param value: 指定的数据
    :return: 划分后的数据集
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[: axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    选择最好的数据集划分方式
    :param data_set: 数据集
    :return: 划分的方式
    """
    # 可供划分的种类（数据维度）
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        # set 只留下不重复的数据
        unique_vals = set(feat_list)
        new_entropy = 0
        # 依次以每种数据来划分
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / len(data_set)
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        # 求出最好的信息增益
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    # 返回最好的信息增益对应的划分方式
    return best_feature


def majority_cnt(class_list):
    """
    返回出现次数最多的 class
    :param class_list: 所有数据的 class
    :return: 出现次数最多的 class
    """
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    根据给定的数据集和 labels 生成决策树
    :param data_set: 数据集
    :param labels: 数据集对应标签
    :return: 最终生成的决策树
    """
    # 取出所有数据的 class
    class_list = [example[-1] for example in data_set]
    # 当所有数据的 class 都相同时停止划分，返回这些数据的 class
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果递归遍历完所有数据，则返回出现次数最多的 class
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    # 选择最优的划分，及其 label
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]
    # 根据最优的 label 生成 tree
    my_tree = {
        best_feat_label: {}
    }
    # 删除已经在 tree 上出现的 label
    del (labels[best_feat])
    # 得到列表包含的所有属性值
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        # 递归生成决策树
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    """
    根据已生成的决策树对指定的数据进行分类
    :param input_tree: 决策树
    :param feat_labels: 决策树的 labels
    :param test_vec: 测试数据
    :return: 测试数据的分类
    """
    first_str = next(iter(input_tree))
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            return classify(second_dict[key], feat_labels, test_vec) if isinstance(second_dict[key], dict) else \
                second_dict[key]


def store_tree(input_tree, filename):
    """
    将决策树存入文件
    :param input_tree: 要存储的决策树
    :param filename: 存储的文件
    :return:
    """
    import pickle
    with open(filename, 'w') as fw:
        pickle.dumps(input_tree, fw)


def grab_tree(filename):
    """
    从文件中读取决策树
    :param filename: 要读取的文件
    :return: 决策树
    """
    import pickle
    with open(filename) as fr:
        return pickle.load(fr)


if __name__ == '__main__':
    with open('lenses.txt') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 标签列表
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 创建决策树
    lensesTree = create_tree(lenses, lensesLabels)
    # 打印树
    print(lensesTree)
    # 显示树形图
    TreePlotter.create_plot(lensesTree)
