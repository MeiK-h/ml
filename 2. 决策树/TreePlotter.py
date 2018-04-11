# coding=utf-8
"""
Author: @jiang16
GitHub: https://github.com/jiang16
Page: https://github.com/jiang16/Machine-Learning/blob/master/DecisionTree/DecisionTree.py

Refactored as python3
"""
import matplotlib.pyplot as plt

# 设定文本框和箭头格式
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


# 使用文本注解绘制节点
def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction',
                             va='center', ha='center', bbox=node_type, arrowprops=arrow_args)


# 标注有向边的属性值
def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)


# 获取叶节点的数目
def get_num_leafs(my_tree):
    # 初始化叶子数目
    num_leafs = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


# 获取树的层次
def get_tree_depth(my_tree):
    # 初始化树的高度
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        # 更新层次
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


# 绘制决策树
def plot_tree(my_tree, parent_pt, node_txt):
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    # 中心位置
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    # 绘制节点
    plot_node(first_str, cntr_pt, parent_pt, decisionNode)
    # 标注有向边属性值
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    # 减少y偏移
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in second_dict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if isinstance(second_dict[key], dict):
            # 不是叶结点，递归调用继续绘制
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:  # 如果是叶结点，绘制叶结点，并标注有向边属性值
            # 增加x偏移
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            # 绘制叶子节点
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntr_pt, leafNode)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


# 创建绘制面板
def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_num_leafs(in_tree))
    plot_tree.totalD = float(get_tree_depth(in_tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()
