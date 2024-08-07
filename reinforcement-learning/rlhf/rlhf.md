---
description: RLHF
---

# RLHF

## 问题背景 <a href="#hp1av" id="hp1av"></a>

强化学习任务本身的复杂性使得复杂问题的奖励函数设计是困难的，这个问题可以通过引入人类反馈进行解决。在此之前的做法是使用比较复杂的人类专家打分机制，而RLHF用人类偏好替代人类专家打分，这样一举两得：

1\. 复杂的奖励函数由神经网络自己拟合，不需要人为设计复杂的奖励函数；

2.人工成本大幅度降低，只需要非专家人类给出偏好即可。

## 方法 <a href="#x2uoq" id="x2uoq"></a>

基本的3个过程：

1. policy与environment交互获得一系列trajectories，基于传统强化学习算法优化策略以最大化收获的奖励；
2. 选择多对trajectories对发送给人类做偏好判断（在一对中选择其中之一）；
3. 基于人类反馈来优化奖励函数。

### 策略优化（过程1） <a href="#k1p5s" id="k1p5s"></a>

三个细节：

1. 应当选择对奖励函数变化鲁棒的强化学习策略（因为reward 模型是不断根据人类反馈更新的）；
2. 选择了A2C与TRPO两个强化学习算法，由于TRPO算法依赖Trust region来确保充分探索，如果奖励函数发生变化可能导致探索不充分，因此人为调整了超参数entropy bouns；
3. 对奖励模型输出的奖励值做normalize，确保均值为0且有恒定的标准差。

### 偏好标记（过程2） <a href="#mip7q" id="mip7q"></a>

三种情况：

1. 更喜欢两个轨迹其中之一，就给出更喜欢的轨迹；
2. 觉得同样好，那两个轨迹每个分配0.5的概率值；
3. 无法比较的轨迹对被排除，不纳入数据集

### 拟合奖励函数（过程3） <a href="#freeq" id="freeq"></a>

基本方案是基于交叉熵损失函数拉近两个分布的差距：

![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/148456571/1722479319426-ab7fb6e2-7812-4196-9c44-f826debd6eb5.png)

1. 人类偏好分布，即![](https://intranetproxy.alipay.com/skylark/lark/\_\_latex/f029d4f2ee692935f497b27ff4fae847.svg)，要么是一个确定性分布（人类喜欢其中之一），要么是一个均匀分布（人类都喜欢）；
2. 奖励函数输出的分布。

![](https://intranetproxy.alipay.com/skylark/lark/0/2024/png/148456571/1722479656462-5effa012-6402-4749-9653-ab2228e146b5.png)

三个细节：

1. 事实上同时训练多个神经网络模型来预测reward函数，所以对于一个observation action pair，多个网络会分别输出奖励值，然后分别normalize之后求平均值；
2. 使用一些正则化策略如l2 normalization及dropout；
3. 假设人类反馈中有10%的概率出错，那实际上奖励函数的分布建模为：

![](https://intranetproxy.alipay.com/skylark/lark/\_\_latex/74244d7508e557793fa66922a1e77bd6.svg)

即90%是对的，10%是随机建模。

### 如何选择需要人类反馈的轨迹对 <a href="#cmdgo" id="cmdgo"></a>

因为有多个reward model，对于一个轨迹对，多个reward model会预测出多个结果（更偏好哪一个），选择那些预测结果方差最大的轨迹对（说明多个reward model对该轨迹对存在分歧），将这些轨迹对送给人类去寻求偏好反馈。（不确定性越大，熵越高，信息量越大，从而让人类只提供信息量最大的反馈，同样可以缩减人力成本）。
