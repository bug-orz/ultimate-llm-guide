# Layer Normalization

Layer Normalization（层归一化）是一种正则化技术，它在神经网络处理中用于改善训练的稳定性和加速收敛。它应用于每一层的输入，在每一层上进行归一化。此外，它与批归一化（Batch Normalization）不同，不依赖于batch的大小。因此，它在小batch甚至是单个样本时也能很好地工作。

Layer Normalization 的公式如下：

假设输入为 $$\mathbf{x} = [x_1, x_2, \ldots, x_H]$$ ，其中 (H) 是隐藏单元的数量，那么：

1. 首先计算输入的均值和方差： $$\mu = \frac{1}{H} \sum_{i=1}^H x_i ] [ \sigma^2 = \frac{1}{H} \sum_{i=1}^H (x_i - \mu)^2$$
2. 然后使用计算出来的均值和方差对输入进行归一化： $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$ 这里的 ( $$\epsilon$$ ) 是一个非常小的常数，用于防止除零错误。
3. 最后，进行线性变换（可选的尺度和偏移参数）： \[ $$y_i = \gamma \hat{x}_i + \beta$$] 其中 ( $$\gamma$$ ) 和 ( $$\beta$$ ) 是可学习参数，分别是尺度和偏移。

总结一下，Layer Normalization 的步骤可以归纳为： $$\begin{equation} y_i = \gamma \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \end{equation}$$

这种归一化方法会独立地对每个样本进行操作，与其他样本无关，因此在处理序列或单个数据点时具有优势。
