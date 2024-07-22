# Layer Normalization

Layer Normalization（层归一化）是一种正则化技术，它在神经网络处理中用于改善训练的稳定性和加速收敛。它应用于每一层的输入，在每一层上进行归一化。此外，它与批归一化（Batch Normalization）不同，不依赖于batch的大小。因此，它在小batch甚至是单个样本时也能很好地工作。

Layer Normalization 的公式如下：

假设输入为 $$\mathbf{x} = [x_1, x_2, \ldots, x_H]$$ ，其中 (H) 是隐藏单元的数量，那么：

1. 首先计算输入的均值和方差： $$\mu = \frac{1}{H} \sum_{i=1}^H x_i ] [ \sigma^2 = \frac{1}{H} \sum_{i=1}^H (x_i - \mu)^2$$
2. 然后使用计算出来的均值和方差对输入进行归一化： $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$ 这里的 ( $$\epsilon$$ ) 是一个非常小的常数，用于防止除零错误。
3. 最后，进行线性变换（可选的尺度和偏移参数）： \[ $$y_i = \gamma \hat{x}_i + \beta$$] 其中 ( $$\gamma$$ ) 和 ( $$\beta$$ ) 是可学习参数，分别是尺度和偏移。

总结一下，Layer Normalization 的步骤可以归纳为： $$\begin{equation} y_i = \gamma \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \end{equation}$$

这种归一化方法会独立地对每个样本进行操作，与其他样本无关，因此在处理序列或单个数据点时具有优势。

#### Transformer为什么要使用 LayerNorm 而不是 BatchNorm

在Transformer架构中，通常使用Layer Normalization（层归一化）而不是Batch Normalization（批归一化），这主要是由于以下几个原因：

1. **序列数据及变长特性**：Transformer主要用于处理序列数据，如自然语言处理中的句子。Batch Normalization依赖于批次内的统计信息（如均值和方差），这在固定大小的图像数据处理中效果很好。然而，对于变长的序列数据，批次内样本之间的长度差异使得统计信息的获取和应用比较复杂。
2. **并行化需求**：Transformer经常需要在序列的多个位置并行执行操作，尤其是自注意力机制中。Batch Normalization需要集合一批样本进行统计计算，而Layer Normalization是针对单个样本的，不依赖其他样本，因此更适合这种并行处理。
3. **递归依赖问题**：在语言模型中，处理序列的每一步可能依赖先前的步骤。Batch Normalization在这种情况下可能会引入复杂的依赖关系和不稳定性，而Layer Normalization直接在单个样本的维度上进行归一化，可以避免这些问题。
4. **简化训练过程**：Layer Normalization对于不同的输入和批量大小具有更好的鲁棒性，简化了训练过程。它仅对单个序列的特征进行标准化，所以不会因批次间的变化而受到影响。

总的来说，Layer Normalization在处理变长序列和自注意力机制中表现出更好的兼容性和稳定性，因此在Transformer架构中更为常用。
