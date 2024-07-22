# Cross Attention

在神经网络和特别是Transformer结构中，Cross Attention机制是连接编码器（encoder）和解码器（decoder）的关键组件。Cross Attention允许解码器在生成输出时基于编码器生成的表示进行信息整合，从而使得生成的输出能够更好地反映输入序列的特性。

以下是关于Cross Attention的详细介绍及其公式：

#### 1. 基础概念

**自注意力机制（Self-Attention）**

自注意力机制中，输入是一个序列，假设它有  $$T$$ 个元素，每个元素可以表示为一个向量。对于输入序列 $$X = [x_1, x_2, \ldots, x_T]$$，每个 ( x\_i ) 各自转化为三种不同的向量：查询向量 $$q_i$$ 、键向量 $$k_i$$ 和值向量  $$v_i$$ 。

公式如下： $$[ q_i = W_q x_i ] [ k_i = W_k x_i ] [ v_i = W_v x_i ]$$

其中， $$W_k$$， $$W_k$$  和 $$W_v$$ 是可学习的参数矩阵。

**多头机制（Multi-Head Attention）**

注意力机制可以被执行多次（多个头），以捕捉不同的关系： $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h) W_o$$ \
其中，每个头的计算方式为： $$\text{head}_i = \text{Attention}(Q W_q^i, K W_k^i, V W_v^i)$$

#### 2. Cross Attention 机制

Cross Attention 也是基于注意力机制的，但与自注意力不同的是，Cross Attention在解码器中，查询来自解码器的上一层，而键和值来自编码器的输出。这允许解码器在生成每个输出时，引用整个输入序列的信息。

**Cross Attention 公式**

假设编码器输出为 $$H = [h_1, h_2, \ldots, h_T]$$，解码器的输入为 $$S = [s_1, s_2, \ldots, s_T]$$。

1. **查询、键和值的计算**
   * 解码器的每个元素生成查询向量 $$q$$ ： $$[ q_i = W_q s_i ]$$
   * 编码器的输出生成键向量  $$k$$  和值向量  $$v$$ ： $$[ k_j = W_k h_j ] [ v_j = W_v h_j ]$$
2. **注意力权重的计算** 查询和键点积，生成注意力得分，然后通过Softmax归一化： $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$ 其中，Q  是查询矩阵，K 是键矩阵， V  是值矩阵， $$d_k$$ 是键向量的维度（归一化因子）。

将公式展开，用于具体的 ( i ) 和 ( j )： $$\alpha_{ij} = \frac{\exp(q_i \cdot k_j /\sqrt{d_k})}{\sum_{j'} \exp(q_i \cdot k_{j'} /\sqrt{d_k})} ] [ a_i = \sum_{j} \alpha_{ij} v_j$$

这里， $$\alpha_{ij}$$ 表示第 ( i ) 个解码器状态与第 ( j ) 个编码器状态的注意力权重，而 $$a_i$$ 则是基于这些权重计算得到的新的解码器状态。

#### 3. 多头 Cross Attention

与自注意力机制一样，Cross Attention 也可以通过多头机制增强表示能力： $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h) W_o$$\
&#x20;其中，每个头的计算方式为： $$\text{head}_i = \text{Attention}(Q W_q^i, K W_k^i, V W_v^i)$$

通过这种方式，Cross Attention将在多个子空间计算注意力，以捕捉不同的特征和信息关系。

#### 总结

Cross Attention是连接编码器和解码器的重要组成部分，通过使用来自解码器作为查询，编码器作为键和值，能够有效地将编码器的上下文信息注入到生成过程当中。其核心公式主要涉及注意力权重的计算和基于注意力权重生成新的解码器状态。多头机制进一步增强了Cross Attention的灵活性和表示能力。
