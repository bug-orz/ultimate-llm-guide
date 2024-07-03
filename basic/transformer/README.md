# Transformer

{% content-ref url="attention-block.md" %}
[attention-block.md](attention-block.md)
{% endcontent-ref %}

#### 如何计算transformer的参数量

Transformer模型的参数量可以通过计算其各个组成部分的参数来求得。Transformer模型主要由以下几部分组成：输入嵌入层（embedding layer）、自注意力层（self-attention layer）、前馈网络层（feed-forward network layer）、层规范化（layer normalization）和输出全连接层（output fully connected layer）。下面是计算每一部分参数量的方法：

#### 1. 输入嵌入层（Embedding Layer）

假设词汇表的大小为 ( V )，嵌入的维度为 ( d )，那么输入嵌入层的参数量即为词汇表中的每个单词都有一个长度为 ( d ) 的向量。因此，参数量为： $$\text{Embedding Layer Parameters} = V \times d$$

#### 2. 自注意力层（Self-Attention Layer）

**2.1 多头注意力机制（Multi-Head Attention）**

假设有 $$h$$ 个注意力头，每个头的维度为 ( d\_k )，输入和输出的维度为 ( d )（通常 ( d ) 可以是 ($$h \times d_k$$ )）。注意力机制的参数量主要包括：查询（query）、键（key）、值（value）线性变换的参数和输出线性变换的参数。

每个头的查询、键、值线性变换的参数量为： $$\text{Parameter per head} = 3 \times (d \times d_k)$$ 由于有 ( h ) 个头，总的查询、键、值线性变换的参数量为： $$\text{QKV Parameters} = h \times 3 \times (d \times d_k)$$ 输出线性变换的参数量为： $$\text{Output Linear Parameters} = d \times d$$ 因此，多头注意力机制的总参数量为： $$\text{Multi-Head Attention Parameters} = h \times 3 \times (d \times d_k) + (d \times d)$$

#### 3. 前馈网络层（Feed-Forward Network Layer）

前馈网络一般由两个全连接层组成。假设中间层的维度为 ( d\_{ff} )，输入和输出的维度为 ( d )。 \
第一个全连接层的参数量为： $$\text{First Linear Layer Parameters} = d \times d_{ff}$$ \
第二个全连接层的参数量为： $$\text{Second Linear Layer Parameters} = d_{ff} \times d$$ \
因此，前馈网络的总参数量为：$$\text{Feed-Forward Network Parameters} = d \times d_{ff} + d_{ff} \times d$$

#### 4. 层规范化（Layer Normalization）

层规范化包含两个可训练的参数：缩放因子 $$\gamma$$ 和偏移因子 $$\beta$$，每个参数的维度为 ( d )，所以层规范化的总参数量为： $$\text{Layer Normalization Parameters} = 2 \times d$$

#### 5. 输出全连接层（Output Fully Connected Layer）

通常在Transformer的最后会有一个输出全连接层，用于预测。假设输出类别数为 ( C )，那么输出全连接层的参数量为： $$\text{Output Linear Layer Parameters} = d \times C$$

#### 汇总

假设Transformer的层数为 ( L )，那么总的参数量按以下公式汇总：

1. Embedding Layer Parameters： $$V \times d$$
2. Multi-Head Attention Parameters： $$L \times (h \times 3 \times (d \times d_k) + (d \times d))$$
3. Feed-Forward Network Parameters： $$L \times (d \times d_{ff} + d_{ff} \times d)$$
4. Layer Normalization Parameters： $$L \times 2 \times d$$
5. Output Linear Layer Parameters： $$d \times C$$

总参数量为所有部分的参数之和： $$\text{Total Parameters} = V \times d + L \times (h \times 3 \times (d \times d_k) + (d \times d)) + L \times (d \times d_{ff} + d_{ff} \times d) + L \times 2 \times d + d \times C$$

这个公式可以帮助你较精确地计算一个标准Transformer模型的参数总量。
