# 向量流动视角

一个decoder-only 的LLM ， d\_model=768

输入 batch=64，序列长度 512 的序列， \
编码后的向量为 \[64, 512, 768]\
QKV的线性网络为 $$768\times768$$\
编码后的向量乘以QKV以后，维度是 \[64, 512, 768]\
转变为以下维度：\[batch\_size,  num\_heads, 序列长度,head\_dim].     \[64, 12, 512, 64].      \
(self.head\_dim = d\_model // num\_heads)

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

QK点乘后的维度为 \[64, 12, 512, 512]\
再乘以 V ( \[64, 12, 512, 64]) 以后， 得到的维度是 \[64, 12, 512, 64]\
最后放缩为 \[64, 512, 768]\
\




