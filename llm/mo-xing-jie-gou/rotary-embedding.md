# Rotary Embedding

LlamaRotaryEmbedding 是一种位置编码方法，通过对查询（query）和键（key）向量施加旋转变换，使得模型能够将相对位置信息融入到注意力机制中。相对于传统的位置编码，Rotary Embedding 在计算效率和效果上都有提升。

#### 公式解释

LlamaRotaryEmbedding 的核心在于通过旋转矩阵对查询和键向量进行变换，使得位置编码直接作用于词向量。其公式如下：

1. **旋转频率计算**： \[ \text{inv\_freq}\[i] = \frac{1}{\text{base}^{\frac{2i}{d\}}} ] 其中 ( d ) 为嵌入维度的一半，(\text{base}) 是一个常数（通常为10000）。
2. **位置编码计算**： \[ \text{pos\_enc}\_{t, i} = t \cdot \text{inv\_freq}\[i] ] 其中 ( t ) 是位置序号。
3. **构造旋转矩阵**： \[ \text{cos\_t} = \cos(\text{pos\_enc}\_t) \quad \text{sin\_t} = \sin(\text{pos\_enc}\_t) ]
4. **应用旋转位置编码**： 对于查询和键向量 ( q ) 和 ( k )，其变换公式为： \[ q' = q \cdot \cos\_t + (-q\_{\text{half\}}) \cdot \sin\_t ] \[ k' = k \cdot \cos\_t + (-k\_{\text{half\}}) \cdot \sin\_t ] 其中，( q\_{\text{half\}} ) 表示拆分后的向量 ( q ) 的后一半。

#### 伪代码

```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 计算逆频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # 扩展逆频率和位置ID
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # 计算位置编码
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # 应用旋转位置编码
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k)

    return q_embed, k_embed
```

#### 解释

1. **初始化**：
   * `self.inv_freq` 是一个（1, dim//2）大小的张量，用于存储频率的倒数。
   * `self.scaling_factor` 是一个缩放因子，用于位置缩放（可选）。
2. **forward 方法**：
   * `inv_freq_expanded` 扩展了频率倒数，以便与输入位置ID相乘。
   * `position_ids_expanded` 将位置ID扩展为浮点数，并调整形状以进行矩阵乘法。
   * 通过矩阵乘法计算 (\text{freqs})，然后将其拼接为位置编码，并计算 cos 和 sin。
3. **apply\_rotary\_pos\_emb 函数**：
   * 定义了一个辅助函数 `rotate_half`，用于旋转向量的一半。
   * 对查询和键向量施加位置旋转编码 ( q' ) 和 ( k' )。

通过伪代码和公式，你可以了解 LlamaRotaryEmbedding 是如何通过旋转变换将位置信息融入到查询和键向量中的。这种方法相对于传统位置编码具有更好的性能和计算效率。
