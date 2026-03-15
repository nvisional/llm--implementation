# 01 - GPT 架构精读（nanochat/gpt.py）

> 学习日期：2026-03-15
> 文件：`nanochat/nanochat/gpt.py`（508行）
> 本地验证：4层模型，MPS设备，50步训练跑通 ✓

---

## 一、整体结构

```
tokens (B, T)
   │
   ▼ wte (Embedding)
x (B, T, C)  ← C = n_embd (隐藏维度)
   │
   ▼ norm(x)  ← embedding后立刻做RMSNorm（现代做法，无可学参数）
   │
   ├── smear (前一个token的embedding混入，廉价的bigram信息)
   │
   ▼ for i in range(n_layer):
   │     x = resid_lambdas[i] * x + x0_lambdas[i] * x0  ← 每层可学习的残差缩放
   │     x = Block(x, ve, ...)                            ← Transformer块
   │
   ▼ x = x - backout_lambda * x_backout  ← 减去中间层特征（去除低层特征）
   ▼ norm(x)
   ▼ lm_head (Linear)
   ▼ softcap (tanh限幅到[-15,15])
logits (B, T, vocab_size)
```

---

## 二、每个组件的设计要点

### 1. GPTConfig（配置）

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6       # Query头数
    n_kv_head: int = 6    # KV头数（GQA时 < n_head）
    n_embd: int = 768
    window_pattern: str = "SSSL"  # 滑动窗口模式
```

**window_pattern**：控制每一层注意力的窗口大小。
- `L` = 全上下文（full context）
- `S` = 四分之一上下文（sliding window）
- `"SSSL"` = 前三层用短窗口，第四层用全窗口，依此循环。最后一层**永远是L**。
- 好处：大部分层只看局部，省计算量；最后一层整合全局信息。

---

### 2. norm()：无参数 RMSNorm

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

- RMSNorm = 只做L2归一化，不学均值和方差。比LayerNorm更快、更稳定。
- **无可学习参数**（传统LayerNorm有γ、β）。
- 在 embedding 之后立刻normalize，而不是在每个Block之前（Pre-Norm已成主流）。

---

### 3. Linear：主动类型转换

```python
class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))
```

- 权重用 fp32 存（优化器精度高），前向时转成激活值的dtype（bf16）。
- 替代了 `torch.autocast`，更显式可控。

---

### 4. CausalSelfAttention：注意力机制

```
x (B, T, C)
   │
   ├── c_q → q (B, T, n_head, head_dim)
   ├── c_k → k (B, T, n_kv_head, head_dim)   ← GQA：KV头比Q少
   └── c_v → v (B, T, n_kv_head, head_dim)
        │
        ├── Value Residual (ve_gate)：混入token的"静态价值"
        │
        ├── apply_rotary_emb(q, k)  ← RoPE：相对位置编码
        ├── norm(q), norm(k)        ← QK Norm：防止注意力分数爆炸
        ├── q *= 1.2, k *= 1.2      ← 让注意力更锐利（sharper）
        │
        └── Flash Attention（训练）/ flash_attn_with_kvcache（推理）
              │
              ▼
        y (B, T, C)
        c_proj → 输出
```

**关键设计决策：**

| 技术 | 作用 |
|------|------|
| GQA（Group Query Attention） | n_kv_head < n_head，KV cache更小，推理更快 |
| RoPE（旋转位置编码） | 无需学习位置embedding，外推性更好 |
| QK Norm | q和k都做norm，防止softmax饱和 |
| Value Residual | ve = token的静态embedding，与动态v混合，类似ResFormer |

**RoPE实现原理（apply_rotary_emb）：**
```
将head_dim分两半：[x1, x2]
y1 = x1 * cos + x2 * sin   ← 旋转
y2 = -x1 * sin + x2 * cos
```
本质：给每个位置的向量乘以一个旋转矩阵，使 q·k 的内积自然编码相对位置。

---

### 5. MLP：ReLU²激活

```python
def forward(self, x):
    x = self.c_fc(x)     # 升维：C → 4C
    x = F.relu(x).square()  # ReLU²，比GELU更稀疏
    x = self.c_proj(x)   # 降维：4C → C
    return x
```

- ReLU² = relu(x)²，激活更稀疏（大量神经元为0），有利于特征分离。
- 相比GELU计算更快，效果相近。

---

### 6. Block：Pre-Norm残差结构

```python
def forward(self, x, ve, cos_sin, window_size, kv_cache):
    x = x + self.attn(norm(x), ...)  # 先norm再attention，结果加回x
    x = x + self.mlp(norm(x))        # 先norm再MLP，结果加回x
    return x
```

这是**Pre-Norm**：norm在操作之前，训练更稳定（原始Transformer是Post-Norm）。

---

### 7. GPT.forward()：完整前向过程中的额外技巧

**Smear（前token混入）：**
```python
gate = smear_lambda * sigmoid(smear_gate(x[:, 1:, :24]))
x = cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]])
```
把前一个token的embedding用可学习的gate混入当前位置。
代价极低（只用前24个channel做门控），但给模型"免费的bigram信息"。

**x0 残差混入：**
```python
x = resid_lambdas[i] * x + x0_lambdas[i] * x0
```
每层都能从初始embedding补充信息，帮助深层网络不丢失token本身的语义。

**Backout（减去中间层）：**
```python
x = x - backout_lambda * x_backout  # x_backout 是中间那一层的输出
```
从最终表征中减去中间层的低级特征，让最后的logit更聚焦于高层语义。

**Logit Softcap：**
```python
logits = 15 * torch.tanh(logits / 15)
```
平滑地把logits限制在[-15, 15]，防止训练初期loss爆炸。Gemma用的同款技巧。

---

## 三、训练 vs 推理的差异

| 场景 | 注意力 | 关键参数 |
|------|--------|---------|
| 训练 | flash_attn_func（全序列，有causal mask） | kv_cache=None |
| 推理 | flash_attn_with_kvcache（逐token生成） | kv_cache=KVCache对象 |

推理时KV cache存储所有历史的k、v，避免重复计算。每生成一个token，cache向前推进一步。

---

## 四、参数分组与优化器

不同参数用不同优化器和学习率：

| 参数组 | 优化器 | 原因 |
|--------|--------|------|
| 矩阵权重（Attention/MLP） | **Muon** | 矩阵专用优化器，收敛更快 |
| Embedding (wte) | AdamW | 查表操作，梯度稀疏 |
| lm_head | AdamW | 与embedding相关 |
| 标量参数 | AdamW | 维度低，AdamW足够 |

---

## 五、本次训练结果

```
模型：4层，n_embd=256，2头，约7000万参数
设备：Apple M系列 MPS
速度：~7,800 tok/sec
Steps：50步
Loss：11.09 → 11.01（50步太少，仅验证pipeline）
Validation bpb：3.292 → 3.238
```

50步远不足以学到任何语言规律，loss还很高是正常的。
真正的训练需要数十亿token（见speedrun.sh：GPU上跑38B tokens）。

---

## 六、待深入的问题

- [ ] Muon优化器的原理是什么？为什么比AdamW更适合矩阵？（见 `optim.py`）
- [ ] Value Residual（ResFormer）是什么论文提出的？效果如何？
- [ ] Flash Attention 3 的 tiling 算法如何避免HBM读写？（见 `flash_attention.py`）
- [ ] KV Cache 如何动态扩容？（见 `engine.py`）
- [ ] ClimbMix-400B 数据集的构成是什么？和 FineWebEdu 的区别？

---

*下一篇：`engine.py` — KV Cache与推理优化*
