import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64                                             # 并行处理64个序列
block_size = 256                                            # 每段最多处理256个字符
max_iters = 5000                                            # 训练迭代总步数
eval_interval = 500                                         # 每隔多少步评估一下loss
learning_rate = 3e-4

# 自动选择最佳设备：CUDA > MPS > CPU
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"Using device: {device}")
eval_iters = 200                                            # 评估时取多少批次求平均
n_embd = 384                                                # 每个字符会被映射成384维向量
n_head = 6                                                  # 注意力头数
n_layer = 6                                                 # Transformer层数
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# 用with语句打开文件，f为文件对象，通过接口方法操作文件
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() 

# 筛选出每个不同的字符
chars = sorted(list(set(text))) # 去重，转列表，排序
vocab_size = len(chars) # 词汇表大小：不同字符的数量

# 创建字符到整数的映射以及整数到字符到映射，方便后续编码和解码
stoi = { ch:i for i,ch in enumerate(chars) } # 字符到整数的映射
itos = { i:ch for i,ch in enumerate(chars) } # 整数到字符的映射

# 编码器和解码器
encode = lambda s: [stoi[c] for c in s] # 将字符串编码为整数列表
decode = lambda l: ''.join([itos[i] for i in l]) # 将整数列表解码为字符串

# 训练集和验证集划分
data = torch.tensor(encode(text), dtype=torch.long) # 将整个文本编码为整数张亮
n = int(0.9*len(data)) # 90%作为训练集，10%作为验证集
train_data = data[:n] # 训练集数据
val_data = data[n:] # 验证集数据

# 生成一个批次的数据
def get_batch(split):
    data = train_data if split == 'train' else val_data  # 选择数据集 
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 随机选择起始位置
    x = torch.stack([data[i:i+block_size] for i in ix]) # 输入序列
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # 目标序列
    x, y = x.to(device), y.to(device) # 移动到计算设备
    return x, y # 返回输入和目标序列

@torch.no_grad() # 不计算梯度，不做反向传播
# 预估损失函数
def estimate_loss():
    out = {}
    model.eval()  # 切换评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # 储存每次评估的损失
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    ## 初始化阶段：三个线性层
    def __init__(self, head_size):
        super().__init__() # 继承父类的初始化方法
        self.key = nn.Linear(n_embd, head_size, bias=False) # 将输入映射为键
        self.query = nn.Linear(n_embd, head_size, bias=False) # 将输入映射为查询
        self.value = nn.Linear(n_embd, head_size, bias=False) # 将输入映射为值
        # 下三角矩阵，防止模型看到未来信息
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 

        self.dropout = nn.Dropout(dropout) # 随机丢弃20%的神经元（注意力权重）

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel() # 初始化模型
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
