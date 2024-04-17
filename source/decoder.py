import collections
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from tokenizer import BPETokenizer
import loralib as lora
import pdb

class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.max_position_embeddings = 64

        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 16
        self.clip_max_norm = 0.1
        self.dir = './hw3_data/p2_data/'
        self.limit = -1

        # Learning Rates
        self.lr_backbone = 1e-5
        self.lr = 1e-4

        # Epochs
        self.epochs = 20
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-5

        # tokenizer
        self.tokenizer = BPETokenizer(encoder_file='./encoder.json', vocab_file='./vocab.bpe')

        # PEFTmode
        self.PEFT_mode = 'lora'
        # self.PEFT_mode = 'adapter'
        # self.PEFT_mode = 'prefix'
        # self.PEFT_mode = 'normal'


class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.PEFT_mode = cfg.PEFT_mode
        if self.PEFT_mode == 'lora':
            self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=4)
            self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=4)
        else:
            self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
            self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        # if self.PEFT_mode == 'prefix':
        #     self.pre = nn.Embedding(32, cfg.n_embd)
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):

        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class CrossAttention(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.att = None
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x, encoder_x):
        # print(x.size())
        # print(encoder_x.size())
        B, T, C = x.size() # batch, context, embedding
        A, S ,_ = encoder_x.size() # 196
        q, _ ,_ = self.c_attn(x).split(self.n_embd, dim=2)
        _, k, v = self.c_attn(encoder_x).split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(A, S, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(A, S, self.n_head, C // self.n_head).transpose(1, 2)
        # q = b, 128, 12, 64
        # k.T = b, 128, 64, 12
        # 1, 12, 61, 257
        # tmp = q @ k.transpose(-2, -1)
        # breakpoint()
        # tmp = tmp * (1.0 / math.sqrt(k.size(-1)))
        # breakpoint()
        # tmp = tmp.transpose(1, 2)
        # breakpoint()
        # print()
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        self.att = att 

        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):

    def __init__(self, cfg, layer):
        super().__init__()
        self.layer = layer
        self.PEFT_mode = cfg.PEFT_mode
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        if (self.PEFT_mode == 'adapter') and (layer > 10):
            self.adapter = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.crossattn = CrossAttention(cfg)
        self.max_len = cfg.max_position_embeddings

        if self.PEFT_mode == 'lora':
            self.mlp = nn.Sequential(collections.OrderedDict([
                ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=16)),
                ('act', nn.GELU(approximate='tanh')),
                ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=16)),
            ]))
        
        elif (self.PEFT_mode == 'adapter') and (layer > 10):
            self.adapter_mlp = nn.Linear(cfg.n_embd, cfg.n_embd)
            self.mlp = nn.Sequential(collections.OrderedDict([
                ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
                ('act', nn.GELU(approximate='tanh')),
                ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
            ]))
        
        else:
            self.mlp = nn.Sequential(collections.OrderedDict([
                ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
                ('act', nn.GELU(approximate='tanh')),
                ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
            ]))

    def forward(self, x, encoder_x):
        # print(input_.size())
        # x = input_[:,:self.max_len,:]
        # encoder_x = input_[:,self.max_len:,:]
        # print(x.size())
        if (self.PEFT_mode == 'adapter') and (self.layer > 10):
            x = x + self.attn(self.ln_1(x))
            x = x + self.adapter(x)
            x = x + self.crossattn(self.ln_2(x), encoder_x)
            x = x + self.adapter_mlp(x)
            x = x + self.mlp(self.ln_3(x))

        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.crossattn(self.ln_2(x), encoder_x)
            # print(x.size()) b*seq*768
            x = x + self.mlp(self.ln_3(x))

        return x

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.PEFT_mode = cfg.PEFT_mode
        self.block_size = cfg.block_size
        if self.PEFT_mode == 'prefix':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe = nn.Embedding(cfg.block_size, cfg.n_embd), 
                pre = nn.Embedding(32, cfg.n_embd),
                h = nn.Sequential(*[Block(cfg, layer) for layer in range(cfg.n_layer)]),
                ln_f = nn.LayerNorm(cfg.n_embd)
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe = nn.Embedding(cfg.block_size, cfg.n_embd), 
                h = nn.Sequential(*[Block(cfg, layer) for layer in range(cfg.n_layer)]),
                ln_f = nn.LayerNorm(cfg.n_embd)
            ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.block_layer = list(self.transformer.h.children())

        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, encoder_x: Tensor): # x = 16 * 128 * 768  encoder_x(encoder_output) = 16*196*768


        if self.PEFT_mode == 'prefix':
            prefix = torch.arange(32, dtype=torch.long, device=x.device).unsqueeze(0)
            prefix = prefix.repeat(self.cfg.batch_size,1)
            x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
            prefix_result = self.transformer.pre(prefix)
            # print(prefix_result.size()) # [batch_size, 32, 768]
            # print(x.size()) # 14, 64
            pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
            # print(pos)
            # print(prefix)
            # print(x.size())
            x = self.transformer.wte(x) + self.transformer.wpe(pos) # [b*seq*768]
            x = torch.concat((prefix_result, x), dim=1)

            for block in self.block_layer:
                x = block(x, encoder_x)

            x = self.lm_head(self.transformer.ln_f(x))
            x = x[:, 32:, :]
            print(x.size())

        else:
            x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
            pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
            # print(x.size()) # 14,64
            # print(pos.size()) # 1, 64
            x = self.transformer.wte(x) + self.transformer.wpe(pos)

            for block in self.block_layer:
                x = block(x, encoder_x)

            x = self.lm_head(self.transformer.ln_f(x))

        return x
