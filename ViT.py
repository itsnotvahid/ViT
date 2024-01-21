import torch
from torch import nn


class PatchEmbedding(nn.Module):
    
    def __init__(self, embedding_dims, patch_size, square_image_size=256, in_channels=3):
        super().__init__()
        assert square_image_size % patch_size == 0, 'Image size should be divisible by patchsize'
        self.projection = nn.Conv2d(in_channels, embedding_dims, kernel_size=patch_size, stride=patch_size)
        self.linear = nn.Linear(embedding_dims, embedding_dims)
        
    def forward(self, x):
        patches = self.projection(x)
        N, C, H, W = patches.shape
        patches = patches.reshape(N, C, H*W).permute(0, 2, 1)
        output = self.linear(patches)
        return output


class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff): 
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = d_model // num_heads
    
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
    def __rearrange(self, vals):
        batch, sequence, _ = vals.shape
        vals = vals.reshape(batch, sequence, self.num_heads, self.head_dims).permute(0, 2, 1, 3)

        return vals
    
    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        
        key = self.key(key)
        query = self.query(query)
        value = self.value(value)
        
        # batch, head, sequence, f
        key = self.__rearrange(key)
        query = self.__rearrange(query)
        value = self.__rearrange(value)
        den = key.shape[-1] ** 0.5
        wgt = query @ key.transpose(-1, -2) / den
        
        if mask is not None:
            wgt = wgt.masked_fill(mask, float('-inf'))
        
        if key_padding_mask is not None:
            wgt = wgt.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
        
        wgt = torch.softmax(wgt, dim=-1)
        z = wgt @ value
        z = z.permute(0, 2, 1, 3)
        z = z.flatten(2)
        output = self.output(z)
        
        return output


class EncoderBlock(nn.Module):
    def __init__(self, d_model, ff_dim, num_head, dropout):
        super().__init__()
        assert d_model % num_head == 0, 'D model should be divisible by num head'
        self.mha = MultiHeadAttention(d_model, num_head)
        self.ff = FeedForward(d_model, ff_dim)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        temp = x
        x = self.ln1(x)
        x = self.dropout(self.mha(x, x, x))
        x = temp + x

        temp = x
        x = self.ln2(x)
        x = self.dropout(self.ff(x))
        x = temp + x
        return x


class Vit(nn.Module):
    
    def __init__(self, num_layers,
                 d_model,
                 ff_dim,
                 num_head,
                 num_patches,
                 num_classes,
                 dropout=0.1,
                 max_seq_len=1000):

        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
        self.encoder = nn.Sequential(*[EncoderBlock(d_model, ff_dim, num_head, dropout) for _ in range(num_layers)])
        
        self.patch_embedding = PatchEmbedding(d_model, num_patches)
        self.dropout = nn.Dropout()
        self.head = nn.Linear(d_model, num_classes)
                              
    def forward(self, x):
        y = self.patch_embedding(x)
        B, seq, _ = y.shape
        y += self.pos_embedding[:, :seq, :]
        cls_token = self.cls_token.repeat((B, 1, 1))
        y = torch.cat([y, cls_token], dim=1)
        y = self.dropout(y)
        y = self.encoder(y)
        y = y.mean(dim=1)
        y = self.head(y)
        return y
