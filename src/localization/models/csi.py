from src.localization.models.base import ModelResult
import torch
import torch.nn as nn
from src.localization.models.mixture_density import MixtureDensityNetwork,NoiseType

class FeatureEngineeringLayer(nn.Module):
    def __init__(self):
        super(FeatureEngineeringLayer, self).__init__()

    def forward(self, csi):
        # csi: (d, a, m, t), complex tensor
        sample_autocorrelations = torch.einsum('damt,dbnt->dtabmn', csi, torch.conj(csi))
        # Return stacked real and imaginary parts along last axis
        return torch.stack([sample_autocorrelations.real, sample_autocorrelations.imag], dim=-1)

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        #if self.training:
            noise = torch.randn_like(x) * self.sigma * x.abs().detach()  # Scale noise by input magnitude
            return x + noise
        #return x
class SelfAttention1D(nn.Module):
    def __init__(self, N,d_model=32, nhead=4):
        super().__init__()
        self.x_proj = nn.LazyLinear(d_model)   # project scalar → vector
        self.y_proj = nn.LazyLinear(d_model)   # project scalar → vector
        
        self.x_norm = nn.LayerNorm(d_model)
        self.y_norm = nn.LayerNorm(d_model)
        
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.output_proj = nn.LazyLinear( 1)  # project vector → scalar
        self.ln = nn.LayerNorm(N)

    def forward(self, x,y):
        # x: [B, N]
        B, N = x.shape

        # 1️⃣ Project to embedding space
        #x_emb = torch.cat((x,y),-1)
        x_emb = self.x_proj(x.unsqueeze(-1))  # [B, N, d_model]
        y_emb = self.y_proj(y.unsqueeze(-1))     # [B, N, d_model]
        
        x_emb = self.x_norm(x_emb)
        y_emb = self.y_norm(y_emb)
        


        # 2️⃣ Self-attention
        attn_out, attn_weights = self.attn(x_emb, y_emb, y_emb)

        attn_out = self.output_proj(attn_out)  # [B, N]
        attn_out = attn_out.squeeze(-1)
        out = attn_out  # Residual connection
        #out = self.ln(attn_out + x)

        return out

class FeatureAttention(nn.Module):
    def __init__(self, in_features, reduction=4):
        super().__init__()
        
        hidden = in_features // reduction
        
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_features),
            nn.Sigmoid()  # produces weights in [0, 1]
        )

    def forward(self, x):
        """
        x: (batch_size, in_features)
        """
        weights = self.attention(x)   # (batch, in_features)
        return x * weights
    
class Block(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.linear = nn.LazyLinear( out_features)
        self.activation = nn.LeakyReLU()
        self.layernorm = nn.LayerNorm(out_features)
        #self.feature_attention = FeatureAttention(out_features)
        self.dropout = nn.Dropout(0.05)
        
    def forward(self, x):
        """
        x: (batch_size, in_features)
        """
        x = self.linear(x)
        x = self.layernorm(x)
        #x = self.feature_attention(x)
        x = self.activation(x)
        #x = self.dropout(x)
        
        
        return x

from torch.nn import functional as F
class CSIModel(nn.Module):
    def __init__(self ):
        super(CSIModel, self).__init__()
        self.mask_ratio = 0.5
        self.mdn = MixtureDensityNetwork(512,1,8,hidden_dim=32,noise_type=NoiseType.DIAGONAL)
        
        self.fe_layer = FeatureEngineeringLayer()
        self.flatten = nn.Flatten()
        
        self.block_1024 = Block(1024)
        self.block_512 = Block(512)
        self.block_256 = Block(256)
        self.block_128 = Block(128)
        self.block_64 = Block(64)
        self.block_32 = Block(32)
        self.block_16 = Block(16)
        

        
        self.block_128_2 = Block(128)
        
        self.linear_2_2 = nn.LazyLinear( 2)

        self.linear_2 = nn.LazyLinear( 2 , bias=False)
        
        self.block_time_128 = Block(8)
        self.linear_time_1 = nn.LazyLinear(1,bias=True)
        
        self.bn = nn.LazyBatchNorm1d()
        

        
        self.do = nn.Dropout(0.2)
        self.noise = GaussianNoise(sigma=0.01)

    def random_mask(self, x):
        """Apply random mask to features."""

        batch_size, dim = x.shape
        mask = torch.rand(batch_size, dim, device=x.device) < self.mask_ratio # true = mask
        x_masked = x.masked_fill(mask, 0.0)  # Masked positions set to zero
        return x_masked, mask

    def dropout_mask(self):
        block_list = [self.block_1024,self.block_512,self.block_256,self.block_128,self.block_64,self.block_32,self.block_16]
        if self.training and self.mask_ratio>0.0:
            for block in block_list:
                block.dropout.p = self.mask_ratio
        else:
            for block in block_list:
                    block.dropout.p = 0.0
                
        pass
        
        
    def f2(self, x):
        x = self.block_128(x)
        x = self.block_64(x)
        _2 = self.linear_2(x)
        return _2
    
    def forward(self, x:torch.Tensor,t:torch.Tensor,index:torch.Tensor)->ModelResult:
        mask = None
        x_flatten = None
        time = None
        y = None
        p = None
        x = self.fe_layer(x)
        x = self.flatten(x)
        #x_flatten = x.clone()

        if  self.training and self.mask_ratio>0.0:  # only mask during training
            x, mask = self.random_mask(x)
            x= x * 1/(1.0 - self.mask_ratio)  # scale to keep expected value the same
            #x = self.noise(x)
            

            pass
        else:
            mask = torch.ones_like(x, dtype=torch.bool)  # no positions masked
            x = x
            pass
        
        

        

        x = self.block_1024(x)
        x = self.block_512(x)
        
        x = self.block_256(x)
        

        h = x
        
        x = self.block_128(x)
        

        
        x = self.block_64(x)

        _2 = self.linear_2(x)

        time = self.linear_time_1(h)
        


        
        return ModelResult(
            mask  = mask,
            x_flattened = x_flatten,
            embeddings_2d = _2,
            embeddings_32= p,
            embeddings_128= x,
            embeddings_highdim=h,
            time_prediction=time
            

        )
    

if __name__ == '__main__':
    model = CSIModel()

    x = torch.rand([32,4, 8, 13]) + 1j * torch.rand([32,4, 8, 13])
    t =  torch.rand((x.shape[0],1))
    y = model(x,t,0)
    print(y)
    
    """scores = self.fwa(x)
    #scores = scores.masked_fill(mask, 0.0)  # Masked positions set to -inf for attention
    scores = torch.sigmoid(scores)  # [B, N]
    
    #scores = scores.masked_fill(~mask, float('-inf'))  # Masked positions set to -inf for attention
    #scores = torch.softmax(scores, dim=1)  # Normalize to sum to 1 across N
    x = x * scores  # Weighted sum of features"""
        