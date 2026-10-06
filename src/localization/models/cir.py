#%%

import torch
import torch.nn as nn
from src.localization.models.base import ModelResult
from src.localization.models.csi import SelfAttention1D, GaussianNoise , Block
class CIRModel(nn.Module):
    def fe_block(self,num_channels,stride=1):
        r = 8

        return nn.Sequential(        

        nn.LazyConv2d( num_channels, kernel_size=(5),stride=stride, padding=2),
        
        nn.LeakyReLU(),
        
        nn.LazyInstanceNorm2d(),
    
        )
        
    
        
    def __init__(self,stride=1):
        super(CIRModel, self).__init__()
        self.flatten = nn.Flatten()
        
        self.block_1024 = Block(1024)
        self.block_512 = Block(512)
        self.block_256 = Block(256)
        
        self.block_128 = Block(128)

        
        
       

        self.linear_2 =nn.LazyLinear( 2, bias=False)




        num_channels=32


        self.s1=self.fe_block(num_channels,stride=stride)
        self.s2=self.fe_block(num_channels,stride=stride)

        
        self.linear_2_2 = nn.LazyLinear(2)
  
        self.mask_ratio = 0.0
        
        self.noise = GaussianNoise(0.01)
        
        self.linear_time_1 = nn.LazyLinear(1,bias=True)



                        

    def random_mask(self, x):
        """Apply random mask to features."""
        mask = torch.rand_like(x) < self.mask_ratio 
        x_masked = x.masked_fill(mask, 0.0)  

        return x_masked, mask
    


    def forward(self, input,t=None,index = None)->ModelResult:
        mask = None
        if len(input.shape) == 3:
            x = input[:,None,:,:]
        else:
            x = input
            
        if self.training and self.mask_ratio > 0.0: 

                            #x = self.noise(x)
                            pass
        x = self.s1(x)
        x = self.s2(x)

        x = torch.flatten(x, 1)    


        if self.training and self.mask_ratio > 0.0: 

                    #x = self.noise(x)
                    x, mask = self.random_mask(x)
                    

                    pass
        else:
                    mask = torch.ones_like(x, dtype=torch.bool)  
                    
        x = self.block_1024(x)
        x = self.block_512(x)
        x = self.block_256(x)
        h = x
        x = self.block_128(x)
        

        _2 = self.linear_2(x)
        
        time_prediction = self.linear_time_1(h)


        return ModelResult(
            mask  = mask,
            x_flattened = input,
            embeddings_2d = _2,
            embeddings_highdim=h,
            time_prediction = time_prediction
        )
    

if __name__ == '__main__':
    model = CIRModel()


   
