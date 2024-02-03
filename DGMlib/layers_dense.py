import torch
import numpy
from DGMlib.layers import *

from torch.nn import Module, ModuleList, Sequential
from torch import nn
import gc


class DGM_d(nn.Module):
    def __init__(self, embed_f, k=5, distance=pairwise_euclidean_distances, sparse=True):
        super(DGM_d, self).__init__()
        
        self.sparse=sparse
        
        self.temperature = nn.Parameter(torch.tensor(1. if distance=="hyperbolic" else 4.).float())
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.k = k  ##
        
        self.debug=False
        if distance == 'euclidean':
            self.distance = pairwise_euclidean_distances
        else:
            self.distance = pairwise_poincare_distances
        '''x.shape torch.Size([1, 2708, 32])   A.shape  torch.Size([2, 10556])'''
    def forward(self, x, A, not_used=None, fixedges=None):
        x = self.embed_f(x,A)   # torch.Size([1, 2708, 4])
        
        if self.training:
            if fixedges is not None:                
                return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
            
            D, _x = self.distance(x) # torch.Size([1, 2708, 2708])  ## 这一步没了！！！
           
            #sampling here edges_hat  torch.Size([2, 13540])   logprobs.shape torch.Size([1, 2708, 5])
            edges_hat, logprobs = self.sample_without_replacement(D)
                
        else:
            with torch.no_grad():
                if fixedges is not None:                
                    return x, fixedges, torch.zeros(fixedges.shape[0],fixedges.shape[-1]//self.k,self.k,dtype=torch.float,device=x.device)
                D, _x = self.distance(x)

                #sampling here
                edges_hat, logprobs = self.sample_without_replacement(D)

              
        if self.debug:  # False
            self.D = D
            self.edges_hat=edges_hat
            self.logprobs=logprobs
            self.x=x

        return x, edges_hat, logprobs
    

    def sample_without_replacement(self, logits):
        b,n,_ = logits.shape # torch.Size([1, 2708, 2708])
#         logits = logits * torch.exp(self.temperature*10)
        logits = logits * torch.exp(torch.clamp(self.temperature,-5,5))

        q = torch.rand_like(logits) + 1e-8  ## 填充了均值为 0 方差为 1 的 正态分布 的随机值
        lq = (logits-torch.log(-torch.log(q)))  #torch.log(x) 就是ln(x)
        logprobs, indices = torch.topk(-lq,self.k) # [1, 14300, 5]
    
        rows = torch.arange(n).view(1,n,1).to(logits.device).repeat(b,1,self.k) # torch.Size([1, 2708, 5])
        edges = torch.stack((indices.view(b,-1),rows.view(b,-1)),-2) # torch.Size([1, 2, 13540])

        if self.sparse:
            return (edges+(torch.arange(b).to(logits.device)*n)[:,None,None]).transpose(0,1).reshape(2,-1), logprobs
        return edges, logprobs