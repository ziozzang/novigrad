"""Small feature/time token-to-latent bridge for simulated site observations.

These inputs are engineered site/time features, not biological tokens or neural recordings.
An optional 32-to-640 projection belongs to the downstream LLM adapter, outside this module.
Mode 'fixed' fixes query vectors only; its attention and site/time metadata layers train.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MODES=('learned','fixed','pooled_mlp')

class NeuralResampler(nn.Module):
    def __init__(self,mode='learned',dim=32,latents=4,heads=4,sites=4):
        super().__init__()
        if mode not in MODES:raise ValueError(f'mode must be one of {MODES}')
        if dim<=0 or latents<=0 or sites<=0 or heads not in (1,4) or dim%heads:raise ValueError('invalid dimensions or heads')
        self.mode,self.dim,self.latents,self.heads,self.sites=mode,dim,latents,heads,sites
        self.site_embedding=nn.Embedding(sites,dim)
        self.time_embedding=nn.Linear(1,dim)
        if mode in ('learned','fixed'):
            self.attention=nn.MultiheadAttention(dim,heads,batch_first=True)
            self.norm=nn.LayerNorm(dim)
            if mode=='learned':self.queries=nn.Parameter(torch.empty(latents,dim));nn.init.normal_(self.queries,std=dim**-.5)
            else:
                position=torch.arange(latents,dtype=torch.float32)[:,None];frequency=torch.exp(-torch.arange(0,dim,2,dtype=torch.float32)*torch.log(torch.tensor(10000.))/dim)
                q=torch.zeros(latents,dim);q[:,0::2]=torch.sin(position*frequency);q[:,1::2]=torch.cos(position*frequency[:q[:,1::2].shape[1]])
                self.register_buffer('queries',q)
        else:
            # At default dimensions this has 4,667 trainable parameters versus 4,608
            # for learned-query attention (1.3% difference).
            hidden=max(1,round((self._attention_parameter_target()-latents*dim-2*dim-sites*dim)/(dim+latents*dim+1)))
            self.pool_mlp=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,latents*dim))

    def _attention_parameter_target(self):
        return 4*self.dim*self.dim+4*self.dim+self.latents*self.dim+2*self.dim+self.sites*self.dim+2*self.dim

    @property
    def trainable_parameters(self):return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _validate(self,features,valid_mask,site_ids,timestamps,observation_time):
        if features.ndim!=3 or features.shape[2]!=self.dim:raise ValueError(f'features must be [B,T,{self.dim}]')
        shape=features.shape[:2]
        if valid_mask.shape!=shape or valid_mask.dtype!=torch.bool:raise ValueError('valid_mask must be bool [B,T]')
        if site_ids.shape!=shape or site_ids.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):raise ValueError('site_ids must be integer [B,T]')
        if timestamps.shape!=shape or not torch.is_floating_point(timestamps):raise ValueError('timestamps must be floating [B,T]')
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():raise ValueError('features must be finite floating values')
        if not torch.isfinite(timestamps).all():raise ValueError('timestamps must be finite')
        if not valid_mask.any(1).all():raise ValueError('each row needs at least one valid token')
        valid_sites=site_ids[valid_mask]
        if (valid_sites<0).any() or (valid_sites>=self.sites).any():raise ValueError('valid site_ids out of bounds')
        obs=torch.as_tensor(observation_time,dtype=timestamps.dtype,device=timestamps.device)
        if obs.ndim==0:obs=obs.expand(shape[0])
        if obs.shape!=(shape[0],) or not torch.isfinite(obs).all():raise ValueError('observation_time must be finite scalar or [B]')
        if (timestamps[valid_mask]>obs[:,None].expand(shape)[valid_mask]).any():raise ValueError('future timestamp exceeds observation_time')
        return obs

    def forward(self,features,valid_mask,site_ids,timestamps,observation_time):
        obs=self._validate(features,valid_mask,site_ids,timestamps,observation_time)
        # A fixed time unit makes relative times invariant to clock-origin shifts.
        relative=(timestamps-obs[:,None]).unsqueeze(-1).to(features.dtype)
        tokens=features+self.site_embedding(site_ids.clamp(0,self.sites-1))+self.time_embedding(relative)
        if self.mode=='pooled_mlp':
            mask=valid_mask.unsqueeze(-1);pooled=(tokens*mask).sum(1)/mask.sum(1);return self.pool_mlp(pooled).reshape(len(features),self.latents,self.dim)
        queries=self.queries.unsqueeze(0).expand(len(features),-1,-1)
        value,_=self.attention(queries,tokens,tokens,key_padding_mask=~valid_mask,need_weights=False)
        return self.norm(queries+value)

    def config(self):return {'mode':self.mode,'dim':self.dim,'latents':self.latents,'heads':self.heads,'sites':self.sites}

def save_resampler(model,path):
    metadata={'format':'novigrad.neural_resampler','version':'1','config':json.dumps(model.config(),sort_keys=True,separators=(',',':')),'scope':'simulated_site_time_features'}
    state={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('cannot save non-finite state')
    save_file(state,str(path),metadata=metadata)

def load_resampler(path):
    with safe_open(str(path),framework='pt',device='cpu') as f:metadata=f.metadata() or {}
    if metadata.get('format')!='novigrad.neural_resampler' or metadata.get('version')!='1' or metadata.get('scope')!='simulated_site_time_features':raise ValueError('invalid resampler metadata')
    try:config=json.loads(metadata['config']);model=NeuralResampler(**config)
    except (KeyError,TypeError,ValueError,json.JSONDecodeError) as e:raise ValueError('invalid resampler config') from e
    state=load_file(str(path),device='cpu')
    if set(state)!=set(model.state_dict()):raise ValueError('invalid resampler tensor schema')
    for name,expected in model.state_dict().items():
        value=state[name]
        if value.dtype!=expected.dtype or value.shape!=expected.shape or not torch.isfinite(value).all():raise ValueError(f'invalid resampler tensor: {name}')
    model.load_state_dict(state,strict=True);return model
