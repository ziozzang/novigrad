"""Export the actual PN port identities and learned software coordinates."""
import json
import numpy as np
from safetensors.numpy import load_file
from precise_bridge import ROOT,OUT,MODES,PortBridge,checkpoint
from inhibition_mechanism import sha256

def main():
    graph=load_file(str(checkpoint(601)))
    fanout=np.bincount(graph['input_pre'].astype(int),minlength=len(graph['input_ids']))
    result={'boundary':'Real graph root IDs with engineered embedding coordinates; PCA axes are not biological odor/glomerulus identities. Input identity order comes from the native checkpoint, not TSV sorting.',
            'graph_checkpoint_sha256':sha256(checkpoint(601)),'input_ports':[],'mappings':{}}
    for i,root_id in enumerate(graph['input_ids']):
        result['input_ports'].append({'index':i,'pn_root_id':str(int(root_id)),'kc_edge_count':int(fanout[i]),'coordinate':i if i<128 else i-128 if i<256 else None,'sign':'positive' if i<128 else 'negative' if i<256 else 'unused'})
    for mode in MODES:
        b=PortBridge.load(OUT/f'ports-{mode}.safetensors',mode)
        result['mappings'][mode]={'artifact_sha256':sha256(OUT/f'ports-{mode}.safetensors'),'rank':int(np.linalg.matrix_rank(b.projection)),
                                 'active_coordinate_indices':np.flatnonzero(np.any(b.projection!=0,axis=0)).tolist(),
                                 'formula':'z = normalize((embedding768 - train_mean768) @ projection768x128); ports[j]=max(z[j],0), ports[j+128]=max(-z[j],0)'}
    (OUT/'wiring.json').write_text(json.dumps(result,indent=2)+'\n');print('Exported 319 PN identities and 3 calibrated mappings')
if __name__=='__main__':main()
