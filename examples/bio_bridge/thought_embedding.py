"""Read simulated neural features through a frozen text-embedding vocabulary.

This reconstructs engineered input semantics, not subjective fly thoughts.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from novigrad import Engine
from safetensors.numpy import load_file,save_file
from inhibition_mechanism import ShadowEngine,calibrate,checkpoint,sha256
from delayed_credit import ROOT,semantic_ports

OUT=ROOT/'results/thought-bridge'
GOALS=['water','food','warmth','rest']


def unit(rows):
    x=np.ascontiguousarray(rows,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('finite matrix required')
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def fit_alignment(hidden,embeddings,ridge=.1):
    x,y=unit(hidden),unit(embeddings)
    if len(x)!=len(y) or len(x)==0 or not np.isfinite(ridge) or ridge<=0:
        raise ValueError('matched nonempty rows and positive ridge required')
    return x,np.linalg.solve(x@x.T+ridge*np.eye(len(x)),y)


def decode(basis,coefficients,hidden):
    return unit(unit(hidden)@basis.T@coefficients)


def candidate_data():
    descriptions=json.loads(Path(__file__).with_name('thought_descriptions.json').read_text())['descriptions']
    data=load_file(str(OUT/'description-embeddings.safetensors'))
    embeddings=data['embeddings']
    return descriptions,unit(embeddings[:,:128])


def retrieve(vectors,descriptions,description_vectors):
    scores=unit(vectors)@description_vectors.T
    order=np.argsort(-scores,axis=1,kind='stable')
    result=[{'category':descriptions[row[0]]['category'],'description_id':descriptions[row[0]]['id'],
             'top3':[{'description':descriptions[j]['text'],'category':descriptions[j]['category'],
                      'cosine':float(score[j])} for j in row[:3]],
             'top2_margin':float(score[row[0]]-score[row[1]]),
             'interpretation':'nearest authored semantic candidate, not a confidence or a biological thought'}
            for row,score in zip(order,scores)]
    for i,norm in enumerate(np.linalg.norm(vectors,axis=1)):
        if norm<=1e-12:
            result[i]={'category':None,'description_id':None,'top3':[], 'top2_margin':0.,
                       'interpretation':'no nonzero decoded signal; no semantic assertion'}
    return result


def load_dataset():
    old=ROOT/'results/gemma-bridge';rows=json.loads((old/'dataset.json').read_text())
    vectors=load_file(str(old/'embeddings.safetensors'))['embeddings']
    datasets={}
    for split in ['train','test','korean']:
        ix=np.array([r['split']==split for r in rows]);datasets[split]=(vectors[ix],np.array([r['label'] for r in rows])[ix],[r['text'] for r in rows if r['split']==split])
    cases=json.loads(Path(__file__).with_name('sparsity_confirmation.json').read_text())
    datasets['reused_confirmation']=(load_file(str(ROOT/'results/bio-bridge/confirmation-features.safetensors'))['embeddings'],
        np.array([GOALS.index(r['class']) for r in cases]),[r['text'] for r in cases])
    return datasets


def evaluation(predicted,actual,labels,texts,descriptions,description_vectors):
    decoded=retrieve(predicted,descriptions,description_vectors)
    catalog_scores=unit(predicted)@unit(actual).T
    return {'semantic_category_accuracy':float(np.mean([r['category']==GOALS[y] for r,y in zip(decoded,labels)])),
            'mean_cosine_to_input_embedding':float(np.mean(np.sum(unit(predicted)*unit(actual),axis=1))),
            'catalog_identity_retrieval_accuracy':float(np.mean(catalog_scores.argmax(1)==np.arange(len(actual)))),
            'catalog_boundary':'posthoc diagnostic: recover matching row from this known input catalog, not free-text reconstruction',
            'unsupported_candidate_rate':float(np.mean([r['category'] not in GOALS for r in decoded])),
            'cases':[{'text':text,'expected_category':GOALS[y],**row} for text,y,row in zip(texts,labels,decoded)]}


def hidden_for(shadow,vectors):
    rates=semantic_ports(vectors)
    raw=shadow.raw_hidden(rates)
    hidden=shadow.inhibit(raw,'native_topk_2',{})
    return hidden


def mbon_for(shadow,vectors):
    return np.asarray(shadow.plastic_matrix@hidden_for(shadow,vectors).T).T


def run(baseline=False):
    start=time.perf_counter();data=load_dataset();descriptions,description_vectors=candidate_data()
    shadow=ShadowEngine(checkpoint(601));training,_,_=data['train']
    target=unit(training[:,:128]);train_hidden=hidden_for(shadow,training)
    mean=np.mean(target,axis=0,keepdims=True)
    report={'design':{'task':'decode engineered semantic input from frozen KC rate activity into authored text candidates',
        'not_fly_thought':'no biological neural recordings, subjective reports, or autonomous internal state',
        'training':'32 paired language embeddings and simulated KC vectors; no test alignment fitting',
        'dimensions':128,'ridge':.1,'model':'frozen original EmbeddingGemma-300m; documented128MRL truncate+normalize',
        'candidates':'32 bilingual need descriptions plus8 unsupported-state distractors, authored before embedding/evaluation',
        'readout':'candidatecosine scores are similarity, not confidence; no calibrated rejection',
        'shared_encoder':'PN-KC layers precede plastic readout; decoder success need not reflect learned policy or causal action use',
        'settings':'alignment and candidate settings fixed before measurement; untrained and catalog-identity controls added after initial scores; reused evaluation sets'},'variants':{}}
    fitted={}
    if not baseline:
        fitted['aligned']=fit_alignment(train_hidden,target)
        for seed in [11,12,13,14,15]:
            permutation=np.random.default_rng(seed).permutation(len(target))
            fitted[f'shuffled_{seed}']=fit_alignment(train_hidden,target[permutation])
        fitted['mbon_aligned']=fit_alignment(mbon_for(shadow,training),target)
        for name,(basis,coefficients) in fitted.items():
            path=OUT/f'semantic-decoder-{name}.safetensors'
            save_file({'train_hidden_basis':basis,'dual_coefficients':coefficients},str(path),
                metadata={'format':'novigrad.semantic_alignment','version':'1','ridge':'.1','target_dimensions':'128','input_semantics':'L2 normalized '+('MBON' if name=='mbon_aligned' else 'KC')+' activity','interpretation':'engineered semantic reconstruction, not fly thoughts'})
            probe_rows=mbon_for(shadow,training) if name=='mbon_aligned' else train_hidden
            restored=load_file(str(path));np.testing.assert_array_equal(decode(basis,coefficients,probe_rows),decode(restored['train_hidden_basis'],restored['dual_coefficients'],probe_rows))
        report['checkpoints']={name:{'path':str((OUT/f'semantic-decoder-{name}.safetensors').relative_to(ROOT)),
            'sha256':sha256(OUT/f'semantic-decoder-{name}.safetensors'),'roundtrip_exact':True} for name in fitted}
    variants=['mean_embedding'] if baseline else ['raw_embedding','mean_embedding',*fitted]
    for name in variants:
        report['variants'][name]={}
        for split,(vectors,labels,texts) in data.items():
            actual=unit(vectors[:,:128])
            if name=='raw_embedding':predicted=actual
            elif name=='mean_embedding':predicted=np.repeat(mean,len(vectors),axis=0)
            else:predicted=decode(*fitted[name],mbon_for(shadow,vectors) if name=='mbon_aligned' else hidden_for(shadow,vectors))
            report['variants'][name][split]=evaluation(predicted,actual,labels,texts,descriptions,description_vectors)
    # Frozen input encoding is identical across trained policies, despite plastic readouts differing.
    reference=hidden_for(shadow,data['reused_confirmation'][0])
    report['hidden_max_difference_across_policy_seeds']={str(s):float(np.max(np.abs(hidden_for(ShadowEngine(checkpoint(s)),data['reused_confirmation'][0])-reference))) for s in [602,603]}
    if not baseline:
        untrained=Engine.from_edges(ROOT/'data/pn_kc.tsv',ROOT/'data/kc_mbon.tsv',actions=4,
            learning_rate=.3,logit_gain=1.,active_fraction=.02,homeostasis=False,readout='opponent')
        path=OUT/'untrained-native.safetensors';untrained.save(path,overwrite=True)
        untrained_shadow=ShadowEngine(path);vectors,labels,texts=data['reused_confirmation']
        initial_hidden=hidden_for(untrained_shadow,vectors)
        report['untrained_control']={'status':'posthoc after initial semantic decoding; no tuning',
            'hidden_max_difference':float(np.max(np.abs(initial_hidden-reference))),
            'native_untrained_accuracy':float(np.mean(untrained_shadow.probabilities(initial_hidden).argmax(1)==labels)),
            'native_trained_accuracy':float(np.mean(shadow.probabilities(reference).argmax(1)==labels)),
            'semantic_decode_max_difference':float(np.max(np.abs(decode(*fitted['aligned'],initial_hidden)-decode(*fitted['aligned'],reference)))),
            'interpretation':'PN-KC representation is fixed before plastic learning; semantic decodability is not evidence for learned internal thought',
            'checkpoint':{'path':str(path.relative_to(ROOT)),'sha256':sha256(path)}}
        initial_mbon=mbon_for(untrained_shadow,vectors)
        mbon_decoded=decode(*fitted['mbon_aligned'],initial_mbon)
        report['posthoc_mbon_control']={'status':'added after KC readout results, fixed same ridge0.1; no hyperparameter tuning',
            'stage':'96 MBON-like rates after plastic KC-MBON weights, before output gains/action aggregation',
            'policy_seed':601,'trained_alignment_applied_to_untrained_mbon':evaluation(mbon_decoded,unit(vectors[:,:128]),labels,texts,descriptions,description_vectors),
            'max_decoded_embedding_change':float(np.max(np.abs(mbon_decoded-decode(*fitted['mbon_aligned'],mbon_for(shadow,vectors))))),
            'boundary':'still paired to input semantics, not trained on independent reward expectations or actual neural recordings'}
    paths=[Path(__file__).resolve(),Path(__file__).with_name('thought_descriptions.json'),OUT/'description-embeddings.safetensors',
        ROOT/'results/gemma-bridge/dataset.json',ROOT/'results/gemma-bridge/embeddings.safetensors',
        ROOT/'examples/bio_bridge/sparsity_confirmation.json',ROOT/'results/bio-bridge/confirmation-features.safetensors',*[checkpoint(s) for s in [601,602,603]]]
    report['provenance_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    report['runtime_seconds']=time.perf_counter()-start
    name='decoding-baseline' if baseline else 'decoding'
    (OUT/f'{name}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({name:{k:{s:r['semantic_category_accuracy'] for s,r in v.items()} for k,v in report['variants'].items()}}))


def demo(text,stage='kc'):
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer('/Users/a405394/models/google_embeddinggemma-300m',device='mps',local_files_only=True)
    vector=model.encode([text],prompt_name='Classification',normalize_embeddings=True)
    shadow=ShadowEngine(checkpoint(601));hidden=hidden_for(shadow,vector)
    saved=load_file(str(OUT/('semantic-decoder-mbon_aligned.safetensors' if stage=='mbon' else 'semantic-decoder-aligned.safetensors')))
    neural=mbon_for(shadow,vector) if stage=='mbon' else hidden
    semantic=decode(saved['train_hidden_basis'],saved['dual_coefficients'],neural)
    descriptions,vectors=candidate_data();answer=retrieve(semantic,descriptions,vectors)[0]
    answer['input']=text;answer['native_action_probabilities']=dict(zip(GOALS,map(float,shadow.probabilities(hidden)[0])))
    answer['readout_stage']=stage
    answer['boundary']='Readout of a text-driven simulated circuit, not a biological fly recording.'
    print(json.dumps(answer,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');p.add_argument('--text');p.add_argument('--stage',choices=['kc','mbon'],default='kc');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.text is not None:demo(a.text,a.stage)
    else:run(a.baseline)


if __name__=='__main__':main()
