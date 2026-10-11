"""Matched-cohort V4 audit. Old V4 test is now diagnostic data, never a new holdout."""
import argparse, json, time, hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit
from train_models import load_structure, load_relation
from relations import tensors, build_graph, relation_targets
from experiment_v3 import setup, measurement, aggregate, write_csv
from evaluate import endpoint_quality
from pathcut_solver import PathCutConfig, solve_pathcut

def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def probability_summary(logits, target):
    logits=np.asarray(logits,dtype=float); y=np.asarray(target,dtype=float); p=expit(logits)
    bins=[]
    for lo,hi in zip(np.arange(0,1,.1),np.arange(.1,1.1,.1)):
        pick=(p>=lo)&(p<hi if hi<1 else p<=hi)
        bins.append(dict(lower=float(lo),upper=float(hi),n=int(pick.sum()),
            confidence=float(p[pick].mean()) if pick.any() else None,
            positive_frequency=float(y[pick].mean()) if pick.any() else None))
    pos=p[y>.5]
    return dict(n=len(y),positive=int(y.sum()),bce=float(np.mean(np.logaddexp(0,logits)-y*logits)),
        brier=float(np.mean((p-y)**2)),ece_10=float(sum(b['n']*abs(b['confidence']-b['positive_frequency']) for b in bins if b['n'])/len(y)),
        positive_quantiles=dict(zip(['min','q10','q25','q50','q75','q90','max'],map(float,np.quantile(pos,[0,.1,.25,.5,.75,.9,1])))),
        positive_mean=float(pos.mean()),positive_weight_mean=float(np.maximum(2*pos-1,0).mean()),
        positive_bins={f'{lo}-{hi}':int(((pos>=lo)&(pos<hi if hi<1 else pos<=1)).sum()) for lo,hi in [(0,.5),(.5,.7),(.7,.9),(.9,1)]},bins=bins)

def run(out, budget=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): raise FileExistsError(out)
    torch.set_num_threads(1);torch.manual_seed(5819)
    cfg=PathCutConfig(**json.loads(Path('results/confirmatory/selection.json').read_text())['selected'])
    ext=load_structure('models/seed11/structure.pt');net=load_relation('models/seed11/gnn.pt')
    rows=[]; pairrows=[]; endpoints=[];logits=[];targets=[];repro=[]
    for path in sorted(Path('data/test').glob('*.npz')):
        x,pr,w,g,t,z0=setup(path,ext,cfg)
        with torch.no_grad(): l=net(tensors(g));p=l.sigmoid()
        logits.extend(l.tolist());targets.extend(t.tolist())
        oracle_g=build_graph(pr.y.numpy(),x['observed'],x['road'],x['roi'])
        exact=all(np.array_equal(g[k],oracle_g[k]) for k in g)
        endpoints.append(dict(case=path.stem,exact_graph_match=exact,**endpoint_quality(g,x)))
        methods=[('v4_joint100',g,p,False,100,None),
                 ('oracle_ports_gnn_joint100',oracle_g,None,False,100,None),
                 ('gnn_structure100',g,p,True,100,None)]
        if budget:methods += [('v4_joint300',g,p,False,300,None),('gnn_structure300',g,p,True,300,None)]
        if str(x['scene_kind'])=='curved':methods += [('oracle_pure300',oracle_g,torch.from_numpy(relation_targets(oracle_g,x['road'])),True,300,torch.from_numpy(x['road']))]
        arrays={}
        for name,gg,pp,pure,steps,anchor in methods:
            if pp is None:
                with torch.no_grad():pp=net(tensors(gg)).sigmoid()
            tt=relation_targets(gg,x['road']);start=time.perf_counter()
            pred,z,s,info=solve_pathcut(pr,w,ext,gg,pp,replace(cfg,steps=steps),z0,structure_only=pure,anchor_values=anchor)
            row,image,su=measurement(path,x,pr,gg,tt,pred,s,ext,time.perf_counter()-start,name)
            rows.append(row);dump(out/f'{path.stem}_{name}.json',info)
            if name=='v4_joint100':
                old=np.load(f'results/confirmatory/test_{path.stem}_arrays.npz')
                repro.append(dict(case=path.stem,max_s_delta=float(np.max(np.abs(s.numpy()-old['pathcut_gnn_s']))),max_image_delta=float(np.max(np.abs(image-old['pathcut_gnn_image'])))))
                rs=relation_targets(g,s.numpy())>.5;ru=relation_targets(g,su)>.5
                for k,(a,b) in enumerate(g['pairs']):pairrows.append(dict(case=path.stem,pair=k,a=int(a),b=int(b),target=int(t[k]),logit=float(l[k]),p=float(p[k]),relation_positive=int(p[k]>.5),s_connected=int(rs[k]),u_connected=int(ru[k])))
            arrays[name+'_s']=s.numpy();arrays[name+'_u']=su
        np.savez_compressed(out/f'{path.stem}.npz',**arrays)
        write_csv(out/'cases.csv',rows);write_csv(out/'pairs.csv',pairrows)
        print('AUDIT',path.stem,flush=True)
    rr=[r for r in pairrows if r['target']]
    transitions={f'p{a}_s{b}_u{c}':sum(r['relation_positive']==a and r['s_connected']==b and r['u_connected']==c for r in rr) for a in [0,1] for b in [0,1] for c in [0,1]}
    macro_relation=float(np.mean([np.mean([r['relation_positive'] for r in rr if r['case']==c]) for c in sorted({r['case'] for r in rr})]))
    dump(out/'summary.json',dict(scope='V4 old test reused only for audit; not confirmatory',summary=aggregate(rows),
        probabilities=probability_summary(logits,targets),endpoints=endpoints,positive_pair_transitions=transitions,
        macro_relation_recall=macro_relation,reproduction=repro))
    print(json.dumps(dict(summary=aggregate(rows),transitions=transitions),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results_v5/audit');a=p.parse_args();run(a.out)
