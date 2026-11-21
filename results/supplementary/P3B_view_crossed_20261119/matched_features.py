"""Frozen, auditable adapters. No archived feature code is edited."""
import inspect, sys
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO/'src'))
sys.path.insert(0,str(ROOT.parent/'P3B_pilot_20261119'))
import evaluate_response_graph_features as original_graph
from phaseguard_features import build_features,entropy

# Copy the exact archived function; change only the chosen answer lookup.
source=inspect.getsource(original_graph.graph_table)
assert source.count('selected = int(base_answers[reference_index])')==1
source=source.replace('selected = int(base_answers[reference_index])',
    'selected = int(answer_targets.get((dataset, qid, reference_model), base_answers[reference_index]))')

def graph_table(metadata,values,models,targets=None):
    scope=dict(vars(original_graph));scope['answer_targets']=targets or {}
    exec(compile(source,str(__file__),'exec'),scope)
    return scope['graph_table'](metadata,values,models)

def majority(answers):
    counts=Counter(answers)
    return max(answers,key=lambda a:counts[a])

def pg_features(question,options,answers,probabilities,target,peers):
    vote,confidence,features=build_features(question,options,answers,probabilities)
    labels=list('ABCDE'[:len(options)])
    p=np.asarray(probabilities)
    values=p[:,labels.index(target)]
    for stat,value in [('mean',values.mean()),('min',values.min()),('max',values.max()),('range',np.ptp(values)),('variance',values.var())]:
        features['voted_probability_'+stat]=float(value)
    single=dict(features)
    peer_votes=[];phase_agreement=[]
    for name,peer_answers,peer_probs in peers:
        q=np.asarray(peer_probs);v=majority(peer_answers);peer_votes.append(v)
        hist=np.array([peer_answers.count(a)/4 for a in labels])
        phase_agreement.extend(a==b for a,b in zip(answers,peer_answers))
        for field,value in [('vote_consensus',hist.max()),('vote_entropy',entropy(hist)),
            ('mean_max_probability',q.max(axis=1).mean()),('mean_own_vote_probability',q[:,labels.index(v)].mean()),
            ('mean_target_probability',q[:,labels.index(target)].mean()),('majority_matches_target',v==target)]:
            features[name+'_'+field]=float(value)
    votes=[vote]+peer_votes
    features.update(panel_vote_consensus=max(Counter(votes).values())/3,
        peer_phase_agreement=float(np.mean(phase_agreement)),
        target_matches_panel_majority=float(target==majority(votes)),
        peer_majorities_agree=float(peer_votes[0]==peer_votes[1]))
    assert len(single)==32 and len(features)==48 and np.isfinite(list(features.values())).all()
    return single,features
