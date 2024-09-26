import pytest
import numpy as np
import os
import pandas as pd
from pathlib import Path
import discriminative_lexicon_model.mapping as pm
import discriminative_lexicon_model.performance as lper
import xarray as xr

TEST_ROOT = Path(__file__).parent
RESOURCES = TEST_ROOT / 'resources'

infl = pd.DataFrame({'word'  :['walk', 'walk', 'walks', 'walked', 'walked', 'walked'],
                     'lemma' :['walk', 'walk', 'walk' , 'walk'  , 'walk'  , 'walk'  ],
                     'person':['1'   , '2'   , '3'    , '1'     , '2'     , '3'     ],
                     'tense' :['pres', 'pres', 'pres' , 'past'  , 'past'  , 'past'  ]})
cmat = pm.gen_cmat(infl.word, gram=3, count=False, noise=0, randseed=None, differentiate_duplicates=True)
smat = pm.gen_smat_sim(infl, form='word', sep='/', dim_size=100, seed=10, differentiate_duplicates=True)
chat = pm.gen_chat(smat=smat, cmat=cmat)
shat = pm.gen_shat(cmat=cmat, smat=smat)

def expand_to_mats (x):
    if x=='c':
        ret = [chat, cmat]
    elif x=='s':
        ret = [shat, smat]
    else:
        raise ValueError('Invalid input.')
    return ret

mats = ['c', 's']
pars = [ tuple(expand_to_mats(i[0])) for i in mats ]
@pytest.mark.parametrize('hat, mat', pars)
def test_accuracy (hat, mat):
    assert lper.accuracy(pred=hat, gold=mat) == 0.5


mats = ['c', 's']
gues = [1, 2]
pars = [ [i,j] for i in mats for j in gues ]
pars = [ tuple(expand_to_mats(i[0]) + i[1:]) for i in pars ]
pars = [ [i,*j] for i,j in enumerate(pars) ]
@pytest.mark.parametrize('ind, hat, mat, gues', pars)
def test_accuracy_df (ind, hat, mat, gues):
    pred = lper.predict_df(pred=hat, gold=mat, n=gues)
    _prd = '{}/predict_df_{:02d}.csv'.format(RESOURCES, ind)
    _prd = pd.read_csv(_prd, sep='\t', header=0)
    assert pred.equals(_prd)


