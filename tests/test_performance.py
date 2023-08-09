import pytest
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pyldl.mapping as pm
import pyldl.performance as lper
import xarray as xr

TEST_ROOT = Path(__file__).parent
RESOURCES = TEST_ROOT / 'resources'

infl = pd.DataFrame({'word'  :['walk', 'walk', 'walks', 'walked', 'walked', 'walked'],
                     'lemma' :['walk', 'walk', 'walk' , 'walk'  , 'walk'  , 'walk'  ],
                     'person':['1'   , '2'   , '3'    , '1'     , '2'     , '3'     ],
                     'tense' :['pres', 'pres', 'pres' , 'past'  , 'past'  , 'past'  ]})
cmat = pm.gen_cmat(infl.word, cores=1, differentiate_duplicates=True)
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
dist = [True, False]
pars = [ (i,j) for i in mats for j in dist ]
pars = [ tuple(expand_to_mats(i[0]) + [i[1]]) for i in pars ]
@pytest.mark.parametrize('hat, mat, dist', pars)
def test_accuracy (hat, mat, dist):
    assert lper.accuracy(hat, mat, dist) == 0.5


mats = ['c', 's']
gues = [1, 2]
dist = [True, False]
pars = [ [i,j,k] for i in mats for j in gues for k in dist ]
pars = [ tuple(expand_to_mats(i[0]) + i[1:]) for i in pars ]
pars = [ [i,*j] for i,j in enumerate(pars) ]
@pytest.mark.parametrize('ind, hat, mat, gues, dist', pars)
def test_accuracy (ind, hat, mat, gues, dist):
    pred = lper.predict_df(hat, mat, gues, dist)
    _prd = '{}/predict_df_{:02d}.csv'.format(RESOURCES, ind)
    _prd = pd.read_csv(_prd, sep='\t', header=0)
    assert pred.equals(_prd)


wrds = ['walk0', 'walks']
mats = ['c', 's']
dist = [True, False]
pars = [ [i,j,k] for i in wrds for j in mats for k in dist ]
pars = [ [i[0]] + expand_to_mats(i[1]) + i[2:] for i in pars ]
pars = [ [i,*j] for i,j in enumerate(pars) ]
@pytest.mark.parametrize('ind, wrd, hat, mat, dist', pars)
def test_predict (ind, wrd, hat, mat, dist):
    pred = lper.predict(wrd, hat, mat, dist)
    _prd = '{}/predict_{:02d}.csv'.format(RESOURCES, ind)
    _prd = pd.read_csv(_prd, sep='\t', header=None).squeeze('columns')
    assert pred.equals(_prd)
