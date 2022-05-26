import pytest
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pyldl.mapping as pm
import pyldl.performance as lper
import xarray as xr

# TEST_ROOT = Path(__file__).parent
# RESOURCES = TEST_ROOT / 'resources'

infl = pd.DataFrame({'word'  :['walk', 'walked', 'talks'],
                     'lemma' :['walk', 'walk'  , 'talk' ],
                     'person':['1/2' , '1/2/3' , '1'    ],
                     'tense' :['pres', 'past'  , 'pres' ]})
cmat = pm.gen_cmat(infl.word, cores=1)
smat = pm.gen_smat_sim(infl, form='word', sep='/', dim_size=100, seed=10)
chat = pm.gen_chat(smat=smat, cmat=cmat)
shat = pm.gen_shat(cmat=cmat, smat=smat)
print(lper.predict_df(chat,cmat))
print(lper.predict_df(shat,smat))



infl = pd.read_csv('/home/motoki/latin.csv', sep='\t')
cmat = pm.gen_cmat(infl.Word, cores=1)
smat = pm.gen_smat_sim(infl, form='Word', sep='/', dim_size=len(cmat.cues), seed=10)
chat = pm.gen_chat(smat=smat, cmat=cmat)
shat = pm.gen_shat(cmat=cmat, smat=smat)
mmat = pm.gen_mmat(infl, form='Word')
jmat = pm.gen_jmat(mmat, dim_size=len(cmat.cues), seed=10)
print(lper.predict_df(chat,cmat))
print(lper.predict_df(shat,smat))



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
# @pytest.mark.parametrize('hat, mat, dist', pars)
# def test_accuracy (hat, mat, dist):
#     lper.accuracy(hat, 

