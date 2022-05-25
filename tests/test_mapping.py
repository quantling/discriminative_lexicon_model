import pytest
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pyldl.mapping as pm
import xarray as xr

TEST_ROOT = Path(__file__).parent
RESOURCES = TEST_ROOT / 'resources'

infl = pd.DataFrame({'word':['walk','walked','walks'], 'lemma':['walk','walk','walk'], 'person':['1/2','1/2/3','3'], 'tense':['pres','past','pres']})

pars_to_ngram = [
    (1,True,True,  ['#','a','b']),
    (1,True,False, ['#','a','b']),
    (1,False,True, ['#','a','b','a','b','a','#']),
    (1,False,False,['#','#','a','a','a','b','b']),
    (2,True,True,  ['#a','ab','ba','a#']),
    (2,True,False, ['#a','a#','ab','ba']),
    (2,False,True, ['#a','ab','ba','ab','ba','a#']),
    (2,False,False,['#a','a#','ab','ab','ba','ba']),
    (3,True,True,  ['#ab','aba','bab','ba#']),
    (3,True,False, ['#ab','aba','ba#','bab']),
    (3,False,True, ['#ab','aba','bab','aba','ba#']),
    (3,False,False,['#ab','aba','aba','ba#','bab'])]
@pytest.mark.parametrize('gram, unique, keep_order, result', pars_to_ngram)
def test_to_ngram (gram, unique, keep_order, result):
    assert pm.to_ngram('ababa', gram=gram, unique=unique, keep_order=keep_order) == result

def test_gen_cmat ():
    cmat = pm.gen_cmat(infl.word, cores=1)
    _cmat = np.array([True, True, False, False, False, True, False, False, True, True, True, True, True, False, False, True, False, True, True, True, False, False, True, False, False, True, True]).reshape(3,9)
    coords ={'word':infl.word.tolist(), 'cues':['#wa', 'alk', 'ed#', 'ked', 'ks#', 'lk#', 'lke', 'lks', 'wal']}
    _cmat = xr.DataArray(_cmat, coords=coords)
    assert cmat.identical(_cmat)

frms = [None, 'word', 'lemma']
seps = [None, '/']
dims = [3, 5]
seds = [10]
pars_gen_smat_sim = [ (i,j,k,l) for i in frms for j in seps for k in dims for l in seds ]
pars_gen_smat_sim = [ (i,j[0],j[1],j[2],j[3]) for i,j in enumerate(pars_gen_smat_sim) ]
pars_gen_smat_sim = pars_gen_smat_sim + [(12, 'word', '/', 5, None)]
@pytest.mark.parametrize('ind, form, sep, dim_size, seed', pars_gen_smat_sim)
def test_gen_smat_sim (ind, form, sep, dim_size, seed):
    _smat = '{}/smat_sim_{:02d}.nc'.format(RESOURCES, ind)
    _smat = xr.open_dataarray(_smat)
    smat = pm.gen_smat_sim(infl, form, sep, dim_size, seed)
    assert smat.identical(_smat) if ind!=12 else not smat.identical(_smat)

def test_gen_fmat():
    cmat = pm.gen_cmat(infl.word, cores=1)
    smat = pm.gen_smat_sim(infl, form='word', sep='/', dim_size=5, seed=10)
    fmat = pm.gen_fmat(cmat, smat)
    _fmat = '{}/fmat.nc'.format(RESOURCES)
    _fmat = xr.open_dataarray(_fmat)
    assert fmat.identical(_fmat)

def test_gen_gmat():
    cmat = pm.gen_cmat(infl.word, cores=1)
    smat = pm.gen_smat_sim(infl, form='word', sep='/', dim_size=5, seed=10)
    gmat = pm.gen_gmat(cmat, smat)
    _gmat = '{}/gmat.nc'.format(RESOURCES)
    _gmat = xr.open_dataarray(_gmat)
    assert gmat.identical(_gmat)

cmat = pm.gen_cmat(infl.word, cores=1)
smat = pm.gen_smat_sim(infl, form='word', sep='/', dim_size=5, seed=10)
fmat = pm.gen_fmat(cmat, smat)
hmat = np.array(np.matmul(np.matmul(np.array(cmat),np.linalg.pinv(np.matmul(np.array(cmat).T,np.array(cmat)))),np.array(cmat).T))
hmat = xr.DataArray(hmat, coords={'word':cmat.word.values, 'wordc':cmat.word.values})
pars_gen_shat = [(1, cmat, fmat, None, None),
                 (2, cmat, None, smat, None),
                 (3, None, None, smat, hmat),
                 (4, None, fmat, smat, None)]
@pytest.mark.parametrize('ind, cmat, fmat, smat, hmat', pars_gen_shat)
def test_gen_shat (ind, cmat, fmat, smat, hmat):
    if ind==4:
        with pytest.raises(ValueError) as e_info:
            shat = pm.gen_shat(cmat, fmat, smat, hmat)
            assert e_info == '(C, F), (C, S), or (H, S) is necessary.'
    else:
        shat = pm.gen_shat(cmat, fmat, smat, hmat)
        _shat = '{}/shat.nc'.format(RESOURCES)
        _shat = xr.open_dataarray(_shat)
        if ind==3: # Rounding due to rounding errors when producing hmat.
            shat  = shat.round(10)
            _shat = _shat.round(10)
        assert shat.identical(_shat)


gmat = pm.gen_gmat(cmat, smat)
hmat = np.array(np.matmul(np.matmul(np.array(smat),np.linalg.pinv(np.matmul(np.array(smat).T,np.array(smat)))),np.array(smat).T))
hmat = xr.DataArray(hmat, coords={'word':smat.word.values, 'wordc':smat.word.values})
pars_gen_chat = [(1, smat, gmat, None, None),
                 (2, smat, None, cmat, None),
                 (3, None, None, cmat, hmat),
                 (4, None, gmat, cmat, None)]
@pytest.mark.parametrize('ind, smat, gmat, cmat, hmat', pars_gen_chat)
def test_gen_chat (ind, smat, gmat, cmat, hmat):
    if ind==4:
        with pytest.raises(ValueError) as e_info:
            chat = pm.gen_chat(smat, gmat, cmat, hmat)
            assert e_info == '(S, G), (S, C), or (H, C) is necessary.'
    else:
        chat = pm.gen_chat(smat, gmat, cmat, hmat)
        _chat = '{}/chat.nc'.format(RESOURCES)
        _chat = xr.open_dataarray(_chat)
        if ind==3: # Rounding due to rounding errors when producing hmat.
            chat  = chat.round(10)
            _chat = _chat.round(10)
        assert chat.identical(_chat)


