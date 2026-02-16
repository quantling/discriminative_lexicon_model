import pytest
import numpy as np
import os
import pandas as pd
from pathlib import Path
import discriminative_lexicon_model.mapping as pm
import xarray as xr

# TEST_ROOT = Path('.')
TEST_ROOT = Path(__file__).parent
RESOURCES = TEST_ROOT / 'resources'

infl = pd.DataFrame({'word'  :['walk','walk','walks','walked'],
                     'lemma' :['walk','walk','walk' ,'walk'  ],
                     'person':['1'   ,'2'   ,'1/2/3','3'     ],
                     'tense': ['pres','pres','pres' ,'past'  ]})

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



grams  = [2,3]
counts = [True, False]
noises = [0, 0.1]
pars = [ (i,j,k) for i in grams for j in counts for k in noises ]
@pytest.mark.parametrize('gram, count, noise', pars)
def test_gen_cmat (gram, count, noise):
    cmat0_shape = (2,7) if gram==2 else (2,8)
    cmat0_dims = ('word', 'cues')
    cmat0_word_vals = ['banana', 'aaaa']
    cmat0_cues_vals = ['#b', 'ba', 'an', 'na', 'a#', '#a', 'aa'] if gram==2 else ['#ba', 'ban', 'ana', 'nan', 'na#', '#aa', 'aaa', 'aa#']
    d = {(2, True): [1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 3],
         (2, False): [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
         (3, True): [1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1],
         (3, False): [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]}
    cmat0_vals = np.array(d[(gram, count)]).reshape(cmat0_shape)
    if noise:
        rand = np.random.default_rng(100).normal(scale=0.1, size=cmat0_shape)
        cmat0_vals = cmat0_vals + rand
    cmat0 = xr.DataArray(np.array(cmat0_vals).reshape(cmat0_shape), dims=cmat0_dims, coords={'word':cmat0_word_vals, 'cues':cmat0_cues_vals})
    cmat_test = pm.gen_cmat(words=['banana', 'aaaa'], gram=gram, count=count, noise=noise, randseed=100)
    assert cmat_test.identical(cmat0)


def test_gen_cmat_cues_same_words ():
    """Passing cues derived from the same words should produce an identical result."""
    words = ['banana', 'aaaa']
    cues = pm.to_cues(words, gram=2)
    cmat_auto = pm.gen_cmat(words, gram=2)
    cmat_cues = pm.gen_cmat(words, gram=2, cues=cues)
    assert cmat_cues.identical(cmat_auto)


def test_gen_cmat_cues_shared_dimensions ():
    """Two word sets with the same pre-defined cues should have identical cue dimensions."""
    all_words = ['banana', 'aaaa', 'ban']
    cues = pm.to_cues(all_words, gram=2)
    cmat_a = pm.gen_cmat(['banana', 'aaaa'], gram=2, cues=cues)
    cmat_b = pm.gen_cmat(['ban'], gram=2, cues=cues)
    assert list(cmat_a.cues.values) == list(cmat_b.cues.values)


def test_gen_cmat_cues_superset ():
    """Cues from a larger word set produce zero columns for absent n-grams."""
    cues = pm.to_cues(['banana', 'aaaa', 'ban'], gram=2)
    cmat = pm.gen_cmat(['ban'], gram=2, cues=cues)
    assert cmat.shape == (1, len(cues))
    # 'aa' is a cue from 'aaaa' but never appears in 'ban'
    assert cmat.sel(word='ban', cues='aa').item() == 0


def test_gen_cmat_cues_subset ():
    """Passing a subset of cues limits the columns accordingly."""
    cues_subset = ['#b', 'ba', 'an']
    cmat = pm.gen_cmat(['banana'], gram=2, cues=cues_subset)
    assert list(cmat.cues.values) == cues_subset
    assert cmat.shape == (1, 3)


def test_df_to_smat ():
    words = ['ban','banban','banbanban']
    df = pd.DataFrame({'ban':[1, 1, 1], 'intensity':[1, 2, 3]}, index=words)
    smat_test = pm.gen_smat_from_df(df)
    smat0_shape = (3,2)
    smat0_values = np.array([1,1,1,2,1,3]).reshape(smat0_shape)
    smat0_dims = ('word','semantics')
    smat0 = xr.DataArray(smat0_values, dims=smat0_dims, coords={'word':words, 'semantics':['ban','intensity']})
    assert smat0.identical(smat_test)


frms = [None, 'word', 'lemma']
seps = [None, '/']
dims = [3, 5]
mns  = [0, 100]
sds  = [1, 100]
incl = [True, False]
difs = [True, False]
seds = [10]
pars_gen_smat_sim = [ (i,j,k,l,m,n,o,p) for i in frms for j in seps for k in dims for l in mns for m in sds for n in incl for o in difs for p in seds ]
pars_gen_smat_sim = pars_gen_smat_sim + [('word', '/', 5, 0, 1, True, True, None)]
pars_gen_smat_sim = [ (i,*j) for i,j in enumerate(pars_gen_smat_sim) ]
@pytest.mark.parametrize('ind, form, sep, dim_size, mn, sd, incl, diff, seed', pars_gen_smat_sim)
def test_gen_smat_sim (ind, form, sep, dim_size, mn, sd, incl, diff, seed):
    if (form is None) and (not incl):
        with pytest.raises(ValueError) as e_info:
            smat = pm.gen_smat_sim(infl, form, sep, dim_size, mn, sd, incl, diff, seed)
            assert e_info == 'Specify which column to drop by the argument "form" when "include_form" is False.'
    else:
        _smat = '{}/smat_sim_{:03d}.nc'.format(RESOURCES, ind)
        _smat = xr.open_dataarray(_smat)
        smat = pm.gen_smat_sim(infl, form, sep, dim_size, mn, sd, incl, diff, seed)
        if seed is None:
            assert not smat.round(10).identical(_smat.round(10))
        else:
            xr.testing.assert_allclose(smat, _smat)


def test_gen_fmat():
    cmat = pm.gen_cmat(infl.word, gram=3, count=False, noise=0)
    smat = pm.gen_smat_sim(infl, form='word', sep='/', dim_size=5, seed=10)
    fmat = pm.gen_fmat(cmat, smat)
    _fmat = '{}/fmat.nc'.format(RESOURCES)
    _fmat = xr.open_dataarray(_fmat)
    xr.testing.assert_allclose(fmat, _fmat)

def test_gen_gmat():
    cmat = pm.gen_cmat(infl.word, gram=3, count=False, noise=0)
    smat = pm.gen_smat_sim(infl, form='word', sep='/', dim_size=5, seed=10)
    gmat = pm.gen_gmat(cmat, smat)
    _gmat = '{}/gmat.nc'.format(RESOURCES)
    _gmat = xr.open_dataarray(_gmat)
    xr.testing.assert_allclose(gmat, _gmat)

cmat = pm.gen_cmat(infl.word, gram=3, count=False, noise=0)
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
        xr.testing.assert_allclose(shat, _shat)


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
        xr.testing.assert_allclose(chat, _chat)

def test_update_weight_matrix ():
    w = np.zeros((4,2))
    c = [1, 1, 0, 0]
    o = [1, 0]
    l = 0.1
    w = pm.update_weight_matrix(w, c, o, l, nlms=False)
    g = np.array([0.1, 0, 0.1, 0, 0, 0, 0, 0]).reshape(4,2)
    ok0 = np.allclose(w, g, atol=1e-18)
    c = [1, 0, 1, 1]
    o = [0, 1]
    w = pm.update_weight_matrix(w, c, o, l, nlms=False)
    g = np.array([0.09, 0.1, 0.1, 0, -0.01, 0.1, -0.01, 0.1]).reshape(4,2)
    ok1 = np.allclose(w, g, atol=1e-18)
    assert ok0 and ok1

def test_incremental_learning01 ():
    cmat = pm.gen_cmat(['a','an'], gram=2)
    smat = pm.gen_mmat(pd.DataFrame({'Word':['a','an']}))
    fmat = pm.incremental_learning(['a', 'an'], cmat, smat, nlms=False)
    gold = np.array([0.09, 0.1, 0.1, 0, -0.01, 0.1, -0.01, 0.1]).reshape((4,2))
    gold = xr.DataArray(gold, dims=('cues', 'feature'), coords={'cues':['#a','a#','an','n#'], 'feature':['Word:a','Word:an']})
    assert fmat.round(15).identical(gold.round(15))

def test_incremental_learning02 ():
    cmat = pm.gen_cmat(['a','an'], gram=2)
    smat = xr.DataArray([[0.7, -0.2, 0.1], [0,0,0]], dims=('word','feature'), coords={'word':['a','an'], 'feature':['Word:a','Word:an','X']})
    fmat = pm.incremental_learning(['a'], cmat, smat, nlms=False)
    gold = np.array([0.07, -0.02, 0.01]*2 + [0, 0, 0]*2).reshape(4,3)
    gold = xr.DataArray(gold, dims=('cues', 'feature'), coords={'cues':['#a','a#','an','n#'], 'feature':['Word:a','Word:an','X']})
    assert fmat.round(15).identical(gold.round(15))

def test_incremental_learning03 ():
    cmat  = pm.gen_cmat(['a','an'], gram=2)
    smat  = pm.gen_mmat(pd.DataFrame({'Word':['a','an']}))
    fmats = pm.incremental_learning(['a', 'an'], cmat, smat, return_intermediate_weights=True, nlms=False)
    gold0 = np.array([0.00, 0.00, 0.00, 0.00,  0.00, 0.00,  0.00, 0.00]).reshape((4,2))
    gold1 = np.array([0.10, 0.00, 0.10, 0.00,  0.00, 0.00,  0.00, 0.00]).reshape((4,2))
    gold2 = np.array([0.09, 0.10, 0.10, 0.00, -0.01, 0.10, -0.01, 0.10]).reshape((4,2))
    gold0 = xr.DataArray(gold0, dims=('cues', 'feature'), coords={'cues':['#a','a#','an','n#'], 'feature':['Word:a','Word:an']})
    gold1 = xr.DataArray(gold1, dims=('cues', 'feature'), coords={'cues':['#a','a#','an','n#'], 'feature':['Word:a','Word:an']})
    gold2 = xr.DataArray(gold2, dims=('cues', 'feature'), coords={'cues':['#a','a#','an','n#'], 'feature':['Word:a','Word:an']})
    ok0   = fmats[0].round(15).identical(gold0)
    ok1   = fmats[1].round(15).identical(gold1)
    ok2   = fmats[2].round(15).identical(gold2)
    assert (ok0 and ok1) and ok2

def test_incremental_learning_byind01 ():
    cmat  = pm.gen_cmat(['a','an'], gram=2)
    smat  = pm.gen_mmat(pd.DataFrame({'Word':['a','an']}))
    events = [0, 1, 1]
    fmat_test = pm.incremental_learning_byind(events, cmat, smat)
    fmat0_values = [0.04683333310916667, 0.06333333313333334, 0.04999999975, 0.0, -0.0031666666408333337, 0.06333333313333334, -0.0031666666408333337, 0.06333333313333334]
    fmat0_shape = (4, 2)
    fmat0_dims = ('cues', 'feature')
    fmat0_cues_values = ['#a', 'a#', 'an', 'n#']
    fmat0_feature_values = ['Word:a', 'Word:an']
    fmat0 = xr.DataArray(np.array(fmat0_values).reshape(fmat0_shape), dims=fmat0_dims, coords={'cues':fmat0_cues_values, 'feature':fmat0_feature_values})
    xr.testing.assert_allclose(fmat_test, fmat0)

def test_incremental_learning_byind02 ():
    cmat  = pm.gen_cmat(['a','an'], gram=2)
    smat  = pm.gen_mmat(pd.DataFrame({'Word':['a','an']}))
    events_byind = [0, 1, 1]
    events       = ['a', 'an', 'an']
    fmat_byind_test = pm.incremental_learning_byind(events_byind, cmat, smat)
    fmat_test       = pm.incremental_learning(events, cmat, smat)
    assert fmat_test.identical(fmat_byind_test)

def test_weight_by_freq ():
    words  = ['as','bs','bd']
    freqs  = [298, 1, 1]
    cmat   = pm.gen_cmat(words, gram=2)
    cmat_f = pm.weight_by_freq(cmat, freqs)
    freqs  = np.array(freqs)
    freqs  = freqs / freqs.max()
    freqs  = np.sqrt(freqs)
    freqs  = np.diag(freqs)
    gold   = xr.DataArray(np.matmul(freqs, cmat.values), dims=cmat.dims, coords=cmat.coords)
    assert gold.identical(cmat_f)

def test_gen_vmat ():
    words = ['abx', 'aby']
    vmat_test = pm.gen_vmat(words)

    vmat0_shape = (6,5)
    vmat0_values = [False, True, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False]
    vmat0_values = np.array(vmat0_values).reshape(vmat0_shape)
    vmat0_dims = ('current','next')
    vmat0_current_values = ['#ab', 'abx', 'bx#', 'aby', 'by#', '']
    vmat0_next_values = ['#ab', 'abx', 'bx#', 'aby', 'by#']
    vmat0 = xr.DataArray(vmat0_values, dims=vmat0_dims, coords={'current':vmat0_current_values, 'next':vmat0_next_values})
    assert vmat0.identical(vmat_test)

