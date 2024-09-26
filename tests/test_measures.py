import pytest
import numpy as np
import os
import pandas as pd
from pathlib import Path
import discriminative_lexicon_model.mapping as pm
import discriminative_lexicon_model.measures as lmea
import xarray as xr

TEST_ROOT = Path(__file__).parent
RESOURCES = TEST_ROOT / 'resources'

infl = pd.DataFrame({'word':['walk','walked','walks'], 'lemma':['walk','walk','walk'], 'person':['1/2','1/2/3','3'], 'tense':['pres','past','pres']})
cmat = pm.gen_cmat(infl.word, gram=3, count=False, noise=0)
smat = pm.gen_smat_sim(infl, form='word', sep='/', dim_size=5, seed=10)
fmat = pm.gen_fmat(cmat, smat)
chat = pm.gen_chat(smat=smat, cmat=cmat)
shat = pm.gen_shat(cmat=cmat, smat=smat)


sfx = ['ed#', '#wa', 'xxx']
wrd = ['walk', 'walked', 'xxx']
mtd = ['corr', 'mse', 'xxx']

pars_functional_load = [ (i,j,k) for i in sfx for j in wrd for k in mtd ]
pars_functional_load = [ (i,*j) for i,j in enumerate(pars_functional_load) ]
@pytest.mark.parametrize('ind, cue, word, method', pars_functional_load)
def test_functional_load (ind, cue, word, method):
    if 'xxx' in (cue, word):
        with pytest.raises(KeyError) as e_info:
            fl = lmea.functional_load(cue, fmat, word, smat, method)
            assert e_info == 'xxx'
    elif 'xxx' == method:
        with pytest.raises(ValueError) as e_info:
            fl = lmea.functional_load(cue, fmat, word, smat, method)
            assert e_info == 'method must be corr or mse.'
    else:
        fl = lmea.functional_load(cue, fmat, word, smat, method)
        _fl = ["-0.6234144595", "7.3524877073", "None", "0.8494412122", "1.6152075193", "None", "None", "None", "None", "0.9516095962", "3.3653333311", "None", "0.1372308544", "3.4029651394", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None"]
        assert '{:12.10f}'.format(fl) == _fl[ind]


pars_semantic_support = [ (i,j) for i in wrd for j in sfx ]
pars_semantic_support = [ (i,*j) for i,j in enumerate(pars_semantic_support) ]
@pytest.mark.parametrize('ind, word, cue', pars_semantic_support)
def test_semantic_support (ind, word, cue):
    if 'xxx' in (word, cue):
        with pytest.raises(KeyError) as e_info:
            sp = lmea.semantic_support(word, cue, chat)
            assert e_info == 'xxx'
    else:
        sp = lmea.semantic_support(word, cue, chat)
        _sp = [0, 1, "None", 1, 1, "None", "None", "None", "None"]
        assert round(sp,10) == _sp[ind]

pars_semantic_support_word = [ (i,j) for i,j in enumerate(wrd) ]
@pytest.mark.parametrize('ind, word', pars_semantic_support_word)
def test_semantic_support_word (ind, word):
    if 'xxx' == word:
        with pytest.raises(KeyError) as e_info:
            sp = lmea.semantic_support_word(word, chat)
            assert e_info == 'xxx'
    else:
        sp = lmea.semantic_support_word(word, chat)
        _sp = ["4.0000000000", "6.0000000000", "None"]
        assert '{:12.10f}'.format(sp) == _sp[ind]

pars_prod_acc = [ (i,j) for i in wrd for j in mtd ]
pars_prod_acc = [ (i,*j) for i,j in enumerate(pars_prod_acc) ]
@pytest.mark.parametrize('ind, word, method', pars_prod_acc)
def test_prod_acc (ind, word, method):
    if 'xxx' == word:
        with pytest.raises(KeyError) as e_info:
            pa = lmea.prod_acc(word, cmat, chat, method)
            assert e_info == 'xxx'
    elif 'xxx' == method:
        with pytest.raises(ValueError) as e_info:
            pa = lmea.prod_acc(word, cmat, chat, method)
            assert e_info == 'method must be corr or mse.'
    else:
        pa = lmea.prod_acc(word, cmat, chat, method)
        _pa = ["1.0000000000", "0.0000000000", "None", "1.0000000000", "0.0000000000", "None", "None", "None", "None"]
        assert '{:12.10f}'.format(pa) == _pa[ind]

cmats = [ (chat,cmat) for i in range(len(wrd)) ]
smats = [ (shat,smat) for i in range(len(wrd)) ]
mats = cmats + smats
pars_uncertainty = wrd * 2
pars_uncertainty = [ (i,*j) for i,j in zip(pars_uncertainty, mats) ]
pars_uncertainty = [ (i,*j) for i,j in enumerate(pars_uncertainty) ]
@pytest.mark.parametrize('ind, word, hat, mat', pars_uncertainty)
def test_uncertainty (ind, word, hat, mat):
    if 'xxx' == word:
        with pytest.raises(KeyError) as e_info:
            unc = lmea.uncertainty(word, hat, mat)
            assert e_info == 'xxx'
    else:
        unc = lmea.uncertainty(word, hat, mat)
        _unc = ["2.1507838011", "2.1429429640", "None", "2.5984557392", "2.0955753635", "None"]
        assert '{:12.10f}'.format(unc) == _unc[ind]


pars_vector_length = [ (i,j) for i,j in enumerate(wrd) ]
@pytest.mark.parametrize('ind, word', pars_vector_length)
def test_vector_length (ind, word):
    if 'xxx' == word:
        with pytest.raises(KeyError) as e_info:
            sp = lmea.vector_length(word, smat)
            assert e_info == 'xxx'
    else:
        vl = lmea.vector_length(word, smat)
        _vl = "8.3922761281", "7.8530324610", "None",
        assert '{:12.10f}'.format(vl) == _vl[ind]
