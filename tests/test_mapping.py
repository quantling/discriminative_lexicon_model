import numpy as np
import xarray as xr
import pytest
import pyldl.mapping as pm

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
    (3,False,False,['#ab','aba','aba','ba#','bab'])
]
@pytest.mark.parametrize('gram, unique, keep_order, result', pars_to_ngram)
def test_to_ngram (gram, unique, keep_order, result):
    assert pm.to_ngram('ababa', gram=gram, unique=unique, keep_order=keep_order) == result

def test_gen_cmat ():
    words = ['alt','halt','kalt']
    cmat = pm.gen_cmat(words, cores=1)
    _cmat = np.array([True,False,False,True,False,False,True,False,True,False,True,True,False,True,False,False,True,True,False,True,True])
    _cmat = _cmat.reshape(3,7)
    assert cmat.dims == ('word','cues')
    assert (cmat.values==_cmat).all()
    assert (cmat.word.values==np.array(words)).all()
    assert (cmat.cues.values==np.array(['#al','#ha','#ka','alt','hal','kal','lt#'])).all()
