import pytest
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pyldl.mapping as pm
import xarray as xr
from pyldl.ldl import LDL

TEST_ROOT = Path('.')
#TEST_ROOT = Path(__file__).parent
RESOURCES = TEST_ROOT / 'resources'

words = ['ban','banban']
freqs = [10, 20]
semdf = pd.DataFrame({'hit':[1,1], 'intensity':[1,2]}, index=words)

cmat_shape = (2, 5)
cmat_values = np.array([1, 1, 1, 0, 0, 1, 2, 1, 1, 1]).reshape(cmat_shape)
cmat_dims = ('word', 'cues')
cmat_word_values = ['ban', 'banban']
cmat_cues_values = ['#ba', 'ban', 'an#', 'anb', 'nba']
cmat = xr.DataArray(cmat_values, dims=cmat_dims, coords={cmat_dims[0]: cmat_word_values, cmat_dims[1]:cmat_cues_values})

cmatfreq_shape = (2, 5)
cmatfreq_values = np.array([0.7071067811865476, 0.7071067811865476, 0.7071067811865476, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 1.0]).reshape(cmatfreq_shape)
cmatfreq_dims = ('word', 'cues')
cmatfreq_word_values = ['ban', 'banban']
cmatfreq_cues_values = ['#ba', 'ban', 'an#', 'anb', 'nba']
cmatfreq = xr.DataArray(cmatfreq_values, dims=cmatfreq_dims, coords={cmatfreq_dims[0]: cmatfreq_word_values, cmatfreq_dims[1]:cmatfreq_cues_values})

smat_shape = (2, 2)
smat_values = np.array([1, 1, 1, 2]).reshape(smat_shape)
smat_dims = ('word', 'semantics')
smat_word_values = ['ban', 'banban']
smat_semantics_values = ['hit', 'intensity']
smat = xr.DataArray(smat_values, dims=smat_dims, coords={smat_dims[0]: smat_word_values, smat_dims[1]:smat_semantics_values})

smatfreq_shape = (2, 2)
smatfreq_values = np.array([0.7071067811865476, 0.7071067811865476, 1.0, 2.0]).reshape(smatfreq_shape)
smatfreq_dims = ('word', 'semantics')
smatfreq_word_values = ['ban', 'banban']
smatfreq_semantics_values = ['hit', 'intensity']
smatfreq = xr.DataArray(smatfreq_values, dims=smatfreq_dims, coords={smatfreq_dims[0]: smatfreq_word_values, smatfreq_dims[1]:smatfreq_semantics_values})

fmat_shape = (5, 2)
fmat_values = np.array([0.37500000000000006, 0.2499999999999999, 0.25000000000000006, 0.4999999999999999, 0.37499999999999967, 0.24999999999999994, -0.125, 0.2499999999999999, -0.125, 0.2499999999999999]).reshape(fmat_shape)
fmat_dims = ('cues', 'semantics')
fmat_cues_values = ['#ba', 'ban', 'an#', 'anb', 'nba']
fmat_semantics_values = ['hit', 'intensity']
fmat = xr.DataArray(fmat_values, dims=fmat_dims, coords={fmat_dims[0]: fmat_cues_values, fmat_dims[1]:fmat_semantics_values})

gmat_shape = (2, 5)
gmat_values = np.array([0.9999999999999964, -8.881784197001252e-16, 0.9999999999999964, -0.9999999999999973, -0.9999999999999973, 1.9984014443252818e-15, 1.0000000000000002, 1.9984014443252818e-15, 0.9999999999999982, 0.9999999999999982]).reshape(gmat_shape)
gmat_dims = ('semantics', 'cues')
gmat_semantics_values = ['hit', 'intensity']
gmat_cues_values = ['#ba', 'ban', 'an#', 'anb', 'nba']
gmat = xr.DataArray(gmat_values, dims=gmat_dims, coords={gmat_dims[0]: gmat_semantics_values, gmat_dims[1]:gmat_cues_values})

vmat_shape = (6, 5)
vmat_values = np.array([False, True, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False]).reshape(vmat_shape)
vmat_dims = ('current', 'next')
vmat_current_values = ['#ba', 'ban', 'an#', 'anb', 'nba', '']
vmat_next_values    = ['#ba', 'ban', 'an#', 'anb', 'nba']
vmat = xr.DataArray(vmat_values, dims=vmat_dims, coords={vmat_dims[0]: vmat_current_values, vmat_dims[1]:vmat_next_values})

shat_shape = (2, 2)
shat_values = np.array([0.9999999999999998, 0.9999999999999998, 1.0, 1.9999999999999991]).reshape(shat_shape)
shat_dims = ('word', 'semantics')
shat_word_values = ['ban', 'banban']
shat_semantics_values = ['hit', 'intensity']
shat = xr.DataArray(shat_values, dims=shat_dims, coords={shat_dims[0]: shat_word_values, shat_dims[1]:shat_semantics_values})

chat_shape = (2, 5)
chat_values = np.array([0.9999999999999984, 0.9999999999999993, 0.9999999999999984, 8.881784197001252e-16, 8.881784197001252e-16, 1.0000000000000004, 1.9999999999999996, 1.0000000000000004, 0.9999999999999991, 0.9999999999999991]).reshape(chat_shape)
chat_dims = ('word', 'cues')
chat_word_values = ['ban', 'banban']
chat_cues_values = ['#ba', 'ban', 'an#', 'anb', 'nba']
chat = xr.DataArray(chat_values, dims=chat_dims, coords={chat_dims[0]: chat_word_values, chat_dims[1]:chat_cues_values})

def test_empty_initialize ():
    ldl = LDL()
    assert isinstance(ldl, LDL)
    assert ldl.__dict__ == dict()

def test_initialize_with_matrices ():
    ldl = LDL(words, semdf, allmatrices=True)
    assert all(pd.Series(ldl.__dict__.keys()) == pd.Series(['words', 'cmat',
        'smat', 'fmat', 'gmat', 'vmat', 'chat', 'shat']))
    assert all(pd.Series(ldl.words) == pd.Series(['ban', 'banban']))
    assert ldl.cmat.identical(cmat)
    assert ldl.smat.identical(smat)
    assert ldl.fmat.identical(fmat)
    assert ldl.gmat.identical(gmat)
    assert ldl.vmat.identical(vmat)
    assert ldl.shat.identical(shat)
    assert ldl.chat.identical(chat)

def test_gen_cmat ():
    ldl = LDL()
    ldl.gen_cmat(words=words)
    assert ldl.cmat.identical(cmat)

def test_gen_cmat_withfreq ():
    ldl = LDL()
    ldl.gen_cmat(words=words, freqs=freqs)
    assert ldl.cmat.identical(cmatfreq)

def test_gen_smat ():
    ldl = LDL()
    ldl.gen_smat(embed_or_df=semdf, words=words)
    assert ldl.smat.identical(smat)

def test_gen_smat_withfreq ():
    ldl = LDL()
    ldl.gen_smat(embed_or_df=semdf, words=words, freqs=freqs)
    assert ldl.smat.identical(smatfreq)
