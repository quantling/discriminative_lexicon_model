import pandas as pd
import numpy as np
import xarray as xr
from multiprocessing import Pool

def to_ngram (x, gram=2, unique=True, keep_order=True):
    x = '#{}#'.format(x)
    cues = [ x[(i-gram):i] for i in range(gram,len(x)+1) ]
    if unique and keep_order:
        cues = list(dict.fromkeys(cues))
    elif unique and (not keep_order):
        cues = sorted(list(set(cues)))
    elif (not unique) and (not keep_order):
        cues = sorted(cues)
    return cues

def _cue_exist (x):
    i,j = x
    return j in '#'+i+'#'

def gen_cmat (words, cores=1):
    cues = [ to_ngram(i, gram=3) for i in words ]
    cues = sorted(list(set([ j for i in cues for j in i ])))
    if cores==1:
        cmat = [ j in '#'+i+'#' for i in words for j in cues ]
    else:
        cmat = [ (i,j) for i in words for j in cues ]
        with Pool(cores) as p:
            cmat = p.map(_cue_exist, cmat)
    cmat = np.array(cmat).reshape(len(words), len(cues))
    coor = {'word':list(words), 'cues':cues}
    cmat = xr.DataArray(cmat, dims=('word','cues'), coords=coor)
    return cmat

def gen_smat_sim (infl, form=None, sep=None, dim_size=5, mn=0, sd=1, include_form=True, seed=None):
    mmat = gen_mmat(infl, form, sep, include_form)
    jmat = gen_jmat(mmat, dim_size, mn, sd, seed)
    words = list(mmat.word.values)
    semantics = list(jmat.semantics.values)
    if not all([ isinstance(i, np.ndarray) for i in [mmat, jmat] ]):
        mmat = np.array(mmat)
        jmat = np.array(jmat)
    shat = np.matmul(mmat,jmat)
    coor = {'word':words, 'semantics':semantics}
    shat = xr.DataArray(np.stack(shat), dims=('word', 'semantics'), coords=coor)
    return shat

def gen_mmat (infl, form=None, sep=None, include_form=True, cores=1):
    def one_hot (clm, sep=None, cores=1):
        clm = to_nlist(clm, sep)
        unq = pd.Series(sorted(list(set([ j for i in clm for j in i ]))))
        ent = [ (i,unq) for i in clm ]
        if cores==1:
            arr = [ _one_hot(i) for i in ent ]
        else:
            with Pool(cores) as p:
                arr = p.map(_one_hot, ent)
        arr = np.array(arr)
        return arr
    def _one_hot (entry):
        ent,unq = entry
        return list((unq.isin(pd.Series(ent))).astype(int))
    def to_nlist (clm, sep=None):
        if sep is None:
            clm = [ [i] for i in clm ]
        else:
            clm = [ i.split(sep) for i in clm ]
        return clm
    def to_unique (clm, sep=None):
        clm = to_nlist(clm, sep)
        unq = sorted(list(set([ j for i in clm for j in i ])))
        return unq

    if (form is None) and (include_form):
        words = infl.iloc[:,0]
    elif (form is None) and (not include_form):
        raise ValueError('Specify which column to drop by the argument "form" when "include_form" is False.')
    elif (not (form is None)) and (include_form):
        words = infl[form]
    elif (not (form is None)) and (not include_form):
        words = infl[form]
        infl = infl.drop(columns=[form])
    
    aaa = [ one_hot(infl[i], sep=sep, cores=cores) for i in infl.columns ]
    aaa = np.concatenate(aaa, axis=1)
    bbb = [ [ '{}:{}'.format(i,j) for j in to_unique(infl[i],sep) ] for i in infl.columns ]
    bbb = [ j for i in bbb for j in i ]
    coor = {'word':words, 'feature':bbb}
    aaa = xr.DataArray(aaa, dims=('word','feature'), coords=coor)
    return aaa

def gen_jmat (mmat, dim_size, mn=0, sd=1, seed=None):
    def rand_norm_seed (dim_size, seed=None):
        if not (seed is None):
            np.random.seed(seed)
        vec = np.random.normal(loc=mn, scale=sd, size=dim_size)
        return vec
    features = list(mmat.feature.values)
    if seed is None:
        aaa = [ rand_norm_seed(dim_size, seed=seed) for j in features ]
    else:
        aaa = [ rand_norm_seed(dim_size, seed=seed+i) for i,j in enumerate(features) ]
    coor = {'feature': features, 'semantics':[ 'S{:03d}'.format(i) for i in range(dim_size) ]}
    aaa = xr.DataArray(np.stack(aaa), dims=('feature', 'semantics'), coords=coor)
    return aaa

def gen_fmat (cmat, smat):
    if isinstance(cmat, xr.DataArray) and isinstance(smat, xr.DataArray):
        rname = list(cmat.coords)[1]
        rvals = cmat[rname]
        cname = list(smat.coords)[1]
        cvals = smat[cname]
    if not all([ isinstance(i, np.ndarray) for i in [cmat, smat] ]):
        cmat = np.array(cmat)
        smat = np.array(smat)
    fmat = np.matmul(np.matmul(np.linalg.pinv(np.matmul(cmat.T,cmat)),cmat.T),smat)
    fmat = xr.DataArray(fmat, dims=(rname, cname), coords={rname:rvals, cname:cvals})
    return fmat

def gen_gmat (cmat, smat):
    if isinstance(cmat, xr.DataArray) and isinstance(smat, xr.DataArray):
        rname = list(smat.coords)[1]
        rvals = smat[rname]
        cname = list(cmat.coords)[1]
        cvals = cmat[cname]
    if not all([ isinstance(i, np.ndarray) for i in [cmat, smat] ]):
        cmat = np.array(cmat)
        smat = np.array(smat)
    gmat = np.matmul(np.matmul(np.linalg.pinv(np.matmul(smat.T,smat)),smat.T),cmat)
    gmat = xr.DataArray(gmat, dims=(rname, cname), coords={rname:rvals, cname:cvals})
    return gmat

def gen_shat (cmat=None, fmat=None, smat=None, hmat=None):
    if all([ not (i is None) for i in [cmat, fmat] ]):
        if isinstance(cmat, xr.DataArray) and isinstance(fmat, xr.DataArray):
            rname = list(cmat.coords)[0]
            rvals = cmat[rname]
            cname = list(fmat.coords)[1]
            cvals = fmat[cname]
        cmat = np.array(cmat) if not isinstance(cmat, np.ndarray) else cmat
        fmat = np.array(fmat) if not isinstance(fmat, np.ndarray) else fmat
        shat = np.matmul(cmat, fmat)
        shat = xr.DataArray(shat, dims=(rname, cname), coords={rname:rvals, cname:cvals})
    elif all([ not (i is None) for i in [cmat, smat] ]):
        fmat = gen_fmat(cmat, smat)
        if isinstance(cmat, xr.DataArray) and isinstance(fmat, xr.DataArray):
            rname = list(cmat.coords)[0]
            rvals = cmat[rname]
            cname = list(fmat.coords)[1]
            cvals = fmat[cname]
        cmat = np.array(cmat) if not isinstance(cmat, np.ndarray) else cmat
        fmat = np.array(fmat) if not isinstance(fmat, np.ndarray) else fmat
        shat = np.matmul(cmat, fmat)
        shat = xr.DataArray(shat, dims=(rname, cname), coords={rname:rvals, cname:cvals})
    elif all([ not (i is None) for i in [hmat, smat] ]):
        if isinstance(hmat, xr.DataArray) and isinstance(smat, xr.DataArray):
            rname = list(hmat.coords)[0]
            rvals = hmat[rname]
            cname = list(smat.coords)[1]
            cvals = smat[cname]
        hmat = np.array(hmat) if not isinstance(hmat, np.ndarray) else hmat
        smat = np.array(smat) if not isinstance(smat, np.ndarray) else smat
        shat = np.matmul(hmat, smat)
        shat = xr.DataArray(shat, dims=(rname, cname), coords={rname:rvals, cname:cvals})
    else:
        raise ValueError('(C, F), (C, S), or (H, S) is necessary.')
    return shat

def gen_chat (smat=None, gmat=None, cmat=None, hmat=None):
    if all([ not (i is None) for i in [smat, gmat] ]):
        if isinstance(smat, xr.DataArray) and isinstance(gmat, xr.DataArray):
            rname = list(smat.coords)[0]
            rvals = smat[rname]
            cname = list(gmat.coords)[1]
            cvals = gmat[cname]
        smat = np.array(smat) if not isinstance(smat, np.ndarray) else smat
        gmat = np.array(gmat) if not isinstance(gmat, np.ndarray) else gmat
        chat = np.matmul(smat, gmat)
        chat = xr.DataArray(chat, dims=(rname, cname), coords={rname:rvals, cname:cvals})
    elif all([ not (i is None) for i in [smat, cmat] ]):
        gmat = gen_gmat(cmat, smat)
        if isinstance(smat, xr.DataArray) and isinstance(gmat, xr.DataArray):
            rname = list(smat.coords)[0]
            rvals = smat[rname]
            cname = list(gmat.coords)[1]
            cvals = gmat[cname]
        smat = np.array(smat) if not isinstance(smat, np.ndarray) else smat
        gmat = np.array(gmat) if not isinstance(gmat, np.ndarray) else gmat
        chat = np.matmul(smat, gmat)
        chat = xr.DataArray(chat, dims=(rname, cname), coords={rname:rvals, cname:cvals})
    elif all([ not (i is None) for i in [hmat, cmat] ]):
        if isinstance(hmat, xr.DataArray) and isinstance(cmat, xr.DataArray):
            rname = list(hmat.coords)[0]
            rvals = hmat[rname]
            cname = list(cmat.coords)[1]
            cvals = cmat[cname]
        hmat = np.array(hmat) if not isinstance(hmat, np.ndarray) else hmat
        cmat = np.array(cmat) if not isinstance(cmat, np.ndarray) else cmat
        chat = np.matmul(hmat, cmat)
        chat = xr.DataArray(chat, dims=(rname, cname), coords={rname:rvals, cname:cvals})
    else:
        raise ValueError('(S, G), (S, C), or (H, C) is necessary.')
    return chat

