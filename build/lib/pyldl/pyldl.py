import pandas as pd
import numpy as np
import xarray as xr
from multiprocessing import Pool
import scipy.spatial.distance as spd


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
    cmat = cmat.loc[sorted(list(set(cmat.word.values))),:]
    return cmat

def gen_smat_sim (infl, form=None, sep=None, dim_size=5):
    mmat = gen_mmat(infl, form, sep)
    jmat = gen_jmat(mmat, dim_size)
    words = list(mmat.word.values)
    semantics = list(jmat.semantics.values)
    if not all([ isinstance(i, np.ndarray) for i in [mmat, jmat] ]):
        mmat = np.array(mmat)
        jmat = np.array(jmat)
    shat = np.matmul(mmat,jmat)
    coor = {'word':words, 'semantics':semantics}
    shat = xr.DataArray(np.stack(shat), dims=('word', 'semantics'), coords=coor)
    return shat

def gen_mmat (infl, form=None, sep=None, cores=1):
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
    
    aaa = [ one_hot(infl[i], sep=sep, cores=cores) for i in infl.columns ]
    aaa = np.concatenate(aaa, axis=1)
    bbb = [ [ '{}:{}'.format(i,j) for j in to_unique(infl[i],sep) ] for i in infl.columns ]
    bbb = [ j for i in bbb for j in i ]
    if form is None:
        form = infl.columns[0]
    coor = {'word':infl[form], 'feature':bbb}
    aaa = xr.DataArray(aaa, dims=('word','feature'), coords=coor)
    return aaa

def gen_jmat (mmat, dim_size):
    features = list(mmat.feature.values)
    aaa = [ np.random.normal(loc=0, scale=1, size=dim_size) for i in features ]
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

def functional_load (cue, fmat, word, smat, method='corr'):
    cvec = fmat.loc[cue,:]
    wvec = smat.loc[word,:]
    if method=='corr':
        fload = float(xr.corr(cvec, wvec).values)
    elif method=='mse':
        fload = float(((cvec-wvec)**2).mean().values)
    else:
        raise ValueError('method must be corr or mse.')
    return fload

def semantic_support (word, cue, cmat):
    return float(cmat.loc[word,cue].values)

def prod_acc (word, cmat, chat, method='corr'):
    cmvec = cmat.loc[word,:].astype(int)
    chvec = chat.loc[word,:]
    if method=='corr':
        chacc = float(xr.corr(cmvec, chvec).values)
    elif method=='mse':
        chacc = float(((cmvec-chvec)**2).mean().values)
    else:
        raise ValueError('method must be corr or mse.')
    return chacc

def uncertainty (word, hat, mat, method='cosine', distance=False):
    if distance:
        distance = False
        print('WARNING: The current version of this function allows only distance=False. Therefore, distance was reset to False.')
    def normalize (x):
        x = np.array(x)
        return (x - x.min()) / (x.max()-x.min())
    coss = spd.cdist(np.tile(hat.loc[word,:], (1,1)), np.array(mat), method)
    if distance:
        pass
    else:
        coss = 1 - coss
    coss = normalize(coss[0,:])
    coss.sort()
    coss = sum([ i*j for i,j in enumerate(coss) ])
    return coss

def vector_length (word, smat, method='l1norm'):
    # Only l1norm is available now.
    return np.absolute(smat.loc[word,:].values).sum()


### Below is not necessary for the current project. ###
### But they need to be included in the package 'pyldl' ###

def accuracy (hat, mat, max_guess=1, method='cosine', distance=False):
    coss = spd.cdist(np.array(hat), np.array(mat), method)
    if distance:
        pos1 = [np.argmin(coss, axis=1)]
        sign = 1
    else:
        coss = 1 - coss
        pos1 = [np.argmax(coss, axis=1)]
        sign = -1
    assert isinstance(max_guess, int)
    if max_guess>1:
        pos = [ np.apply_along_axis(lambda x: np.argsort(x)[(sign*i)], 1, coss) for i in range(2,max_guess+1) ]
    else:
        pos = []
    pos = pos1 + pos
    prds = [ [ mat.word.values[j] for j in i ] for i in pos ]
    hits = [ [ j==k for j,k in zip(i,hat.word.values) ] for i in prds ]
    if len(prds)==1:
        prds = [ pd.DataFrame({'pred':j}) for j in prds ]
        hits = [ pd.DataFrame({'acc':j}) for j in hits ]
    else:
        prds = [ pd.DataFrame({'pred{:d}'.format(i+1):j}) for i,j in enumerate(prds) ]
        hits = [ pd.DataFrame({'acc{:d}'.format(i+1):j}) for i,j in enumerate(hits) ]
    prds = pd.concat(prds, axis=1)
    hits = pd.concat(hits, axis=1)
    wrds = pd.DataFrame({'WordDISC':hat.word.values})
    dddd = pd.concat([wrds,prds,hits], axis=1)
    return dddd

def predict (word, hat, mat, method='cosine', distance=False):
    hat = np.tile(hat.loc[word,:], (1,1))
    coss = spd.cdist(np.array(hat), np.array(mat), method)
    if distance:
        sign = 1
    else:
        coss = 1 - coss
        sign = -1
    coss = coss[0,:]
    pred = mat.word.values[np.argsort(sign*coss)]
    return pd.Series(pred)

