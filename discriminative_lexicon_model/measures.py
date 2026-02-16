import numpy as np
import xarray as xr
import scipy.spatial.distance as spd

from . import mapping as pmap

__all__ = [
    "functional_load",
    "partial_semantic_support",
    "semantic_support",
    "semantic_support_word",
    "prod_acc",
    "uncertainty",
    "vector_length",
]

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

def partial_semantic_support (current_cue, *, prev_cues, current_word, cmat,
                              smat, fmat, gmat, vmat):
    """
    Calculates the partial semantic support value for the cue provided by
    'current_cue', given a certain context, namely a set of cues provided by
    'prev_cues'.

    Parameters
    ----------
    current_cue : str
        The cue (e.g., triphone) you want to calculate a partial semantic
        support value for (e.g., 'lk#' for 'walk').
    prev_cues : list
        A set of cues (e.g., triphones) that serve as a previous context. A
        partial semantic support value will be calculated for a certain cue
        (i.e., current_cue) given the context specified in this argument.
    current_word : str
        A string that represents the word you focus on. The cues specified by
        current_cue and prev_cues should be a part of this word usually,
        although any cue can be specified with any word technically.
    cmat : xarray.core.dataarray.DataArray
        A form matrix "C". It is assumed to have "word" and "cues" as its
        dimenstions.
    smat : xarray.core.dataarray.DataArray
        A semantic matrix "S". It is assumed to have "word" and "semantics" as
        its dimenstions.
    fmat : xarray.core.dataarray.DataArray
        A cue-semantic weight matrix "F". It is assumed to have "cues" and
        "semantics" as its dimenstions.
    gmat : xarray.core.dataarray.DataArray
        A semantic-cue weight matrix "G". It is assumed to have "semantics" and
        "cues" as its dimenstions.
    vmat : xarray.core.dataarray.DataArray
        A weight matrix between the current and next cues (i.e., "V"). It is
        assumed to have "current" and "next" as its dimenstions.

    Returns
    -------
    p_semsup : float
        A partial semantic support value. It represents how much a certain cue
        is supported from the meaning of the word the cue belongs to, given a
        set of cues as a context, which is provided by the argument
        "prev_cues". A context is usually all the cues of the word preceding
        the target cue. For example, in a usual application, the partial
        semantic support from the meaning of the word "walks" to one of its
        cues "ks#" is the degree to which "ks#" is predictable from the meaning
        of "walks", discounted by how obvious the cue is coming after "#wa",
        "wal", "alk", and "lks".
    """
    prev_cue = prev_cues[-1]

    gmat_t = np.matmul(gmat.values, np.diag(vmat.loc[prev_cue,:].values))
    gmat_t = xr.DataArray(gmat_t, dims=gmat.dims, coords=gmat.coords)

    cvec_t = xr.DataArray(np.zeros((1, cmat.shape[1])), dims=cmat.dims, coords={'word':[current_word], 'cues':cmat.cues.values})
    cvec_t.loc[current_word, prev_cues] = 1
    svec_pred = cvec_t.dot(fmat)
    svec_gold = smat.loc[[current_word],:]
    svec_t = (svec_gold - svec_pred)
    
    chat_t = svec_t.dot(gmat_t)
    p_semsup = float(chat_t.loc[current_word, current_cue].values)
    return p_semsup

def semantic_support (word, cue, chat):
    return float(chat.loc[word,cue].values)

def semantic_support_word (word, chat):
    cuelen = list(set([ len(i) for i in chat.cues.values ]))
    if len(cuelen)!=1:
        raise('Different lengths of cues were detected. Use the "gram" argument when the C-matrix you provide has more than one length of cues.')
    else:
        gram = cuelen[0]
        cues = pmap.to_ngram(x=word, gram=gram)
    return float(chat.loc[word,cues].sum().values)

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

def uncertainty (word, hat, mat):
    def normalize (x):
        x = np.array(x)
        return (x - x.min()) / (x.max()-x.min())
    coss = spd.cdist(np.tile(hat.loc[word,:], (1,1)), np.array(mat), 'cosine')
    coss = 1 - coss
    coss = normalize(coss[0,:])
    coss.sort()
    coss = sum([ i*j for i,j in enumerate(coss) ])
    return coss

def vector_length (word, smat):
    return np.absolute(smat.loc[word,:].values).sum()

