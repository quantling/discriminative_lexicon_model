import numpy as np
import xarray as xr
import scipy.spatial.distance as spd
import pyldl.mapping as pmap

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


