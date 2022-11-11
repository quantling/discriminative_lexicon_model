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

def partial_semantic_support (form, word, cmat, smat, fmat, gmat, vmat=None):
    cues = pmap.to_ngram(word, len(form), unique=False, keep_order=True)
    pos  = [ i for i,j in enumerate(cues) if j==form ]
    if len(pos)==0:
        raise ValueError('The specified form was not found in the specified word.')
    elif len(pos)>1:
        raise ValueError('The specified form was found at multiple locations of the specified word.')
    else:
        pos = pos[0]

    gold = smat.loc[word,:]

    prev_cues = cues[:pos]
    cvec_t = np.zeros((1,cmat.shape[1]))
    cvec_t = xr.DataArray(cvec_t, dims=('word','cues'), coords={'word':[word], 'cues':cmat.cues.values})
    cvec_t.loc[word, prev_cues] = 1
    shat_t_minus1 = np.matmul(np.array(cvec_t), np.array(fmat))
    svec_t = np.array(gold) - shat_t_minus1

    if vmat is None:
        print('No V-matrix is provided. A V-matrix will be estimated. This process may take a while.')
        vmat = pmap.gen_vmat(cmat.cues.values)

    if len(prev_cues)==0:
        prev_cues = ['']
    gmat_t = np.matmul(np.array(gmat), np.array(np.diag(vmat.loc[prev_cues[-1],:])))

    chat_t = np.matmul(svec_t, gmat_t)
    chat_t = xr.DataArray(chat_t, dims=('word','cues'), coords={'word':[word], 'cues':cmat.cues.values})
    semsup_t = chat_t.loc[word, form]

    return float(semsup_t)

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

