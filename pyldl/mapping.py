import pandas as pd
import numpy as np
import xarray as xr
from multiprocessing import Pool
import pyldl.mapping as lmap
from tqdm import tqdm

def to_cues (words, gram=3):
    words = [ '#' + i + '#'for i in words ]
    words = [ i.ljust(max([ len(i) for i in words ]), ' ') for i in words ]
    cues = [ [ i[j:(j+gram)] for j in range(len(i)-(gram-1)) ] for i in words ]
    cues = [ [ j for j in i if not (' ' in j) ] for i in cues ]
    cues = [ j for i in cues for j in i ]
    cues = sorted(list(set(cues)))
    return cues

def gen_vmat (words, gram=3):
    cues = to_cues(words, gram=gram)
    gram = infer_gram(cues)
    cur  = [ i[-(gram-1):] for i in cues ]
    nex  = [ i[:(gram-1)]  for i in cues ]
    vmat = np.equal.outer(cur, nex)
    vmat = xr.DataArray(vmat, dims=('current','next'), coords={'current':cues, 'next':cues})
    add  = xr.DataArray(np.array([ i[0]=='#' for i in cues]).reshape((1,-1)), dims=vmat.dims, coords={'current':[''], 'next':vmat.next.values})
    vmat = xr.concat((vmat, add), dim='current')
    return vmat

def infer_gram (cues):
    cuelens = sorted(list(set([ len(i) for i in cues ])))
    if len(cuelens)==1:
        gram = cuelens[0]
    else:
        raise ValueError('Multiple grams are found.')
    return gram

def to_ngram (x, gram=2, unique=True, keep_order=True, word_boundary='#'):
    x = '{}{}{}'.format(word_boundary, x, word_boundary)
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

def gen_cmat (words, gram=3, cores=1, differentiate_duplicates=False):
    cues = [ to_ngram(i, gram) for i in words ]
    cues = list(dict.fromkeys([ j for i in cues for j in i ]))
    # cues = sorted(list(set([ j for i in cues for j in i ])))
    if cores==1:
        cmat = [ j in '#'+i+'#' for i in words for j in cues ]
    else:
        cmat = [ (i,j) for i in words for j in cues ]
        with Pool(cores) as p:
            cmat = p.map(_cue_exist, cmat)
    cmat = np.array(cmat).reshape(len(words), len(cues))
    if differentiate_duplicates:
        words = _differentiate_duplicates(words)
    coor = {'word':list(words), 'cues':cues}
    cmat = xr.DataArray(cmat, dims=('word','cues'), coords=coor)
    return cmat.astype(int)

def _differentiate_duplicates (words):
    def _assign_id (x):
        x = x.reset_index(drop=True)
        if len(x)>1:
            x['word'] = x['word'] + x.index.astype(str)
        else:
            x['word'] = x['word']
        return x
    words = pd.DataFrame({'word':words})
    words['id'] = range(len(words))
    words = words.groupby('word', group_keys=False).apply(_assign_id).reset_index(drop=True)
    words = words.sort_values('id')
    return words['word'].tolist()

def gen_smat_sim (infl, form=None, sep=None, dim_size=5, mn=0, sd=1, include_form=True, differentiate_duplicates=False, seed=None):
    mmat = gen_mmat(infl, form, sep, include_form, differentiate_duplicates)
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

def gen_mmat (infl, form=None, sep=None, include_form=True, differentiate_duplicates=False, cores=1):
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
    if differentiate_duplicates:
        words = _differentiate_duplicates(words)
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

def incremental_learning (rows, cue_matrix, out_matrix, learning_rate=0.1, weight_matrix=None, return_intermediate_weights=False):
    if weight_matrix is None:
        _dims = (cue_matrix.dims[1], out_matrix.dims[1])
        _coords = {_dims[0]: cue_matrix[_dims[0]].values.tolist(), _dims[1]: out_matrix[_dims[1]].values.tolist()}
        weight_matrix = np.zeros((cue_matrix.shape[1], out_matrix.shape[1]))
        weight_matrix = xr.DataArray(weight_matrix, dims=_dims, coords=_coords)
    makesure_xarray(cue_matrix, out_matrix, weight_matrix)
    if return_intermediate_weights:
        weight_mats = [weight_matrix]
    for i in tqdm(rows):
        cvec = cue_matrix.loc[[i],:]
        ovec = out_matrix.loc[[i],:]
        weight_matrix = update_weight_matrix(weight_matrix, cvec, ovec, learning_rate)
        if return_intermediate_weights:
            weight_mats = weight_mats + [weight_matrix]
    if return_intermediate_weights:
        res = weight_mats
    else:
        res = weight_matrix
    return res

def update_weight_matrix (weight_matrix, cue_vector, out_vector, learning_rate=0.1):
    dlt = delta_weight_matrix(weight_matrix, cue_vector, out_vector, learning_rate)
    weight_matrix = weight_matrix + dlt
    return weight_matrix

def delta_weight_matrix (weight_matrix, cue_vector, out_vector, learning_rate=0.1):
    weight_matrix, cue_vector, out_vector = to_nparray(weight_matrix, cue_vector, out_vector)
    dlt = out_vector - matmul(cue_vector, weight_matrix)
    dlt = matmul(cue_vector.T, dlt) * learning_rate
    return dlt

def matmul (m1, m2):
    if any(is_xarray(m1, m2)):
        if not np.array_equal(m1[m1.dims[1]].values, m2[m2.dims[0]].values):
            raise ValueError('The second dimension values of the first matrix and the first dimension values of the second matrix do not match.')
        mat = m1 @ m2
    else:
        mat = np.matmul(m1, m2)
    return mat

def makesure_xarray (*args):
    x = [ isinstance(i, xr.core.dataarray.DataArray) for i in args ]
    if not all(x):
        raise TypeError('The input array(s) must be an instance of xarray.core.dataarray.DataArray.')
    return None

def is_xarray (*args):
    x = [ isinstance(i, xr.core.dataarray.DataArray) for i in args ]
    return x

def to_nparray (*args):
    x = [ np.atleast_2d(np.array(i)) for i in args ]
    return x

def weight_by_freq (mat, freqs):
    freqs = np.array(freqs)
    freqs = freqs / freqs.max()
    freqs = np.sqrt(freqs)
    freqs = np.diag(freqs)
    if isinstance(mat, xr.core.dataarray.DataArray):
        mat = xr.DataArray(np.matmul(freqs, mat.values), dims=mat.dims, coords=mat.coords)
    else:
        mat = np.matmul(freqs, mat)
    return mat

