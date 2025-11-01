from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import numpy as np
import xarray as xr
import fasttext as ft
import gzip
from tqdm import tqdm

from . import mapping as lmap

def to_cues (words, gram=3):
    words = [ '#' + i + '#'for i in words ]
    words = [ i.ljust(max([ len(i) for i in words ]), ' ') for i in words ]
    cues = [ [ i[j:(j+gram)] for j in range(len(i)-(gram-1)) ] for i in words ]
    cues = [ [ j for j in i if not (' ' in j) ] for i in cues ]
    cues = [ j for i in cues for j in i ]
    cues = list(dict.fromkeys(cues))
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

def gen_cmat (words, gram=3, count=True, noise=0, freqs=None, randseed=None, differentiate_duplicates=False):
    cues = unique_cues(words, gram=gram)
    cuelen = list(set([ len(i) for i in cues ]))
    if len(cuelen)!=1:
        raise ValueError('Variable cue length (gram size) detected. Check length of each cue.')
    cmat = [ lmap.to_ngram('#'+i+'#', gram=cuelen[0], unique=False).count(j) for i in words for j in cues ]
    if not count:
        cmat = [ 1 if i>1 else i for i in cmat ]
    cmat = np.array(cmat).reshape(len(words), len(cues))
    if differentiate_duplicates:
        words = _differentiate_duplicates(words)
    coor = {'word':words, 'cues':cues}
    cmat = xr.DataArray(cmat, dims=('word','cues'), coords=coor)
    if noise:
        if isinstance(noise, bool):
            noise = 0.1
        if randseed is None:
            cmat = cmat + np.random.normal(scale=noise, size=cmat.shape)
        else:
            rng = np.random.default_rng(randseed)
            cmat = cmat + rng.normal(scale=noise, size=cmat.shape)
    if not (freqs is None):
        cmat = weight_by_freq(cmat, freqs)
    assert (np.array(words) == cmat.word.values).all()
    return cmat

def unique_cues (words, gram):
    cues = [ to_ngram(i, gram=gram) for i in words ]
    cues = list(dict.fromkeys([ j for i in cues for j in i ]))
    return cues

def _differentiate_duplicates (words):
    uniqs = pd.Series(words) + pd.Series(words, name='hoge').to_frame().groupby('hoge').cumcount().astype(str)
    non_dup_pos = ~pd.Series(words).duplicated(keep=False)
    uniqs.loc[non_dup_pos] = pd.Series(words).loc[non_dup_pos]
    return uniqs.to_list()

def gen_cmat_from_df (df, noise=0, freqs=None, randseed=None):
    """
    Converts a pandas dataframe to a C-matrix. The input dataframe (i.e., df)
    must have indices and columns. Indices and columns of the dataframe are
    expected to be words and form dimensions (possibly sublexical forms such as
    trigrams) respectively. They will be used as coordinates of the output
    xarray.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        It must have indices (words) and columns (form dimensions).
    noise : int or bool
        If it is 0, no random noise will be added to elements of the C-matrix
        to be produced. If it is greater than 0, it will be used as a scale
        (standard deviation) parameter to generate normally-distributed random
        numbers, which will be added to the C-matrix to be produced. If
        noise=True, noise=0.1 will be used. if noise=False, no noise will be
        added.
    freqs : list-like
        A list of frequency of the words. The C-matrix to be generated will be
        weighted by these frequency values (i.e., frequency-weighted learning).
    randseed : None or int
        This option will have an effect only when noise > 0. If randseed is
        None, a random sequence of numbers will be added to the C-matrix to be
        produced (i.e., irreproducible). If randseed is an integer value, the
        random seed will be used, namely which will be reproducible with the
        same random seed later.

    Returns
    -------
    cmat : xarray.core.dataarray.DataArray
        It has indices of the input dataframe as words (the first dimension)
        and columns of the input dataframe as form dimensions (the second
        dimension). The values of the matrix will be the same as it looks in
        df.
    """
    cmat = xr.DataArray(df.to_numpy(), dims=('word', 'cues'),
            coords={'word':df.index.tolist(), 'cues':df.columns})
    if noise:
        if isinstance(noise, bool):
            noise = 0.1
        if randseed is None:
            cmat = cmat + np.random.normal(scale=noise, size=cmat.shape)
        else:
            rng = np.random.default_rng(randseed)
            cmat = cmat + rng.normal(scale=noise, size=cmat.shape)
    if not (freqs is None):
        cmat = weight_by_freq(cmat, freqs)
    return cmat

def gen_smat (words, embed, noise=0, freqs=None, randseed=None):
    """
    Parameters
    ----------
    words : list-like
        A list-like object of words for these semantic vectors will be
        retrieved.
    embed : fasttext.FastText._FastText
        A fasttext pre-trained model, which is assumed to be loaded by
        fasttext.load_model.
    noise : int or bool
        If it is 0, no random noise will be added to elements of the S-matrix
        to be produced. If it is greater than 0, it will be used as a scale
        (standard deviation) parameter to generate normally-distributed random
        numbers, which will be added to the S-matrix to be produced. If
        noise=True, noise=0.1 will be used. if noise=False, no noise will be
        added.
    freqs : list-like
        A list of frequency of the words. The S-matrix to be generated will be
        weighted by these frequency values (i.e., frequency-weighted learning).
    randseed : None or int
        This option will have an effect only when noise > 0. If randseed is
        None, a random sequence of numbers will be added to the S-matrix to be
        produced (i.e., irreproducible). If randseed is an integer value, the
        random seed will be used, namely which will be reproducible with the
        same random seed later.
    """
    if not isinstance(embed, ft.FastText._FastText):
        raise TypeError('Embeddings (the first argument, "embed") must be a fasttext object (i.e., fasttext.FastText._FastText).')
    smat = np.array([ embed.get_word_vector(i) for i in words ])
    sems  = [ 'S{:03d}'.format(i) for i in range(smat.shape[1]) ]
    smat = xr.DataArray(smat, dims=('word', 'semantics'), coords={'word':words,
        'semantics':sems})
    if noise:
        if isinstance(noise, bool):
            noise = 0.1
        if randseed is None:
            smat = smat + np.random.normal(scale=noise, size=smat.shape)
        else:
            rng = np.random.default_rng(randseed)
            smat = smat + rng.normal(scale=noise, size=smat.shape)
    if not (freqs is None):
        smat = weight_by_freq(smat, freqs)
    assert (np.array(words) == smat.word.values).all()
    return smat

def gen_smat_from_df (df, noise=0, freqs=None, randseed=None):
    """
    Converts a pandas dataframe to an S-matrix. The input dataframe (i.e., df)
    must have indices and columns. Indices and columns of the dataframe are
    expected to be words and semantic dimensions respectively. They will be
    used as coordinates of the output xarray.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        It must have indices (words) and columns (semantic dimensions).
    noise : int or bool
        If it is 0, no random noise will be added to elements of the S-matrix
        to be produced. If it is greater than 0, it will be used as a scale
        (standard deviation) parameter to generate normally-distributed random
        numbers, which will be added to the S-matrix to be produced. If
        noise=True, noise=0.1 will be used. if noise=False, no noise will be
        added.
    freqs : list-like
        A list of frequency of the words. The S-matrix to be generated will be
        weighted by these frequency values (i.e., frequency-weighted learning).
    randseed : None or int
        This option will have an effect only when noise > 0. If randseed is
        None, a random sequence of numbers will be added to the S-matrix to be
        produced (i.e., irreproducible). If randseed is an integer value, the
        random seed will be used, namely which will be reproducible with the
        same random seed later.

    Returns
    -------
    smat : xarray.core.dataarray.DataArray
        It has indices of the input dataframe as words (the first dimension)
        and columns of the input dataframe as semantic dimensions (the second
        dimension). The values of the matrix will be the same as it looks in
        df.
    """
    smat = xr.DataArray(df.to_numpy(), dims=('word', 'semantics'),
            coords={'word':df.index.tolist(), 'semantics':df.columns})
    if noise:
        if isinstance(noise, bool):
            noise = 0.1
        if randseed is None:
            smat = smat + np.random.normal(scale=noise, size=smat.shape)
        else:
            rng = np.random.default_rng(randseed)
            smat = smat + rng.normal(scale=noise, size=smat.shape)
    if not (freqs is None):
        smat = weight_by_freq(smat, freqs)
    return smat

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

def incremental_learning_byind (events, cue_matrix, out_matrix):
    _dims = (cue_matrix.dims[1], out_matrix.dims[1])
    _coords = {_dims[0]: cue_matrix[_dims[0]].values.tolist(), _dims[1]: out_matrix[_dims[1]].values.tolist()}
    weight_matrix = np.zeros((cue_matrix.shape[1], out_matrix.shape[1]))
    weight_matrix = xr.DataArray(weight_matrix, dims=_dims, coords=_coords)
    for i in events:
        cvec = cue_matrix[i,:]
        ovec = out_matrix[i,:]
        weight_matrix = lmap.update_weight_matrix(weight_matrix, cvec, ovec, learning_rate=0.1)
    return weight_matrix

def update_weight_matrix (weight_matrix, cue_vector, out_vector, learning_rate=0.1):
    dlt = delta_weight_matrix(weight_matrix, cue_vector, out_vector, learning_rate)
    weight_matrix = weight_matrix + dlt
    return weight_matrix

def delta_weight_matrix (weight_matrix, cue_vector, out_vector, learning_rate=0.1):
    weight_matrix, cue_vector, out_vector = to_nparray(weight_matrix, cue_vector, out_vector)
    dlt = out_vector - matmul(cue_vector, weight_matrix)
    dlt = matmul(cue_vector.T, dlt) * learning_rate
    return dlt

def update_weight_alt (weight_matrix, cue_vector, out_vector, learning_rate=0.1):
    dlt = delta_weight_alt(weight_matrix, cue_vector, out_vector, learning_rate)
    weight_matrix = weight_matrix + dlt
    return weight_matrix

def delta_weight_alt (weight_matrix, cue_vector, out_vector, learning_rate=0.1):
    weight_matrix, cue_vector, out_vector = to_nparray(weight_matrix, cue_vector, out_vector)
    dlt = out_vector - matmul(cue_vector, weight_matrix)
    dlt = cue_vector.T.dot(out_vector) * dlt * learning_rate
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

def save_mat_as_csv (mat, directory='.', stem='mat', add=''):
    """
    Saves an LDL matrix (e.g., C-mat) in csv files. An array/matrix can be
    saved with its attributes altogether, using for example netCDF. However, it
    can sometimes create a problem in saving and loading across different
    environments. Therefore, this function saves an array/matrix and its values
    in separate csv files, so that they can be opened, edited, and loaded
    easily in any environment.

    Parameters
    ----------
    mat : xarray.core.dataarray.DataArray
        An LDL matrix to be saved.
    directory : str
        A path to the directory, in which the matrix (and its attributes) will
        be saved (e.g., '/home/username')
    stem : str
        The stem part of the filenames of the saved csv files. For example,
        stem='foobar' will create four csv files, namely 'foobar_main.csv' for
        the values of the matrix, 'foobar_xxx.csv' for the values of the first
        dimension named 'xxx', 'foobar_yyy_csv' for the values of the second
        dimension named 'yyy', and 'foobar_meta.csv' for the number of the
        dimensions of the matrix.
    add : str
        An additional string, which will be attached to the end of the stem of
        the filename. For example, add='_X' together with stem='foobar' will
        create 'foobar_main_X.csv', 'foobar_xxx_X.csv' where 'xxx' is the name
        of the first dimension of the matrix to be saved, 'foobar_yyy_X.csv'
        where 'yyy' is the name of the second dimension of the matrix to be
        saved, and 'foobar_meta_X.csv'.
    """
    suffix = '.csv'
    name_main = 'main'
    name_meta = 'meta'
    name_dim0 = mat.dims[0]
    name_dim1 = mat.dims[1]
    path_main = '{}/{}_{}{}{}'.format(directory, stem, name_main, add, suffix)
    path_meta = '{}/{}_{}{}{}'.format(directory, stem, name_meta, add, suffix)
    path_dim0 = '{}/{}_{}{}{}'.format(directory, stem, name_dim0, add, suffix)
    path_dim1 = '{}/{}_{}{}{}'.format(directory, stem, name_dim1, add, suffix)
    vals_main = mat.values
    vals_meta = np.array(mat.dims)
    vals_dim0 = mat[name_dim0].values
    vals_dim1 = mat[name_dim1].values
    np.savetxt(path_main, vals_main, delimiter='\t', comments=None)
    np.savetxt(path_meta, vals_meta, fmt='%s', delimiter='\t', comments=None)
    np.savetxt(path_dim0, vals_dim0, fmt='%s', delimiter='\t', comments=None)
    np.savetxt(path_dim1, vals_dim1, fmt='%s', delimiter='\t', comments=None)
    return None

def load_mat_from_csv (directory, stem, add='', suffix='.csv'):
    """
    Loads the csv files that are assumed to have been saved by save_mat_as_csv.

    Parameters
    ----------
    directory : str
        A path to the directory, in which the csv files to be loaded are
        located.
    stem : str
        The stem of the filenames of the csv files to be loaded. For example,
        stem='foobar' will load 'foobar_main.csv', 'foobar_xxx.csv' (where
        'xxx' is the name of the first dimension), 'foobar_yyy.csv' (where
        'yyy' is the name of the second dimension), and 'foobar_meta.csv'.
    add : str
        An additional string, which will be attached to the ends of the stems
        of the filenames. For example, add='_X' with stem='foobar' will load
        the files 'foobar_main_X.csv', 'foobar_xxx_X.csv' where 'xxx' is the
        name of the first dimension, 'foobar_yyy_X.csv' where 'yyy' is the name
        of the second dimension, and 'foobar_meta_X.csv'.
    suffix : str
        The file extension. As default, it is assumed to be ".csv". You can set
        it to ".csv.gz" if the output by save_mat_as_csv is compressed by gzip.

    Returns
    -------
    mat : xarray.core.dataarray.DataArray
        An xarray matrix, reconstructed from the csv files being loaded.
    """
    name_main = 'main'
    name_meta = 'meta'
    path_main = '{}/{}_{}{}{}'.format(directory, stem, name_main, add, suffix)
    path_meta = '{}/{}_{}{}{}'.format(directory, stem, name_meta, add, suffix)
    vals_meta = np.loadtxt(path_meta, dtype=str, delimiter='\t', comments=None)
    name_dim0 = vals_meta[0]
    name_dim1 = vals_meta[1]
    path_dim0 = '{}/{}_{}{}{}'.format(directory, stem, name_dim0, add, suffix)
    path_dim1 = '{}/{}_{}{}{}'.format(directory, stem, name_dim1, add, suffix)
    vals_main = np.loadtxt(path_main, delimiter='\t', comments=None)
    vals_dim0 = load_csv(path_dim0)
    vals_dim1 = load_csv(path_dim1)
    mat = xr.DataArray(vals_main, dims=vals_meta, coords={name_dim0: vals_dim0, name_dim1: vals_dim1})
    return mat

def load_csv (path):
    """
    This function simply loads a csv file, including empty lines, which is
    important for a V-matrix. Including empty lines was a bit cumbersome with
    numpy.loadtxt and pandas.read_csv, and using them would be an overkill just
    to load a simple one-column vector anyways. Therefore, this function was
    created.

    Parameters
    ----------
    path : str
        A path to the csv file to be loaded.

    Returns
    -------
    csv : list
        A list of dimension values.
    """
    if path[-3:]=='.gz':
        with gzip.open(path, 'rt') as f:
            csv = f.readlines()
    else:
        with open(path, 'r') as f:
            csv = f.readlines()
    csv = [ i.rstrip('\n') for i in csv ]
    return csv

