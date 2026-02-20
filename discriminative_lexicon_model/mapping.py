from pathlib import Path
from multiprocessing import Pool
import warnings

import pandas as pd
import numpy as np
import xarray as xr
import fasttext as ft
import gzip
from tqdm import tqdm

from . import mapping as lmap

__all__ = [
    "to_cues",
    "to_ngram",
    "infer_gram",
    "gen_vmat",
    "gen_cmat",
    "gen_cmat_from_df",
    "gen_smat",
    "gen_smat_from_df",
    "gen_smat_sim",
    "gen_mmat",
    "gen_jmat",
    "gen_fmat",
    "gen_gmat",
    "gen_shat",
    "gen_chat",
    "gen_chat_produce",
    "produce_paradigm",
    "produce",
    "incremental_learning",
    "weight_by_freq",
    "save_mat_as_csv",
    "load_mat_from_csv",
    "load_csv",
    "save_mat",
    "load_mat",
]

try:
    import torch
except ImportError:
    torch = None

def to_cues (words, gram=3):
    cues = [ to_ngram(i, gram=gram) for i in words ]
    cues = list(dict.fromkeys([ j for i in cues for j in i ]))
    return cues

def gen_vmat (words=None, gram=3, cues=None):
    """
    Generate a validity matrix (V-matrix) from a set of cues.

    Parameters
    ----------
    words : list-like or None
        Deprecated. A list of words from which cues will be derived. Use
        cues parameter instead.
    gram : int
        N-gram size. Only used when deriving cues from words (deprecated).
    cues : list-like or None
        A list of cues (e.g., trigrams). If provided, words and gram are
        ignored. This is the recommended parameter.

    Returns
    -------
    vmat : xarray.DataArray
        Validity matrix with dims (current, next).
    """
    if cues is None:
        if words is None:
            raise ValueError('Either cues or words must be provided.')
        warnings.warn(
            'Passing words to gen_vmat is deprecated. '
            'Pass cues instead, e.g. gen_vmat(cues=to_cues(words, gram=gram)).',
            DeprecationWarning,
            stacklevel=2,
        )
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

def gen_cmat (words, gram=3, count=True, noise=0, freqs=None, randseed=None, differentiate_duplicates=False, cues=None):
    if cues is None:
        cues = to_cues(words, gram=gram)
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
    warnings.warn('unique_cues is deprecated. Use to_cues instead.', DeprecationWarning, stacklevel=2)
    return to_cues(words, gram=gram)

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
        rname = cmat.dims[1]
        rvals = cmat[rname]
        cname = smat.dims[1]
        cvals = smat[cname]
    if not all([ isinstance(i, np.ndarray) for i in [cmat, smat] ]):
        cmat = np.array(cmat)
        smat = np.array(smat)
    fmat = np.matmul(np.matmul(np.linalg.pinv(np.matmul(cmat.T,cmat)),cmat.T),smat)
    fmat = xr.DataArray(fmat, dims=(rname, cname), coords={rname:rvals, cname:cvals})
    return fmat

def gen_gmat (cmat, smat):
    if isinstance(cmat, xr.DataArray) and isinstance(smat, xr.DataArray):
        rname = smat.dims[1]
        rvals = smat[rname]
        cname = cmat.dims[1]
        cvals = cmat[cname]
    if not all([ isinstance(i, np.ndarray) for i in [cmat, smat] ]):
        cmat = np.array(cmat)
        smat = np.array(smat)
    gmat = np.matmul(np.matmul(np.linalg.pinv(np.matmul(smat.T,smat)),smat.T),cmat)
    gmat = xr.DataArray(gmat, dims=(rname, cname), coords={rname:rvals, cname:cvals})
    return gmat

def gen_shat (cmat=None, fmat=None, smat=None, hmat=None, backend='auto', device=None):
    """
    Generate predicted semantic matrix (S-hat) via matrix multiplication.

    Parameters
    ----------
    cmat : xr.DataArray or np.ndarray, optional
        C-matrix (cue matrix).
    fmat : xr.DataArray or np.ndarray, optional
        F-matrix (mapping from cues to semantics).
    smat : xr.DataArray or np.ndarray, optional
        S-matrix (semantic matrix). Used with cmat to compute fmat first.
    hmat : xr.DataArray or np.ndarray, optional
        H-matrix. Used with smat for alternative computation.
    backend : str, optional
        Computation backend: 'numpy', 'torch', or 'auto' (default).
        'auto' uses torch with CUDA if available, otherwise numpy.
    device : str, optional
        Device for torch backend: 'cuda', 'cpu', or None (auto-detect).

    Returns
    -------
    shat : xr.DataArray
        Predicted semantic matrix.
    """
    # Determine which matrices to multiply
    if all([ not (i is None) for i in [cmat, fmat] ]):
        mat_a, mat_b = cmat, fmat
        if isinstance(cmat, xr.DataArray) and isinstance(fmat, xr.DataArray):
            rname = list(cmat.coords)[0]
            rvals = cmat[rname]
            cname = list(fmat.coords)[1]
            cvals = fmat[cname]
    elif all([ not (i is None) for i in [cmat, smat] ]):
        fmat = gen_fmat(cmat, smat)
        mat_a, mat_b = cmat, fmat
        if isinstance(cmat, xr.DataArray) and isinstance(fmat, xr.DataArray):
            rname = list(cmat.coords)[0]
            rvals = cmat[rname]
            cname = list(fmat.coords)[1]
            cvals = fmat[cname]
    elif all([ not (i is None) for i in [hmat, smat] ]):
        mat_a, mat_b = hmat, smat
        if isinstance(hmat, xr.DataArray) and isinstance(smat, xr.DataArray):
            rname = list(hmat.coords)[0]
            rvals = hmat[rname]
            cname = list(smat.coords)[1]
            cvals = smat[cname]
    else:
        raise ValueError('(C, F), (C, S), or (H, S) is necessary.')

    # Determine backend
    use_torch = False
    if backend == 'torch':
        use_torch = True
    elif backend == 'auto':
        if torch is not None and torch.cuda.is_available():
            use_torch = True

    # Compute matrix multiplication
    if use_torch:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mat_a_t = torch.as_tensor(np.array(mat_a), dtype=torch.float32, device=device)
        mat_b_t = torch.as_tensor(np.array(mat_b), dtype=torch.float32, device=device)
        shat_vals = torch.mm(mat_a_t, mat_b_t).cpu().numpy()
    else:
        mat_a = np.array(mat_a) if not isinstance(mat_a, np.ndarray) else mat_a
        mat_b = np.array(mat_b) if not isinstance(mat_b, np.ndarray) else mat_b
        shat_vals = np.matmul(mat_a, mat_b)

    shat = xr.DataArray(shat_vals, dims=(rname, cname), coords={rname:rvals, cname:cvals})
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

def gen_chat_produce (smat, cmat, fmat, gmat, vmat=None, roundby=10, max_attempt=50, positive=False, apply_vmat=True, backend='auto', device=None, stop='convergence', tol=0.0):
    """
    Generate predicted cue matrix (C-hat) using incremental production.

    Unlike gen_chat which computes C-hat via matrix multiplication (S @ G),
    this function uses the produce algorithm to incrementally select cues
    for each word. For each word, the semantic vector is fed to produce(),
    which selects cues one by one, each time generating a predicted c-hat
    vector. The c-hat vectors across all steps are summed to form the
    word's row in the resulting C-hat matrix.

    Parameters
    ----------
    smat : xarray.DataArray
        S-matrix (semantic matrix), with dims (word, semantics).
    cmat : xarray.DataArray
        C-matrix (cue matrix).
    fmat : xarray.DataArray
        F-matrix (mapping from cues to semantics).
    gmat : xarray.DataArray
        G-matrix (mapping from semantics to cues).
    vmat : xarray.DataArray or None
        Validity matrix. Required when apply_vmat is True.
    roundby : int
        Number of decimal places to round vectors in produce.
    max_attempt : int
        Maximum number of iterations per word.
    positive : bool
        If True, set negative values to zero during production.
    apply_vmat : bool
        If True, apply validity matrix during production.
    backend : {'numpy', 'torch', 'auto'}
        Backend for produce computation.
    device : str or None
        Device for torch backend.
    stop : {'convergence', 'boundary'}
        Stopping criterion passed to produce().
    tol : float
        Tolerance for the convergence check, passed to produce().

    Returns
    -------
    chat : xarray.DataArray
        Predicted cue matrix with dims (word, cues).
    """
    words = smat.word.values
    cues = cmat.cues.values
    rows = []
    for word in words:
        gold = smat.sel(word=word).values
        result = produce(gold, cmat, fmat, gmat, vmat=vmat, word=False,
                         roundby=roundby, max_attempt=max_attempt,
                         positive=positive, apply_vmat=apply_vmat,
                         backend=backend, device=device, stop=stop, tol=tol)
        numeric_cols = result.drop(columns=['Selected'])
        if len(numeric_cols) == 0:
            row_sum = np.zeros(len(cues))
        else:
            row_sum = numeric_cols.sum(axis=0).values
        rows.append(row_sum)
    chat = np.stack(rows)
    chat = xr.DataArray(chat, dims=('word', 'cues'),
                        coords={'word': list(words), 'cues': list(cues)})
    return chat

def produce_paradigm (smat, cmat, fmat, gmat, vmat=None, roundby=10, max_attempt=50, positive=False, apply_vmat=True, backend='auto', device=None, stop='convergence', tol=0.0):
    """
    Apply produce to each word in smat and return a single DataFrame.

    For each word (row of smat), produce() is called to incrementally
    select cues. The per-word DataFrames are concatenated into one, with
    'index' (positional index in smat) and 'word' columns prepended.

    Parameters
    ----------
    smat : xarray.DataArray
        S-matrix (semantic matrix), with dims (word, semantics).
    cmat : xarray.DataArray
        C-matrix (cue matrix).
    fmat : xarray.DataArray
        F-matrix (mapping from cues to semantics).
    gmat : xarray.DataArray
        G-matrix (mapping from semantics to cues).
    vmat : xarray.DataArray or None
        Validity matrix. Required when apply_vmat is True.
    roundby : int
        Number of decimal places to round vectors in produce.
    max_attempt : int
        Maximum number of iterations per word.
    positive : bool
        If True, set negative values to zero during production.
    apply_vmat : bool
        If True, apply validity matrix during production.
    backend : {'numpy', 'torch', 'auto'}
        Backend for produce computation.
    device : str or None
        Device for torch backend.
    stop : {'convergence', 'boundary'}
        Stopping criterion passed to produce().
    tol : float
        Tolerance for the convergence check, passed to produce().

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame with columns 'index', 'word', 'pred', 'step', 'Selected',
        followed by one column per cue with the c-hat values at each
        time step.
    """
    words = smat.word.values
    dfs = []
    for i, word in enumerate(words):
        gold = smat.values[i]
        result = produce(gold, cmat, fmat, gmat, vmat=vmat, word=False,
                         roundby=roundby, max_attempt=max_attempt,
                         positive=positive, apply_vmat=apply_vmat,
                         backend=backend, device=device, stop=stop, tol=tol)
        # Compute predicted word from selected cues
        from .ldl import concat_cues
        selected = result['Selected']
        if len(selected) > 0 and selected.iloc[-1].endswith('#'):
            predicted = concat_cues(selected)
        else:
            predicted = ''
        result.insert(0, 'step', range(len(result)))
        result.insert(0, 'pred', predicted)
        result.insert(0, 'word', word)
        result.insert(0, 'index', i)
        # Append a sum row for the c-hat vectors
        numeric_cols = result.drop(columns=['index', 'word', 'pred', 'step', 'Selected'])
        sum_vals = numeric_cols.sum(axis=0)
        sum_row = pd.DataFrame([{
            'index': i,
            'word': word,
            'pred': predicted,
            'step': '(sum)',
            'Selected': '(sum)',
            **sum_vals.to_dict(),
        }])
        result = pd.concat([result, sum_row], ignore_index=True)
        dfs.append(result)
    df = pd.concat(dfs, ignore_index=True)
    return df

def produce (gold, cmat, fmat, gmat, vmat=None, word=False, roundby=10, max_attempt=50, positive=False, apply_vmat=True, backend='auto', device=None, stop='convergence', tol=0.0):
    """
    Produce output using discriminative learning (standalone version).

    Parameters
    ----------
    gold : array-like
        The target semantic vector.
    cmat : xarray.DataArray
        The C-matrix (cue matrix).
    fmat : xarray.DataArray
        The F-matrix (mapping from cues to semantics).
    gmat : xarray.DataArray
        The G-matrix (mapping from semantics to cues).
    vmat : xarray.DataArray or None
        The validity matrix. Required when apply_vmat is True.
    word : bool
        If True, concatenate cues to form words.
    roundby : int
        Number of decimal places to round vectors.
    max_attempt : int
        Maximum number of iterations.
    positive : bool
        If True, set negative values to zero.
    apply_vmat : bool
        If True, apply validity matrix (vmat must be provided).
    backend : {'numpy', 'torch', 'auto'}
        'numpy' -> NumPy CPU implementation.
        'torch' -> PyTorch implementation (CPU/GPU depending on 'device').
        'auto'  -> Try torch+CUDA if available, else fall back to NumPy.
    device : str or None
        For torch backend: 'cuda', 'cpu', etc. If None and backend is
        'torch' or 'auto', chooses 'cuda' if available, else 'cpu'.
    stop : {'convergence', 'boundary'}
        'convergence' -> Stop when no cue can improve the semantic
            approximation (all c_prod values <= tol).
        'boundary' -> Also stop when a cue ending with '#' is selected.
    tol : float
        Tolerance for the convergence check. The algorithm stops when all
        c_prod values are <= tol. Default is 0.0.
    """
    if apply_vmat and vmat is None:
        raise ValueError('vmat must be provided when apply_vmat is True.')
    if stop not in ('convergence', 'boundary'):
        raise ValueError(f'Unknown stop "{stop}". Use "convergence" or "boundary".')
    # Determine which backend to use
    use_torch = False
    if backend == 'torch':
        if torch is None:
            raise ImportError('PyTorch is not installed. Install it to use the "torch" backend.')
        use_torch = True
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif backend == 'auto':
        if torch is not None and (device == 'cuda' or (device is None and torch.cuda.is_available())):
            use_torch = True
            device = device or 'cuda'
    elif backend != 'numpy':
        raise ValueError(f'Unknown backend "{backend}". Use "numpy", "torch", or "auto".')

    if use_torch:
        return _produce_torch(gold, cmat, fmat, gmat, vmat, word, roundby, max_attempt, positive, apply_vmat, device, stop, tol)
    else:
        return _produce_numpy(gold, cmat, fmat, gmat, vmat, word, roundby, max_attempt, positive, apply_vmat, stop, tol)

def _produce_numpy (gold, cmat, fmat, gmat, vmat, word, roundby, max_attempt, positive, apply_vmat, stop, tol):
    """NumPy implementation of produce (standalone version)."""
    if not isinstance(gold, np.ndarray):
        gold = np.array(gold, dtype=np.float32)
    else:
        gold = gold.astype(np.float32)
    p = -1
    xs = []
    vecs = []
    cues_values = cmat.cues.values
    fmat_values = fmat.values.astype(np.float32)
    gmat_values = gmat.values.astype(np.float32)
    if apply_vmat:
        vmat_values = vmat.values.astype(np.float32)
    c_comp = np.zeros(cues_values.size, dtype=np.float32)
    for i in range(max_attempt):
        s0 = np.matmul(c_comp, fmat_values)
        if positive:
            s0[s0<0] = 0
        s = gold - s0
        if apply_vmat:
            vmat_row = vmat_values[p]
            g = gmat_values * vmat_row
        else:
            g = gmat_values
        c_prod = np.matmul(s, g)
        if (c_prod<=tol).all():
            break
        else:
            p = np.argmax(c_prod)
            c_comp[p] = c_comp[p] + 1
            xs.append(cues_values[p])
            vecs.append(c_prod)
        if stop == 'boundary':
            is_unigram_onset = len(xs)==1 and len(xs[0])==1 and xs[0]=='#'
            is_end = (xs[-1][-1]=='#') and (not is_unigram_onset)
            if is_end:
                break
        if i==(max_attempt-1):
            print('The maximum number of iterations ({:d}) reached.'.format(max_attempt))
    vecs = [v.round(roundby) for v in vecs]
    df = pd.DataFrame(vecs).rename(columns={ i:j for i,j in enumerate(cues_values) })
    hdr = pd.Series(xs).to_frame(name='Selected')
    df = pd.concat([hdr, df], axis=1)
    if word:
        from .ldl import concat_cues
        df = concat_cues(df.Selected)
    return df

def _produce_torch (gold, cmat, fmat, gmat, vmat, word, roundby, max_attempt, positive, apply_vmat, device, stop, tol):
    """PyTorch implementation of produce with GPU support (standalone version)."""
    if not isinstance(gold, np.ndarray):
        gold = np.array(gold)

    # Move data to torch tensors on specified device
    gold_tensor = torch.as_tensor(gold, dtype=torch.float32, device=device)
    fmat_tensor = torch.as_tensor(fmat.values, dtype=torch.float32, device=device)
    gmat_tensor = torch.as_tensor(gmat.values, dtype=torch.float32, device=device)

    if apply_vmat:
        vmat_tensor = torch.as_tensor(vmat.values, dtype=torch.float32, device=device)

    cues_values = cmat.cues.values
    n_cues = cues_values.size

    # Pre-allocate tensors on GPU
    c_comp = torch.zeros(n_cues, dtype=torch.float32, device=device)
    xs_indices = torch.zeros(max_attempt, dtype=torch.long, device=device)
    vecs_gpu = torch.zeros((max_attempt, n_cues), dtype=torch.float32, device=device)

    p_tensor = torch.tensor(-1, dtype=torch.long, device=device)
    actual_iterations = 0

    for i in range(max_attempt):
        s0 = torch.matmul(c_comp, fmat_tensor)
        if positive:
            s0 = torch.clamp(s0, min=0)
        s = gold_tensor - s0

        if apply_vmat:
            vmat_row = vmat_tensor[p_tensor]
            g = gmat_tensor * vmat_row
        else:
            g = gmat_tensor

        c_prod = torch.matmul(s, g)

        if (c_prod <= tol).all():
            break
        else:
            p_tensor = torch.argmax(c_prod)
            c_comp[p_tensor] += 1

            xs_indices[i] = p_tensor
            vecs_gpu[i] = c_prod
            actual_iterations = i + 1

        if stop == 'boundary':
            p_cpu = p_tensor.item()
            selected_cue = cues_values[p_cpu]
            is_unigram_onset = actual_iterations == 1 and len(selected_cue) == 1 and selected_cue == '#'
            is_end = (selected_cue[-1] == '#') and (not is_unigram_onset)
            if is_end:
                break
        if i == (max_attempt - 1):
            print('The maximum number of iterations ({:d}) reached.'.format(max_attempt))

    # Single GPU-to-CPU transfer at the end
    xs_indices_cpu = xs_indices[:actual_iterations].cpu().numpy()
    vecs_cpu = vecs_gpu[:actual_iterations].cpu().numpy()

    xs = [cues_values[idx] for idx in xs_indices_cpu]

    vecs = [v.round(roundby) for v in vecs_cpu]
    df = pd.DataFrame(vecs).rename(columns={ i:j for i,j in enumerate(cues_values) })
    hdr = pd.Series(xs).to_frame(name='Selected')
    df = pd.concat([hdr, df], axis=1)
    if word:
        from .ldl import concat_cues
        df = concat_cues(df.Selected)
    return df

def incremental_learning (rows, cue_matrix, out_matrix, learning_rate=0.1, weight_matrix=None, return_intermediate_weights=False, backend='numpy', device=None, batch_size=1, rows_by_index=False, nlms=True):
    """
    Incremental learning with optional GPU acceleration and batch processing.

    Parameters
    ----------
    rows : list-like
        Row labels (if rows_by_index=False) or integer indices (if
        rows_by_index=True) specifying the order of learning events.
    cue_matrix : xarray.DataArray
    out_matrix : xarray.DataArray
    learning_rate : float
    weight_matrix : xarray.DataArray or None
    return_intermediate_weights : bool
    backend : {'numpy', 'torch', 'auto'}
        'numpy' -> original NumPy/xarray CPU implementation.
        'torch' -> PyTorch implementation (CPU/GPU depending on 'device').
        'auto'  -> try torch+CUDA if available, else fall back to NumPy.
    device : str or None
        For the torch backend: 'cuda', 'cpu', etc. If None and backend is
        'torch' or 'auto', chooses 'cuda' if available, else 'cpu'.
    batch_size : int
        Number of rows to process in each batch. Default is 1, which gives
        the theoretically "true" incremental learning result where each row's
        prediction uses the weight matrix updated by all previous rows.
        Larger batch sizes are an approximation that trades theoretical
        fidelity for computational speed (especially beneficial for GPU).
        With batch_size > 1, all rows within a batch share the same weight
        matrix for their predictions before the weight update is applied.
    rows_by_index : bool
        If False (default), `rows` contains row labels and selection is done
        via .loc[]. If True, `rows` contains integer indices and selection is
        done via positional indexing. Using indices can be useful when you
        want to specify learning events by position (e.g., allowing the same
        row to appear multiple times).
    nlms : bool
        If True (default), use the Normalized Least Mean Squares (NLMS)
        algorithm which normalizes the weight update by the squared norm of
        the cue vector. This provides more stable learning rates across
        different input magnitudes. If False, use the standard LMS algorithm.
    """
    if backend == 'numpy':
        return _incremental_learning_numpy(
            rows,
            cue_matrix,
            out_matrix,
            learning_rate=learning_rate,
            weight_matrix=weight_matrix,
            return_intermediate_weights=return_intermediate_weights,
            batch_size=batch_size,
            rows_by_index=rows_by_index,
            nlms=nlms,
        )

    if backend == 'torch':
        return _incremental_learning_torch(
            rows,
            cue_matrix,
            out_matrix,
            learning_rate=learning_rate,
            weight_matrix=weight_matrix,
            return_intermediate_weights=return_intermediate_weights,
            device=device,
            batch_size=batch_size,
            rows_by_index=rows_by_index,
            nlms=nlms,
        )

    if backend == 'auto':
        # Prefer torch+CUDA if possible
        if (torch is not None) and (device == 'cuda' or (device is None and torch.cuda.is_available())):
            return _incremental_learning_torch(
                rows,
                cue_matrix,
                out_matrix,
                learning_rate=learning_rate,
                weight_matrix=weight_matrix,
                return_intermediate_weights=return_intermediate_weights,
                device=device or 'cuda',
                batch_size=batch_size,
                rows_by_index=rows_by_index,
                nlms=nlms,
            )
        else:
            return _incremental_learning_numpy(
                rows,
                cue_matrix,
                out_matrix,
                learning_rate=learning_rate,
                weight_matrix=weight_matrix,
                return_intermediate_weights=return_intermediate_weights,
                batch_size=batch_size,
                rows_by_index=rows_by_index,
                nlms=nlms,
            )

    raise ValueError(f'Unknown backend "{backend}". Use "numpy", "torch", or "auto".')

def _incremental_learning_numpy (rows, cue_matrix, out_matrix, learning_rate=0.1, weight_matrix=None, return_intermediate_weights=False, batch_size=1, rows_by_index=False, nlms=True):
    if weight_matrix is None:
        _dims = (cue_matrix.dims[1], out_matrix.dims[1])
        _coords = {_dims[0]: cue_matrix[_dims[0]].values.tolist(), _dims[1]: out_matrix[_dims[1]].values.tolist()}
        weight_matrix = np.zeros((cue_matrix.shape[1], out_matrix.shape[1]))
        weight_matrix = xr.DataArray(weight_matrix, dims=_dims, coords=_coords)
    makesure_xarray(cue_matrix, out_matrix, weight_matrix)
    if return_intermediate_weights:
        weight_mats = [weight_matrix]

    rows = list(rows)
    n_rows = len(rows)
    n_batches = (n_rows + batch_size - 1) // batch_size  # ceiling division

    for batch_idx in tqdm(range(n_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_rows)
        batch_rows = rows[start_idx:end_idx]

        if rows_by_index:
            cvec = cue_matrix[batch_rows, :]
            ovec = out_matrix[batch_rows, :]
        else:
            cvec = cue_matrix.loc[batch_rows, :]
            ovec = out_matrix.loc[batch_rows, :]
        weight_matrix = update_weight_matrix(weight_matrix, cvec, ovec, learning_rate, nlms=nlms)

        if return_intermediate_weights:
            weight_mats = weight_mats + [weight_matrix]

    if return_intermediate_weights:
        res = weight_mats
    else:
        res = weight_matrix
    return res

def _incremental_learning_torch (rows, cue_matrix, out_matrix, learning_rate=0.1, weight_matrix=None, return_intermediate_weights=False, device=None, batch_size=1, rows_by_index=False, nlms=True):

    if torch is None:
        raise ImportError('PyTorch is not installed. Install it to use the "torch" backend.')

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    makesure_xarray(cue_matrix, out_matrix)

    word_dim = cue_matrix.dims[0]
    cue_dim = cue_matrix.dims[1]
    out_dim = out_matrix.dims[1]

    words = cue_matrix[word_dim].values
    cues = cue_matrix[cue_dim].values
    outs = out_matrix[out_dim].values

    # Map row labels to integer indices (only needed when rows_by_index=False):
    if not rows_by_index:
        word_index = {w: idx for idx, w in enumerate(words)}

    # Move data to torch:
    cue_tensor = torch.as_tensor(cue_matrix.values, dtype=torch.float32, device=device)
    out_tensor = torch.as_tensor(out_matrix.values, dtype=torch.float32, device=device)

    if weight_matrix is None:
        weight_tensor = torch.zeros((cue_tensor.shape[1], out_tensor.shape[1]), dtype=torch.float32, device=device)
    else:
        makesure_xarray(weight_matrix)
        weight_tensor = torch.as_tensor(weight_matrix.values, dtype=torch.float32, device=device)

    if return_intermediate_weights:
        weight_mats = [
            xr.DataArray(
                weight_tensor.detach().cpu().numpy(),
                dims=(cue_dim, out_dim),
                coords={cue_dim: cues, out_dim: outs},
            )
        ]

    lr = learning_rate
    rows = list(rows)
    n_rows = len(rows)
    n_batches = (n_rows + batch_size - 1) // batch_size  # ceiling division

    for batch_idx in tqdm(range(n_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_rows)
        batch_rows = rows[start_idx:end_idx]

        # Get indices for this batch
        if rows_by_index:
            batch_indices = batch_rows
        else:
            batch_indices = [word_index[row_label] for row_label in batch_rows]

        cvec = cue_tensor[batch_indices, :]  # (batch, n_cues)
        ovec = out_tensor[batch_indices, :]  # (batch, n_out)

        pred = cvec @ weight_tensor           # (batch, n_out)
        dlt = ovec - pred                     # (batch, n_out)
        grad = cvec.transpose(0, 1) @ dlt     # (n_cues, n_out)

        if nlms:
            # Normalize by squared norm of cue vector (NLMS algorithm) for numerical stability
            cue_norm_sq = torch.sum(cvec ** 2) + 1e-8
            weight_tensor = weight_tensor + lr * grad / cue_norm_sq
        else:
            weight_tensor = weight_tensor + lr * grad

        if return_intermediate_weights:
            weight_mats.append(
                xr.DataArray(
                    weight_tensor.detach().cpu().numpy(),
                    dims=(cue_dim, out_dim),
                    coords={cue_dim: cues, out_dim: outs},
                )
            )

    final_weight = xr.DataArray(
        weight_tensor.detach().cpu().numpy(),
        dims=(cue_dim, out_dim),
        coords={cue_dim: cues, out_dim: outs},
    )

    if return_intermediate_weights:
        return weight_mats
    else:
        return final_weight

def incremental_learning_byind (events, cue_matrix, out_matrix, learning_rate=0.1):
    """
    Deprecated: Use incremental_learning(..., rows_by_index=True) instead.

    This function is kept for backward compatibility but will be removed in a
    future version.
    """
    warnings.warn(
        'incremental_learning_byind is deprecated. '
        'Use incremental_learning(..., rows_by_index=True) instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    return incremental_learning(
        rows=events,
        cue_matrix=cue_matrix,
        out_matrix=out_matrix,
        learning_rate=learning_rate,
        rows_by_index=True,
    )

def update_weight_matrix (weight_matrix, cue_vector, out_vector, learning_rate=0.1, nlms=True):
    dlt = delta_weight_matrix(weight_matrix, cue_vector, out_vector, learning_rate, nlms=nlms)
    weight_matrix = weight_matrix + dlt
    return weight_matrix

def delta_weight_matrix (weight_matrix, cue_vector, out_vector, learning_rate=0.1, nlms=True):
    weight_matrix, cue_vector, out_vector = to_nparray(weight_matrix, cue_vector, out_vector)
    dlt = out_vector - matmul(cue_vector, weight_matrix)
    if nlms:
        # Normalize by squared norm of cue vector (NLMS algorithm) for numerical stability
        cue_norm_sq = np.sum(cue_vector ** 2) + 1e-8  # Add epsilon to avoid division by zero
        dlt = matmul(cue_vector.T, dlt) * learning_rate / cue_norm_sq
    else:
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

def save_mat (xarr, path):
    df = xarr.to_pandas()
    _indname = not (df.index.name is None)
    _colname = not (df.columns.name is None)
    if _indname and _colname:
        df.index.name = df.index.name + '/' + df.columns.name
        df.columns.name = None
    df.to_csv(path, sep='\t', index=True, header=True)
    return None

def load_mat (path):
    df = pd.read_csv(path, sep='\t', index_col=0, header=0, na_filter=False, quoting=3)
    df.index.name, df.columns.name = df.index.name.split('/')
    xarr = xr.DataArray(df)
    return xarr
