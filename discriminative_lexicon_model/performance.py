import numpy as np
import pandas as pd
import xarray as xr
import scipy.spatial.distance as spd

def accuracy (*, pred, gold, method='correlation'):
    pred = predict_df(pred=pred, gold=gold, n=1, method=method)
    acc = pred.Correct.sum() / len(pred)
    return acc

def predict_df (*, pred, gold, n=1, method='correlation'):
    """
    Constructs a dataframe of predictions.

    Parameters
    ----------
    pred : xarray.core.dataarray.DataArray
        A matrix of predictions. It is usually a C-hat or S-hat matrix.
    gold : xarray.core.dataarray.DataArray
        A matrix of gold-standard vectors. It is usually a C or S matrix.
    n : int or None
        The number of predictions to make for each word. When n=1, the first prediction for each word will be produced. When n=2, the first and second predictions for each word will be included in the output dataframe. When n=None, as many predictions as possible will be produced.
    method : str
        Which method to use to calculate distance/similarity. It must be "correlation", "cosine" (for cosine similarity), and "euclidean" (for euclidean distance).

    Returns
    -------
    df : pandas.core.frame.DataFrame
        A dataframe of a model's predictions.

    Examples
    --------
    >>> import discriminative_lexicon_model as dlm
    >>> import pandas as pd
    >>> words = ['cat','rat','hat']
    >>> sems = pd.DataFrame({'<animate>':[1,1,0], '<object>':[0,0,1], '<predator>':[1,0,0]}, index=words)
    >>> mdl = dlm.ldl.LDL()
    >>> mdl.gen_cmat(words)
    >>> mdl.gen_smat(sems)
    >>> mdl.gen_gmat()
    >>> mdl.gen_chat()
    >>> dlm.performance.predict_df(pred=mdl.chat, gold=mdl.cmat, n=2, method='correlation')
      Word Pred1 Pred2  Correct1  Correct2
    0  cat   cat   hat      True     False
    1  rat   rat   hat      True     False
    2  hat   hat   cat      True     False
    """
    if not (method in ['correlation', 'cosine', 'euclidean']):
        raise ValueError('"method" must be "correlation", "cosine", or "euclidean".')
    if not (n is None):
        if not isinstance(n, int):
            raise TypeError('"n" must be integer or None.')
        if not (n>0):
            raise ValueError('"n" must be a positive integer.')
    n = pred.shape[0] if n is None else n
    
    dist = distance_matrix(pred=pred, gold=gold, method=method).values
    dist = dist if method=='euclidean' else 1-dist
    inds = dist.argsort(axis=1) if method=='euclidean' else (-dist).argsort(axis=1)
    inds = inds[:,:n]
    
    prds = np.apply_along_axis(lambda x: gold.word.values[x], 1, inds)
    hits = np.array([ prds[i,:]==j for i,j in zip(range(prds.shape[0]), gold.word.values) ])
    
    clms = ['Pred'] if prds.shape[1]==1 else [ 'Pred{:d}'.format(i) for i in range(1, prds.shape[1]+1) ]
    prds = pd.DataFrame(prds, columns=clms)
    clms = ['Correct'] if hits.shape[1]==1 else [ 'Correct{:d}'.format(i) for i in range(1, hits.shape[1]+1) ]
    hits = pd.DataFrame(hits, columns=clms)
    wrds = pd.DataFrame({'Word':gold.word.values})
    df = pd.concat([wrds, prds, hits], axis=1)
    return df

def distance_matrix (*, pred, gold, method='cosine'):
    """
    Constructs a distance matrix between a matrix of predictions and that of
    gold-standards. If similarity is of more interest than distance (e.g.,
    correlation / cosine similarity), subtract the return value of this
    function from 1.

    Parameters
    ----------
    pred : xarray.core.dataarray.DataArray
        A prediction matrix, which is usually either a C-hat matrix or a S-hat
        matrix.
    gold : xarray.core.dataarray.DataArray
        A gold-standard matrix, which is usually either a C matrix or a S
        matrix.

    Returns
    -------
    dist : xarray.core.dataarray.DataArray
        A 2-d array of the shape m x n, where m represents the number of rows
        in "pred" and n represents the number of rows in "gold". The cell value
        of the i-th row and the j-th column is the distance between the vector
        of the i-th row of "pred" and the vector of the j-th row of "gold". If
        similarity (e.g., correlation / cosine similarity) is of more interest
        than distance, subtract "dist" from 1 (i.e., 1 - dist).
    """
    dist = spd.cdist(pred.values, gold.values, method)
    new_coords = {'pred':pred[pred.dims[0]].values,
                  'gold':gold[gold.dims[0]].values}
    dist = xr.DataArray(dist, dims=('pred','gold'), coords=new_coords)
    return dist

