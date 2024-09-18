import numpy as np
import pandas as pd
import xarray as xr
import scipy.spatial.distance as spd

def accuracy (hat, mat, distance=False):
    pred = predict_df(hat, mat, max_guess=1, distance=distance)
    acc = pred.acc.sum() / len(pred)
    return acc

def predict_df (hat, mat, max_guess=1, distance=False, method='cosine'):
    if not isinstance(max_guess, int): raise TypeError('"max_guess" must be integer')
    coss = distance_matrix(pred=hat, gold=mat, method=method).values
    if distance:
        pos1 = [np.argmin(coss, axis=1)]
        sign = 1
    else:
        coss = 1 - coss
        pos1 = [np.argmax(coss, axis=1)]
        sign = -1

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
    wrds = pd.DataFrame({'Word':hat.word.values})
    dddd = pd.concat([wrds,prds,hits], axis=1)
    return dddd

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

def predict (word, hat, mat, distance=False):
    hat = np.tile(hat.loc[word,:], (1,1))
    coss = spd.cdist(np.array(hat), np.array(mat), 'cosine')
    if distance:
        sign = 1
    else:
        coss = 1 - coss
        sign = -1
    coss = coss[0,:]
    pred = mat.word.values[np.argsort(sign*coss)]
    return pd.Series(pred)

