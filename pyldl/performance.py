import pandas as pd
import numpy as np
import scipy.spatial.distance as spd

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

