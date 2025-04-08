====================
Incremental learning
====================


---------------------------------------
Incremental learning by a list of words
---------------------------------------
Weight matrices in LDL (i.e., `\mathbf{F}` and `\mathbf{G}`) can be estimated also step by step, which is called the *incremental* learning. For a simple example, suppose we have only two words "a" and "an" in the lexicon and we encounter them in the order of "a", "a", "an", and "a". This can be done by discriminative_lexicon_model.mapping.incremental_learning. The first argument of the function is a series of learning events.

.. code-block:: python

    >>> import xarray as xr
    >>> import discriminative_lexicon_model.mapping as pm
    >>> cmat = pm.gen_cmat(['a', 'an'], gram=2)
    >>> print(cmat)

    <xarray.DataArray (word: 2, cues: 4)> Size: 64B
    array([[1, 1, 0, 0],
           [1, 0, 1, 1]])
    Coordinates:
      * word     (word) <U2 16B 'a' 'an'
      * cues     (cues) <U2 32B '#a' 'a#' 'an' 'n#'

    >>> smat = xr.DataArray([[0.9, -0.2, 0.1], [0.1, 0.9, -0.2]], dims=('word','semantics'), coords={'word':['a','an'], 'semantics':['S1','S2','S3']})
    >>> print(smat)

    <xarray.DataArray (word: 2, semantics: 3)> Size: 48B
    array([[ 0.9, -0.2,  0.1],
           [ 0.1,  0.9, -0.2]])
    Coordinates:
      * word       (word) <U2 16B 'a' 'an'
      * semantics  (semantics) <U2 24B 'S1' 'S2' 'S3'

    >>> fmat = pm.incremental_learning(['a', 'a', 'an', 'a'], cmat, smat)
    >>> print(fmat)

    <xarray.DataArray (cues: 4, semantics: 3)> Size: 96B
    array([[ 0.21402,  0.03544,  0.00478],
           [ 0.22022, -0.05816,  0.02658],
           [-0.0062 ,  0.0936 , -0.0218 ],
           [-0.0062 ,  0.0936 , -0.0218 ]])
    Coordinates:
      * cues       (cues) <U2 32B '#a' 'a#' 'an' 'n#'
      * semantics  (semantics) <U2 24B 'S1' 'S2' 'S3'

Note that the `\mathbf{S}` matrix is set up, so that the first dimension "S1" is strongly correlated with "a" while "S2" is correlated "an". In other words, you can conceptually interpret "S1" as the core meaning of "a" and "S2" as that of "an". In the weight matrix (i.e., `\mathbf{F}`), the first two rows, namely the cues "#a" and "a#" are strongly correlated with the first column, namely "S1". The last two rows, namely the cues "an" and "n#" are strongly correlated with the second column, namely "S2". The associations of "an" and "n#" to "S2" are numerically smaller than those of "#a" and "a#" to "S1", because "an" occurs only once while "a" occurs three times in the learning events.

As shown below, after a sufficient number of learning events, the estimates approximate those by the *endstate* learning.

.. code-block:: python

    >>> import pandas as pd
    >>> words = pd.Series(['a', 'an']).sample(1000, replace=True, random_state=518).tolist()
    >>> fmat_inc = pm.incremental_learning(words, cmat, smat)
    >>> fmat_end = pm.gen_fmat(cmat=cmat, smat=smat)
    >>> print(fmat_inc)

    <xarray.DataArray (cues: 4, semantics: 3)> Size: 96B
    array([[ 3.80000000e-01,  1.00000000e-01, -5.65948715e-19],
           [ 5.20000000e-01, -3.00000000e-01,  1.00000000e-01],
           [-1.40000000e-01,  4.00000000e-01, -1.00000000e-01],
           [-1.40000000e-01,  4.00000000e-01, -1.00000000e-01]])
    Coordinates:
      * cues       (cues) <U2 32B '#a' 'a#' 'an' 'n#'
      * semantics  (semantics) <U2 24B 'S1' 'S2' 'S3'

    >>> print(fmat_end)
    <xarray.DataArray (cues: 4, semantics: 3)> Size: 96B
    array([[ 3.80000000e-01,  1.00000000e-01, -2.77555756e-17],
           [ 5.20000000e-01, -3.00000000e-01,  1.00000000e-01],
           [-1.40000000e-01,  4.00000000e-01, -1.00000000e-01],
           [-1.40000000e-01,  4.00000000e-01, -1.00000000e-01]])
    Coordinates:
      * cues       (cues) <U2 32B '#a' 'a#' 'an' 'n#'
      * semantics  (semantics) <U2 24B 'S1' 'S2' 'S3'

    >>> print(fmat_inc.round(10).identical(fmat_end.round(10)))
    True

Note that the order of learning events does matter in the incremental learning. Compare the following two examples.

.. code-block:: python

    >>> import numpy as np
    words_a_first = np.repeat(['a', 'an'], [10, 10])
    words_an_first = np.repeat(['an', 'a'], [10, 10])
    fmat_a_first = pm.incremental_learning(words_a_first, cmat, smat)
    fmat_an_first = pm.incremental_learning(words_an_first, cmat, smat)
    print(fmat_a_first)
    <xarray.DataArray (cues: 4, semantics: 3)> Size: 96B
    array([[ 0.30396166,  0.23117687, -0.03460906],
           [ 0.40168162, -0.08926258,  0.04463129],
           [-0.09771995,  0.32043945, -0.07924035],
           [-0.09771995,  0.32043945, -0.07924035]])
    Coordinates:
      * cues       (cues) <U2 32B '#a' 'a#' 'an' 'n#'
      * semantics  (semantics) <U2 24B 'S1' 'S2' 'S3'

    print(fmat_an_first)
    <xarray.DataArray (cues: 4, semantics: 3)> Size: 96B
    array([[ 0.41961651,  0.07215146,  0.0087615 ],
           [ 0.38722476, -0.21937428,  0.073545  ],
           [ 0.03239175,  0.29152574, -0.0647835 ],
           [ 0.03239175,  0.29152574, -0.0647835 ]])
    Coordinates:
      * cues       (cues) <U2 32B '#a' 'a#' 'an' 'n#'
      * semantics  (semantics) <U2 24B 'S1' 'S2' 'S3'

In the first case, where "a" is encountered first for 100 times before "an" is encountered 100 times consecutively, the estimated associations are "biased" towards to "an". This can be seen, for example, in the cell value of the first row and the second column, namely the association strength between "#a" and "S2". Note that the equilibrium of this association is 0.10 (see the example above for "fmat_end"). Since "an" is encountered many times more "recently", such recent learning events have bigger effects.

In contrast, in the latter case, where "an" is encountered first for 100 times before "a" is encountered 100 times, the association from "#a" to "S1" is much bigger than that from "#a" to "S2". Note that the equilibrium of the association from "#a" to "S1" is 0.38 (from "fmat_end" in the example above). Since "a" is encountered many times towards the end of learning, the weights are biased towards "a".


----------------------------------------------
Incremental learning by a list of word indices
----------------------------------------------
Learning events (i.e., which words to encounter) can be specified by indices of words as well. This can be useful when the `\mathbf{C}` and/or `\mathbf{S}` matrices contain duplicated word labels. Duplicated rows can be an issue when word tokens are involved. Consider the following example:

.. code-block:: python

    >>> import xarray as xr
    >>> import discriminative_lexicon_model.mapping as pm
    >>> cmat = pm.gen_cmat(['a', 'an', 'an'], gram=2)
    >>> smat = xr.DataArray([[0.9, -0.2, 0.1], [0.1, 0.9, -0.2], [0.2, 0.8, -0.1]], dims=('word','semantics'), coords={'word':['a','an','an'], 'semantics':['S1','S2','S3']})
    >>> print(cmat)

    <xarray.DataArray (word: 3, cues: 4)> Size: 96B
    array([[1, 1, 0, 0],
           [1, 0, 1, 1],
           [1, 0, 1, 1]])
    Coordinates:
      * word     (word) <U2 24B 'a' 'an' 'an'
      * cues     (cues) <U2 32B '#a' 'a#' 'an' 'n#'

    >>> print(smat)
    <xarray.DataArray (word: 3, semantics: 3)> Size: 72B
    array([[ 0.9, -0.2,  0.1],
           [ 0.1,  0.9, -0.2],
           [ 0.2,  0.8, -0.1]])
    Coordinates:
      * word       (word) <U2 24B 'a' 'an' 'an'
      * semantics  (semantics) <U2 24B 'S1' 'S2' 'S3'

Note that the word type "an" has two rows. Its form vectors are the same (i.e., the second and third rows of the `\mathbf{C}` matrix), while its semantic vectors are slightly different (i.e., the second and third rows of the `\mathbf{S}` matrix). You can view the different semantic vectors as different meanings of the same word in different contexts. In such a case like this, specifying learning events by a list of words like below would raise "InvalidIndexError", because the function cannot determine which semantic vector to use for "an" in this case.

.. code-block:: python

    >>> fmat = pm.incremental_learning(['a', 'a', 'an', 'a'], cmat, smat)
    >>> # This raises an error.

Instead, you need to specify learning events in terms of indices of the words. For this purpose, discriminative_lexicon_model.mapping.incremental_learning_byind can be used:

.. code-block:: python

    >>> events = [0, 0, 1, 2, 2] # 'a', 'a', 'an' (2nd row), 'an' (3rd row), 'an' (3rd row)
    >>> fmat = pm.incremental_learning_byind(events, cmat, smat)
    >>> print(fmat)

    <xarray.DataArray (cues: 4, semantics: 3)> Size: 96B
    array([[ 0.165422,  0.151984, -0.012742],
           [ 0.162   , -0.036   ,  0.018   ],
           [ 0.003422,  0.187984, -0.030742],
           [ 0.003422,  0.187984, -0.030742]])
    Coordinates:
      * cues       (cues) <U2 32B '#a' 'a#' 'an' 'n#'
      * semantics  (semantics) <U2 24B 'S1' 'S2' 'S3'

