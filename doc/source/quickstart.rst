==========
Quickstart
==========

In the following, I will describe the basic usage of pyldl. For more details about Linear Discriminative Learning, see the section :ref:`Linear Discriminative Learning`.

Installation
============

*pyldl* is not available on PyPI yet. So, you need to clone it before installing it locally.

.. code:: bash

    git clone https://github.com/msaito8623/pyldl
    pip install -e /path/to/the/repo



Train an Linear Discriminative Learning model
=============================================


C-matrix
--------

You can create a C-matrix from a list of words by using pyldl.mapping.gen_cmat.

.. code-block:: python

    >>> import pyldl.mapping as pmap
    >>> words = ['walk','walked','walks']
    >>> cmat  = pmap.gen_cmat(words)
    >>> cmat
    <xarray.DataArray (word: 3, cues: 9)>
    array([[ True,  True, False, False, False,  True, False, False,  True],
           [ True,  True,  True,  True, False, False,  True, False,  True],
           [ True,  True, False, False,  True, False, False,  True,  True]])
    Coordinates:
      * word     (word) <U6 'walk' 'walked' 'walks'
      * cues     (cues) <U3 '#wa' 'alk' 'ed#' 'ked' 'ks#' 'lk#' 'lke' 'lks' 'wal'


S-matrix
--------

An S-matrix by simulated semantic vectors [1]_ can be produced from a pandas dataframe that contains morphological information. This can be achieved in pyldl with pyldl.mapping.gen_smat_sim.


.. code-block:: python

    >>> import pandas as pd
    >>> infl = pd.DataFrame({'Word':['walk','walked','walks'], 'Lemma':['walk','walk','walk'], 'Tense':['PRES','PAST','PRES']})
    >>> smat = pmap.gen_smat_sim(infl, dim_size=5)
    >>> smat.round(2)
    <xarray.DataArray (word: 3, semantics: 5)>
    array([[ 0.75,  1.25,  0.39, -4.41, -0.12],
           [-1.68,  0.6 , -0.  , -3.55, -2.23],
           [-2.77,  0.71, -0.48, -2.76,  0.15]])
    Coordinates:
      * word       (word) <U6 'walk' 'walked' 'walks'
      * semantics  (semantics) <U4 'S000' 'S001' 'S002' 'S003' 'S004'


F-matrix
--------

An F-matrix can be obtained with pyldl.mapping.gen_fmat.

.. code-block:: python

    >>> fmat = pmap.gen_fmat(cmat, smat)
    >>> fmat.round(2)
    <xarray.DataArray (cues: 9, semantics: 5)>
    array([[-0.  , -0.  , -0.  , -0.  , -0.  ],
           [-0.  , -0.  , -0.  , -0.  , -0.  ],
           [-0.56,  0.2 , -0.  , -1.18, -0.74],
           [-0.56,  0.2 , -0.  , -1.18, -0.74],
           [-1.39,  0.35, -0.24, -1.38,  0.07],
           [ 0.75,  1.25,  0.39, -4.41, -0.12],
           [-0.56,  0.2 , -0.  , -1.18, -0.74],
           [-1.39,  0.35, -0.24, -1.38,  0.07],
           [-0.  , -0.  , -0.  , -0.  , -0.  ]])
    Coordinates:
      * cues       (cues) <U3 '#wa' 'alk' 'ed#' 'ked' 'ks#' 'lk#' 'lke' 'lks' 'wal'
      * semantics  (semantics) <U4 'S000' 'S001' 'S002' 'S003' 'S004'


G-matrix
--------

A G-matrix can be obtained with pyldl.mapping.gen_gmat.

.. code-block:: python

    >>> gmat = pmap.gen_gmat(cmat, smat)
    >>> gmat.round(2)
    <xarray.DataArray (semantics: 5, cues: 9)>
    array([[-0.11, -0.11, -0.03, -0.03, -0.27,  0.19, -0.03, -0.27, -0.11],
           [ 0.06,  0.06, -0.06, -0.06,  0.05,  0.08, -0.06,  0.05,  0.06],
           [-0.01, -0.01,  0.03,  0.03, -0.08,  0.04,  0.03, -0.08, -0.01],
           [-0.23, -0.23, -0.01, -0.01, -0.05, -0.17, -0.01, -0.05, -0.23],
           [ 0.02,  0.02, -0.43, -0.43,  0.29,  0.15, -0.43,  0.29,  0.02]])
    Coordinates:
      * semantics  (semantics) <U4 'S000' 'S001' 'S002' 'S003' 'S004'
      * cues       (cues) <U3 '#wa' 'alk' 'ed#' 'ked' 'ks#' 'lk#' 'lke' 'lks' 'wal'


S-hat-matrix
------------

An S-hat-matrix (:math:`\mathbf{\hat{S}}`), predicted semantic vectors based on forms, can be obtained with pyldl.mapping.gen_shat. You can produce an S-hat-matrix from a C-matrix and an F-matrix or from a C-matrix and an S-matrix without producing an F-matrix yourself.

.. code-block:: python

    >>> shat = pmap.gen_shat(cmat=cmat, fmat=fmat)
    >>> shat.round(2)
    <xarray.DataArray (word: 3, semantics: 5)>
    array([[ 0.75,  1.25,  0.39, -4.41, -0.12],
           [-1.68,  0.6 , -0.  , -3.55, -2.23],
           [-2.77,  0.71, -0.48, -2.76,  0.15]])
    Coordinates:
      * word       (word) <U6 'walk' 'walked' 'walks'
      * semantics  (semantics) <U4 'S000' 'S001' 'S002' 'S003' 'S004'

.. code-block:: python

    >>> shat = pmap.gen_shat(cmat=cmat, smat=smat)
    >>> shat.round(2)
    <xarray.DataArray (word: 3, semantics: 5)>
    array([[ 0.75,  1.25,  0.39, -4.41, -0.12],
           [-1.68,  0.6 , -0.  , -3.55, -2.23],
           [-2.77,  0.71, -0.48, -2.76,  0.15]])
    Coordinates:
      * word       (word) <U6 'walk' 'walked' 'walks'
      * semantics  (semantics) <U4 'S000' 'S001' 'S002' 'S003' 'S004'


C-hat-matrix
------------

A C-hat-matrix (:math:`\mathbf{\hat{C}}`), predicted form vectors based on semantics, can be obtained with pyldl.mapping.gen_chat. You can produce a C-hat-matrix from an S-matrix and a G-matrix or from an S-matrix and a C-matrix without producing a G-matrix yourself.

.. code-block:: python

    >>> chat = pmap.gen_chat(smat=smat, gmat=gmat)
    >>> chat.round(2)
    <xarray.DataArray (word: 3, cues: 9)>
    array([[ 1.,  1.,  0.,  0., -0.,  1.,  0., -0.,  1.],
           [ 1.,  1.,  1.,  1.,  0., -0.,  1.,  0.,  1.],
           [ 1.,  1., -0., -0.,  1., -0., -0.,  1.,  1.]])
    Coordinates:
      * word       (word) <U6 'walk' 'walked' 'walks'
      * cues       (cues) <U3 '#wa' 'alk' 'ed#' 'ked' 'ks#' 'lk#' 'lke' 'lks' 'wal'

.. code-block:: python

    >>> chat = pmap.gen_chat(smat=smat, cmat=cmat)
    >>> chat.round(2)
    <xarray.DataArray (word: 3, cues: 9)>
    array([[ 1.,  1.,  0.,  0., -0.,  1.,  0., -0.,  1.],
           [ 1.,  1.,  1.,  1.,  0., -0.,  1.,  0.,  1.],
           [ 1.,  1., -0., -0.,  1., -0., -0.,  1.,  1.]])
    Coordinates:
      * word       (word) <U6 'walk' 'walked' 'walks'
      * cues       (cues) <U3 '#wa' 'alk' 'ed#' 'ked' 'ks#' 'lk#' 'lke' 'lks' 'wal'





Check the model's performance
=============================


Prediction accuracy
-------------------

pyldl.performance.accuracy returns how many words are correcly predicted.

.. code-block:: python

    >>> import pyldl.performance as lp
    >>> lp.accuracy(chat, cmat)
    1.0
    >>> lp.accuracy(shat, smat)
    1.0


Prediction dataframes
---------------------

You can see which word is predicted correctly in more details with pyldl.performance.predict_df. 

.. code-block:: python

    >>> lp.predict_df(chat, cmat)
      WordDISC    pred   acc
    0     walk    walk  True
    1   walked  walked  True
    2    walks   walks  True
    >>> lp.predict_df(shat, smat)
      WordDISC    pred   acc
    0     walk    walk  True
    1   walked  walked  True
    2    walks   walks  True


Obtain predictions for a particular word
----------------------------------------

.. code-block:: python

    >>> lp.predict('walked', chat, cmat)
    0    walked
    1      walk
    2     walks
    dtype: object
    >>> lp.predict('walked', shat, smat)
    0    walked
    1     walks
    2      walk
    dtype: object



Deriving semantic measures
==========================

Semantic support
----------------

Semantic support represents how much a particular form (e.g. triphone) is supported by semantics.

.. code-block:: python

    >>> import pyldl.measures as lmea
    >>> sem_ed = lmea.semantic_support('walked', 'ed#', chat)
    >>> round(sem_ed, 10)
    1.0
    >>> sem_ks = lmea.semantic_support('walked', 'ks#', chat)
    >>> round(sem_ks, 10)
    0.0


Production accuracy
-------------------

Production accuracy is similar to semantic support, but looks into how closely the model makes a prediction to the target form vector.

.. code-block:: python

    >>> p_acc = lmea.prod_acc('walked', cmat, chat)
    >>> p_acc
    1.0


Functional load
---------------

Functional load represents how much a certain form (e.g. triphone) helps to identify the target word's semantics. In the following example, "-ed" is unique to "walked" in this toy example. Therefore, "-ed" is very helpful to discriminate "walked" from the other two, hence a high functional load value. On the other hand, "wa-" is shared by all the three words. Therefore, "wa-" does not help so much to dintinguish the three words, hence a low functional load value.

.. code-block:: python

    >>> fl_ed = lmea.functional_load('ed#', fmat, 'walked', smat)
    >>> fl_wa = lmea.functional_load('wa#', fmat, 'walked', smat)
    >>> round(fl_ed, 10)
    1.0
    >>> round(fl_wa, 3)
    0.113


Uncertainty in production and comprehension
-------------------------------------------

pyldl.measures.uncertainty returns how much uncertainty is among the model's predictions.

.. code-block:: python

    >>> unc_prod = lmea.uncertainty('walked', chat, cmat)
    >>> unc_comp = lmea.uncertainty('walked', shat, smat)
    >>> round(unc_prod, 3)
    2.143
    >>> round(unc_comp, 3)
    2.259


Semantic vector length
----------------------

The length of a semantic vector can be obtained by pyldl.measures.vector_length.

.. code-block:: python

    >>> vlen = lmea.vector_length('walked', smat)
    >>> round(vlen, 3)
    8.062




----

.. [1] Baayen, R. H., Chuang, Y.-Y., & Blevins, J. P. (2018). Inflectional morphology with linear mappings. *The Mental Lexicon*, 13(2), 230-268.
