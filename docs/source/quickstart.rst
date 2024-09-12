==========
Quickstart
==========

*discriminative_lexicon_model* is a python-implementation of Discriminative Lexicon Model [1]_.

Installation
============

*discriminative_lexicon_model* is available on PyPI.

.. code:: bash

    pip install --user discriminative_lexicon_model


Quick overview of the theory "Discriminative Lexicon Model (DLM)"
=================================================================

In DLM, language processing is modelled as linear mappings between word-forms and word-meanings. Word-forms and word-meanings can be defined in any way, as long as each word form/meaning is expressed in the form of a vector (i.e., an array of numbers). Word-forms are stacked up to be a matrix called the *C* matrix. Word-meanings are stacked up to be another matrix called the *S* matrix. The comprehension process can be modelled as receiving word-forms (i.e., C) and predicting word-meanings (i.e., S). Such a matrix that approximates S as closely as possible based on C can be estimated either analytically or computationally (see [1]_ for more detail), and it is called the *F* matrix. With C and F, the approximation (prediction) of S can be derived, and it is called the :math:`\hat{S}` matrix. Similarly, the production process can be modelled as receiving word-meanings (i.e., S) and predicting word-forms (i.e., C). Such a matrix that approximates C based on S is called the *G* matrix. With S and G, the model's predictions about word-forms are obtained as yet another matrix. The matrix is called the :math:`\hat{C}` matrix. It is shown below how to set up and estimate these matrices.


Set up the basis matrices C and S
=================================

C-matrix
--------

The C matrix is a collection of form-vectors of words. You can create a C-matrix from a list of words by using discriminative_lexicon_model.mapping.gen_cmat.

.. code-block:: python

    >>> import discriminative_lexicon_model as dlm
    >>> words = ['walk','walked','walks']
    >>> cmat  = dlm.mapping.gen_cmat(words)
    >>> cmat
    <xarray.DataArray (word: 3, cues: 9)>
    array([[1, 1, 1, 1, 0, 0, 0, 0, 0],
           [1, 1, 1, 0, 1, 1, 1, 0, 0],
           [1, 1, 1, 0, 0, 0, 0, 1, 1]])
    Coordinates:
      * word     (word) <U6 'walk' 'walked' 'walks'
      * cues     (cues) <U3 '#wa' 'wal' 'alk' 'lk#' 'lke' 'ked' 'ed#' 'lks' 'ks#'


S-matrix
--------

The S matrix is a collection of semantic vectors of words. For one method, an S-matrix can be set up by defining semantic dimensions by hand. This can be achieved by discriminative_lexicon_model.mapping.gen_smat_from_df.


.. code-block:: python

    >>> import pandas as pd
    >>> smat = pd.DataFrame({'WALK':[1,1,1], 'Present':[1,0,1], 'Past':[0,1,0], 'ThirdPerson':[0,0,1]}, index=['walk','walked','walks'])
    >>> smat = dlm.mapping.gen_smat_from_df(smat)
    <xarray.DataArray (word: 3, semantics: 4)>
    array([[1, 1, 0, 0],
           [1, 0, 1, 0],
           [1, 1, 0, 1]])
    Coordinates:
      * word       (word) <U6 'walk' 'walked' 'walks'
      * semantics  (semantics) object 'WALK' 'Present' 'Past' 'ThirdPerson'



Estimation of the association matrices
======================================

F-matrix
--------

With C and S established, the comprehension association matrix F can be estimated by discriminative_lexicon_model.mapping.gen_fmat.

.. code-block:: python

    >>> fmat = dlm.mapping.gen_fmat(cmat, smat)
    >>> fmat.round(2)
    <xarray.DataArray (cues: 9, semantics: 4)>
    array([[ 0.28,  0.23,  0.05,  0.08],
           [ 0.28,  0.23,  0.05,  0.08],
           [ 0.28,  0.23,  0.05,  0.08],
           [ 0.15,  0.31, -0.15, -0.23],
           [ 0.05, -0.23,  0.28, -0.08],
           [ 0.05, -0.23,  0.28, -0.08],
           [ 0.05, -0.23,  0.28, -0.08],
           [ 0.08,  0.15, -0.08,  0.38],
           [ 0.08,  0.15, -0.08,  0.38]])
    Coordinates:
      * cues       (cues) <U3 '#wa' 'wal' 'alk' 'lk#' 'lke' 'ked' 'ed#' 'lks' 'ks#'
      * semantics  (semantics) object 'WALK' 'Present' 'Past' 'ThirdPerson'


G-matrix
--------

The production association matrix G can be obtained by discriminative_lexicon_model.mapping.gen_gmat.

.. code-block:: python

    >>> gmat = dlm.mapping.gen_gmat(cmat, smat)
    >>> gmat.round(2)
    <xarray.DataArray (semantics: 4, cues: 9)>
    array([[ 0.67,  0.67,  0.67,  0.33,  0.33,  0.33,  0.33, -0.  , -0.  ],
           [ 0.33,  0.33,  0.33,  0.67, -0.33, -0.33, -0.33, -0.  , -0.  ],
           [ 0.33,  0.33,  0.33, -0.33,  0.67,  0.67,  0.67, -0.  , -0.  ],
           [ 0.  ,  0.  ,  0.  , -1.  ,  0.  ,  0.  ,  0.  ,  1.  ,  1.  ]])
    Coordinates:
      * semantics  (semantics) object 'WALK' 'Present' 'Past' 'ThirdPerson'
      * cues       (cues) <U3 '#wa' 'wal' 'alk' 'lk#' 'lke' 'ked' 'ed#' 'lks' 'ks#'



Prediction of the form and semantic matrices
============================================

S-hat matrix
------------

The S-hat matrix (:math:`\mathbf{\hat{S}}`) can be obtained by discriminative_lexicon_model.mapping.gen_shat.

.. code-block:: python

    >>> shat = dlm.mapping.gen_shat(cmat, fmat)
    >>> shat.round(2)
    <xarray.DataArray (word: 3, semantics: 4)>
    array([[ 1.,  1., -0., -0.],
           [ 1., -0.,  1., -0.],
           [ 1.,  1., -0.,  1.]])
    Coordinates:
      * word       (word) <U6 'walk' 'walked' 'walks'
      * semantics  (semantics) object 'WALK' 'Present' 'Past' 'ThirdPerson'


C-hat matrix
------------

The C-hat matrix (:math:`\mathbf{\hat{C}}`) can be obtained with discriminative_lexicon_model.mapping.gen_chat.

.. code-block:: python

    >>> chat = dlm.mapping.gen_chat(smat, gmat)
    >>> chat.round(2)
    <xarray.DataArray (word: 3, cues: 9)>
    array([[ 1.,  1.,  1.,  1., -0., -0., -0., -0., -0.],
           [ 1.,  1.,  1., -0.,  1.,  1.,  1., -0., -0.],
           [ 1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  1.]])
    Coordinates:
      * word     (word) <U6 'walk' 'walked' 'walks'
      * cues     (cues) <U3 '#wa' 'wal' 'alk' 'lk#' 'lke' 'ked' 'ed#' 'lks' 'ks#'






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

.. [1] Baayen, R. H., Chuang, Y.-Y., Shafaei-Bajestan, & Blevins, J. P. (2019). The discriminative lexicon: A unified computational model for the lexicon and lexical processing in comprehension and production grounded not in (de)composition but in linear discriminative learning. *Complexity* 2019, 1-39.

