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

Short summary
-------------
DLM is a single model of language processing (comprehension and production both) consisting of 4 + 2 components (i.e., matrices). They are :math:`\mathbf{C}` (word-forms), :math:`\mathbf{S}` (word-meanings), :math:`\mathbf{F}` (form-meaning associations), :math:`\mathbf{G}` (meaning-form associations), :math:`\mathbf{\hat{C}}` (predicted word-forms), and :math:`\mathbf{\hat{S}}` (predicted word-meanings).

A little bit more detail
------------------------
DLM is a language processing model based on learning. DLM usually consists of four components (matrices): :math:`\mathbf{C}` (word-forms), :math:`\mathbf{S}` (word-meanings), :math:`\mathbf{F}` (form-meaning associations), and :math:`\mathbf{G}` (meaning-form associations). DLM models the comprehension as mapping from forms to meanings, namely DLM estimates :math:`\mathbf{F}` so that the product of :math:`\mathbf{C}` and :math:`\mathbf{F}`, namely :math:`\mathbf{CF}` (i.e., mapping of forms onto meanings), becomes as close as possible to :math:`\mathbf{S}`. :math:`\mathbf{CF}` is also called :math:`\mathbf{\hat{S}}`. :math:`\mathbf{\hat{S}}` is the model's predictions about word meanings, while :math:`\mathbf{S}` is the gold-standard "correct" meanings of these words. Similarly, DLM models the speech production as mapping from meanings to forms. DLM estimates :math:`\mathbf{G}` so that :math:`\mathbf{SG}` (which is also called :math:`\mathbf{\hat{C}}`) becomes as close as possible to :math:`\mathbf{C}` (i.e., the gold-standard correct form matrix). DLM is conceptually a single model containing these six components (i.e., :math:`\mathbf{C}`, :math:`\mathbf{S}`, :math:`\mathbf{F}`, :math:`\mathbf{G}`, :math:`\mathbf{\hat{C}}`, and :math:`\mathbf{\hat{S}}`). To reflect this conceptualization, *discriminative_lexicon_model* provides a class having these matrices as its attributes. The class is ``discriminative_lexicon_model.ldl.LDL``.




Create a model object
=====================

``discriminative_lexicon_model.ldl.LDL`` creates a model of DLM.

.. code-block:: python

   >>> import discriminative_lexicon_model as dlm
   >>> mdl = dlm.ldl.LDL()
   >>> print(type(mdl))
   <class 'discriminative_lexicon_model.ldl.LDL'>
   >>> mdl.__dict__
   {}

With no argument, ``discriminative_lexicon_model.ldl.LDL`` creates an empty model (of DLM), which is to be populated later with some class methods (see below).



Set up the basis matrices C and S
=================================

In order to estimate association matrices and create predictions based on them, :math:`\mathbf{C}` and :math:`\mathbf{S}` must be set up first.



C-matrix
--------

:math:`\mathbf{C}` is a collection of form-vectors of words. :math:`\mathbf{C}` can be created from a list of words by ``discriminative_lexicon_model.ldl.LDL.gen_cmat``.


.. code-block:: python

   >>> mdl.gen_cmat(['walk','walked','walks'])
   >>> print(mdl.cmat)
   <xarray.DataArray (word: 3, cues: 9)>
   array([[1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 1, 1, 1, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 1, 1]])
   Coordinates:
     * word     (word) <U6 'walk' 'walked' 'walks'
     * cues     (cues) <U3 '#wa' 'wal' 'alk' 'lk#' 'lke' 'ked' 'ed#' 'lks' 'ks#'




S-matrix
--------

:math:`\mathbf{S}` is a collection of semantic vectors of words. :math:`\mathbf{S}` can be set up by means of ``discriminative_lexicon_model.ldl.LDL.gen_smat``. For its argument, semantic vectors need to be set up with ``pandas.core.frame.DataFrame`` with words as its indices and semantic dimensions as its columns. Semantic dimensions can be defined either by hand or by an embeddings algorithm such as word2vec and fastText. Regardless of the method of constructing semantics, ``discriminative_lexicon_model.ldl.LDL.gen_smat`` sets up :math:`\mathbf{S}`, as long as the dataframe given to its (first) argument follows the right format (i.e., rows = words, columns = semantic dimensions). In the example below, semantic dimensions are set up by hand.


.. code-block:: python

   >>> import pandas as pd
   >>> semdf = pd.DataFrame({'WALK':[1,1,1], 'Present':[1,0,1], 'Past':[0,1,0], 'ThirdPerson':[0,0,1]}, index=['walk','walked','walks'])
   >>> print(semdf)
           WALK  Present  Past  ThirdPerson
   walk       1        1     0            0
   walked     1        0     1            0
   walks      1        1     0            1
   >>> mdl.gen_smat(semdf)
   >>> print(mdl.smat)
   <xarray.DataArray (word: 3, semantics: 4)>
   array([[1, 1, 0, 0],
          [1, 0, 1, 0],
          [1, 1, 0, 1]])
   Coordinates:
     * word       (word) <U6 'walk' 'walked' 'walks'
     * semantics  (semantics) object 'WALK' 'Present' 'Past' 'ThirdPerson'




Estimation of the association matrices F and G
==============================================

F-matrix
--------

With :math:`\mathbf{C}` and :math:`\mathbf{S}` established, the comprehension association matrix :math:`\mathbf{F}` can be estimated by ``discriminative_lexicon_model.ldl.LDL.gen_fmat``. It does not require any argument, because :math:`\mathbf{C}` and :math:`\mathbf{S}` are stored already as attributes of the class and therefore accessible by the model.

.. code-block:: python

   >>> mdl.gen_fmat()
   >>> print(mdl.fmat.round(2))
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

Similarly, with :math:`\mathbf{C}` and :math:`\mathbf{S}` established, the production association matrix :math:`\mathbf{G}` can also be estimated by ``discriminative_lexicon_model.ldl.LDL.gen_gmat``. It does not require any argument, either, because :math:`\mathbf{C}` and :math:`\mathbf{S}` are stored already as attributes of the class and therefore accessible by the model.

.. code-block:: python

   >>> mdl.gen_gmat()
   >>> print(mdl.gmat.round(2))
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

The model's predictions about word-meanings based on word-forms (i.e., :math:`\mathbf{\hat{S}}`) can be obtained by discriminative_lexicon_model.ldl.LDL.gen_shat, given that :math:`\mathbf{C}` and :math:`\mathbf{F}` are already set up and stored as attributes of the class instance.

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

Similarly, the model's predictions about word-forms based on word-meanings (i.e., :math:`\mathbf{\hat{C}}`) can be obtained with discriminative_lexicon_model.ldl.LDL.gen_chat, given that :math:`\mathbf{S}` and :math:`\mathbf{G}` are already set up and stored as attributes of the class instance.

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

