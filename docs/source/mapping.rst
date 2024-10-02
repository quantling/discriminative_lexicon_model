==============================
Linear Discriminative Learning
==============================

Linear Discriminative Learning (LDL) [1]_ was developed against the backgroud of Naive Discriminative Learning (NDL) [2]_. NDL is based on the Rescorla-Wagner learning rule [3]_. The learning rule updates associations between cues (e.g., word forms) and outcomes (e.g., meanings) incrementally, based on cooccurrences of cues and outcomes. Incrementally learned associations can asymptote an equilibrium, where association strengths stay almost constant with (almost) no more updates. Such an equilibrium state can theoretically be estimated without incrementally learning associations. The "endstate-of-learning" of the Rescorla-Wagner learning rule is the Danks equation [4]_.

NDL only accepts binary inputs and outputs. Cues or outcomes are present (1) or absent (0). LDL loosenes this constrainty and generalizes NDL so that cues and outcomes can also take real values. For the current implementation, LDL adopts the real-value counterpart of the Danks equation. In other words, LDL estimates the equilibrium state of associations between cues and outcomes at once, without incrementally learning the associations. The method of estimating the equilibrium associations is mathematically equivalent to multivariate regression, where multiple continuous predictors and response variables are accepted. For more detail, see [1]_ and [5]_.

To estimate associations (or weight matrices) between cues and outcomes, LDL requires two matrices. One is a C-matrix (i.e., :math:`\mathbf{C}`), which can also be called a form matrix or a cue matrix. :math:`\mathbf{C}` has words as rows and sublexical units (e.g., triphones) as columns. Each row represents a form vector of a word. In the current implementation, each form vector is coded 1 where the triphone is contained in the word and 0 otherwise.

With discriminative_lexicon_model, you can create a :math:`\mathbf{C}` from a list of words by using discriminative_lexicon_model.mapping.gen_cmat.

.. code-block:: python
    
    >>> import discriminative_lexicon_model.mapping as pmap
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

The other matrix LDL requires is a S-matrix (i.e., :math:`\mathbf{S}`). :math:`\mathbf{S}` can also be called a meaning matrix or an outcome matrix. :math:`\mathbf{S}` also has words as rows as :math:`\mathbf{C}`, but :math:`\mathbf{S}`'s columns are semantic dimensions. Therefore, rows of :math:`\mathbf{S}` can be understood as semantic vectors of words.

While :math:`\mathbf{S}` can be obtained by embedding techniques such as word2vec, discriminative_lexicon_model offers a way of approximating words' semantic vectors by those words' inflectional information. The semantic vectors created in this method are called "simulated semantic vectors" [6]_.

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

Simulated semantic vectors were explained in [6]_ as the sum of the pertinent random normal vectors corresponding to the lemma and morphological features. For example, a semantic vector for "walks" is created by taking the sum of random normal vectors for "WALK" (lemma), third-person, and singular.

This method is implemented with two matrices: :math:`\mathbf{M}` and :math:`\mathbf{J}`. The rows of :math:`\mathbf{M}` are words and its columns are morphological features. Therefore, :math:`\mathbf{M}` encodes which morphological features each word has. :math:`\mathbf{J}` has morphological features as rows and semantic dimensions as columns. Therefore, rows of :math:`\mathbf{J}` are randomly-generated semantic vectors for each morphological feature. Simulated semantic vectors are obtained for words by multiplying them:

.. math::

    \mathbf{S}_{\text{sim}} = \mathbf{MJ}


:math:`\mathbf{M}` and :math:`\mathbf{J}` can be obtained in discriminative_lexicon_model with discriminative_lexicon_model.mapping.gen_mmat and discriminative_lexicon_model.mapping.gen_jmat. They are used internally in discriminative_lexicon_model.mapping.gen_smat_sim.

Now that we have :math:`\mathbf{C}` and :math:`\mathbf{S}`, we can "learn" the associations between them. The associations, or weight matrices, between them are called :math:`\mathbf{F}` and :math:`\mathbf{G}`. These two weight matrices are mathematically obtained as below [1]_:

.. math::

    \mathbf{CF} = \mathbf{S}

    \mathbf{C^{T}CF} = \mathbf{C^{T}S}

    \mathbf{(C^{T}C)^{-1}C^{T}CF} = \mathbf{(C^{T}C)^{-1}C^{T}S}

    \mathbf{IF} = \mathbf{(C^{T}C)^{-1}C^{T}S}

    \mathbf{F} = \mathbf{(C^{T}C)^{-1}C^{T}S}


.. math::

    \mathbf{SG} = \mathbf{C}

    \mathbf{S^{T}SG} = \mathbf{S^{T}C}

    \mathbf{(S^{T}S)^{-1}S^{T}SG} = \mathbf{(S^{T}S)^{-1}S^{T}C}

    \mathbf{IG} = \mathbf{(S^{T}S)^{-1}S^{T}C}

    \mathbf{G} = \mathbf{(S^{T}S)^{-1}S^{T}C}

In discriminative_lexicon_model, :math:`\mathbf{F}` and :math:`\mathbf{G}` can be obtained with discriminative_lexicon_model.mapping.gen_fmat and discriminative_lexicon_model.mapping.gen_gmat:

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


:math:`\mathbf{F}` has cues as its rows and semantics as its columns. It can be used to predict words' meanings based on the words' forms. Namely:

.. math::

    \mathbf{CF} = \mathbf{\hat{S}}

:math:`\mathbf{\hat{S}}` is a predicted semantic matrix (or semantic vectors). Since this equation represents the process to infer meanings based on forms, it can be understood conceptually as the comprehension process of language.

In discriminative_lexicon_model, you can use discriminative_lexicon_model.mapping.gen_shat for this purpose:

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


In fact, you do not have to produce :math:`\mathbf{F}`, if you are only interested in producing :math:`\mathbf{\hat{S}}`. You can directly estimate :math:`\mathbf{\hat{S}}` from :math:`\mathbf{C}` and :math:`\mathbf{S}` with discriminative_lexicon_model.mapping.gen_shat:

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


Similarly to :math:`\mathbf{F}`, :math:`\mathbf{G}` is also used to produce predicted form matrix/vectors (:math:`\mathbf{\hat{C}}`) as below. The equation can be understood conceptually as the production process of language.

.. math::

    \mathbf{SG} = \mathbf{\hat{C}}

In discriminative_lexicon_model, :math:`\mathbf{\hat{C}}` is obtained by discriminative_lexicon_model.mapping.gen_chat.

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

----

.. [1] Baayen, R. H., Chuang, Y.-Y., Shafaei-Bajestan, E., & Blevins, J. P. (2019). The Discriminative Lexicon: A Unified Computational Model for the Lexicon and Lexical Processing in Comprehension and Production Grounded Not in (De)Composition but in Linear Discriminative Learning. *Complexity*, 1-39.
.. [2] Baayen, R. H., Milin, P., Durdevic, D. F., Hendrix, P., & Marelli, M. (2011). An Amorphous Model for Morphological Processing in Visual Comprehension Based on Naive Discriminative Learning. *Psychological Review*, 118(3), 438-481.
.. [3] Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. In A. H. Black & W. F. Prokasy (Eds.), *Classical conditioning II: Curent research and theory* (pp. 64-99). New York: Appleton-Century-Crofts.
.. [4] Danks, D. (2003). Equilibria of the Rescorla-Wagner model. *Journal of Mathematical Psychology*, 47(2), 109-121.
.. [5] Shafaei-Bajestan, E., Moradipour-Tari, M., Uhrig, P., & Baayen, R. H. (2021). LDL-AURIS: a computational model, grounded in error-driven learning, for the comprehension of single spoken words. *Language, Cognition and Neuroscience*, 1-28.
.. [6] Baayen, R. H., Chuang, Y.-Y., & Blevins, J. P. (2018). Inflectional morphology with linear mappings. *The Mental Lexicon*, 13(2), 230-268.

