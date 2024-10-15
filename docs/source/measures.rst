==========================
Semantic measures from LDL
==========================

Several measures can be derived from the LDL matrices. For details about the matrices, see the section :ref:`Linear Discriminative Learning`.

To illustrate the semantic measures, this section will use the same matrices as used in the examples in the section :ref:`Linear Discriminative Learning`. See the section for more detail about the matrices and how to construct them.


Semantic support
================

Semantic support from a word's meaning to a triphone is defined as the value of the corresponding cell in :math:`\mathbf{\hat{C}}`.

.. math::

    \text{SemSup}_{i,j} = \mathbf{\hat{C}}_{i,j}

In discriminative_lexicon_model, semantic support can be obtained by discriminative_lexicon_model.measures.semantic_support.

.. code-block:: python

    >>> import pandas as pd
    >>> import discriminative_lexicon_model.mapping as pmap
    >>> import discriminative_lexicon_model.measures as lmea
    >>> words = ['walk','walked','walks']
    >>> cmat  = pmap.gen_cmat(words)
    >>> infl = pd.DataFrame({'Word':['walk','walked','walks'], 'Lemma':['walk','walk','walk'], 'Tense':['PRES','PAST','PRES']})
    >>> smat = pmap.gen_smat_sim(infl, dim_size=5)
    >>> shat = pmap.gen_shat(cmat=cmat, smat=smat)
    >>> chat = pmap.gen_chat(smat=smat, cmat=cmat)
    >>> sem_ed = lmea.semantic_support('walked', 'ed#', chat)
    >>> round(sem_ed, 10)
    1.0
    >>> sem_ks = lmea.semantic_support('walked', 'ks#', chat)
    >>> round(sem_ks, 10)
    0.0

The meaning of *walked* supports the triphone *-ks#* very strongly, because *-ks#* is a part of *walked*, while the meaning of *walked* does not support *-ks#* at all, because *-ks#* is not contained by *walked*. The system successfully discriminated forms based on semantics, which is unsurprising, considering how tiny this toy example is!

discriminative_lexicon_model.measures.semantic_support only calculates a semantic support value from a word to a triphone. If you would like to know how much support a word's semantics give to its component triphones in total, you need to add up all the semantic values from the word's semantics to all the component triphones. For this purpose, discriminative_lexicon_model.measures.semantic_support_word can be used:

.. code-block:: python

    >>> sem_wd = lmea.semantic_support_word('walked', chat)
    >>> round(sem_wd, 10)
    6.0


Production accuracy
===================

A different way of understanding semantic support is to focus on how well word-forms are predicted. If a word's component triphones are all well-supported, the word's form is predicted well. Production accuracy of a word :math:`i` is the correlation between predicted and gold-standard (correct) form vectors.

.. math::

    \text{ProdAcc}_{i} = \text{cor}(\mathbf{\hat{C}}_{i,*}, \mathbf{C}_{i,*})

:math:`\text{ProdAcc}` can be obtained by discriminative_lexicon_model.measures.prod_acc.

.. code-block:: python

    >>> p_acc = lmea.prod_acc('walked', cmat, chat)
    >>> p_acc
    1.0


Functional load
===============

Semantic support is a measure from the perspective of speech production. Its counterpart in comprehension process is functional load. Functional load can conceptually be understood as how much a particular triphone helps to identify the target word's semantics. Functional load of a triphone :math: `j` to a word :math:`i` is defined as below:

.. math::

    \text{FuncLoad}_{j,i} = \text{cor}(\mathbf{F}_{j,*}, \mathbf{\hat{S}}_{i,*})

where :math:`\mathbf{F}_{j,*}` and :math:`\mathbf{\hat{S}}_{i,*}` represent the :math:`j`-th and :math:`i`-th row vectors of :math:`\mathbf{F}` and :math:`\mathbf{\hat{S}}` respectively.

In discriminative_lexicon_model, functional load can be obtained with discriminative_lexicon_model.measures.functional_load.

.. code-block:: python

    >>> fl_ed = lmea.functional_load('ed#', fmat, 'walked', smat)
    >>> fl_wa = lmea.functional_load('wa#', fmat, 'walked', smat)
    >>> round(fl_ed, 10)
    1.0
    >>> round(fl_wa, 3)
    0.113

*-ed#* is unique to *walked* in this tiny toy example. Therefore, *-ed#* helps to dintinguish *walked* from the others a lot, hence a high functional load value. On the other hand, *#wa* is shared by all the three words in this example. Because of that, *#wa* has a very weak discriminative power and does not help so much to distinguish target words, hence a low functional load value.


Uncertainty in production and comprehension
===========================================

Semantic support and functional load are the measures that care how much the target triphone/word is supported. Semantic measures can be set up from another perspective, namely from the perspective of the target word/triphone's neighborhood. If the target word is supported (or activated) strongly alone with the others being not activated so much, then the target word has less chance to be confused with other similar words. On the other hand, if the target word has many neighbors activated at the same time with very close competition, then the target word may be difficult to process, even if it receives the strongest activation/support.

This concept of "uncertainty" is defined in discriminative_lexicon_model as the sum of the products of the correlation coefficients between the predicted vector of the target word and all the other words' vectors and the correlation's ranks:

.. math::

    \text{UncertProd}_{i} = \sum_{k} \big( \text{cor}(\mathbf{\hat{C}}_{i,*}, \mathbf{C}_{k,*}) \times \text{rank}(\text{cor}(\mathbf{\hat{C}}_{i,*}, \mathbf{C}_{k,*})) \big)

This measure represents how much uncertainty there is in the production process. Uncertainty can also be defined for the comprehension process:

.. math::

   \text{UncertComp}_{i} = \sum_{k} \big( \text{cor}(\mathbf{\hat{S}}_{i,*}, \mathbf{S}_{k,*}) \times \text{rank}(\text{cor}(\mathbf{\hat{S}}_{i,*}, \mathbf{S}_{k,*})) \big)

:math:`\text{UncertProd}` and :math:`\text{UncertComp}` only differ in which group of matrices to use, namely :math:`\mathbf{C}` and :math:`\mathbf{\hat{C}}` vs. :math:`\mathbf{S}` and :math:`\mathbf{\hat{S}}`. Therefore, in discriminative_lexicon_model, there is only one method, which can be used for :math:`\text{UncertProd}` and :math:`\text{UncertComp}`.

.. code-block:: python

    >>> unc_prod = lmea.uncertainty('walked', chat, cmat)
    >>> unc_comp = lmea.uncertainty('walked', shat, smat)
    >>> round(unc_prod, 3)
    2.143
    >>> round(unc_comp, 3)
    2.259


Semantic vector length
======================

Another aspect of semantic vectors is their lengths. It can be obtained by discriminative_lexicon_model.measures.vector_length.

.. math::

    \text{SemLen}_{i} = \sum_{j}|S_{ij}|

.. code-block:: python

    >>> vlen = lmea.vector_length('walked', smat)
    >>> round(vlen, 3)
    8.062



