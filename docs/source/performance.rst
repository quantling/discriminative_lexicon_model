====================
Performance measures
====================

To illustrate the performance measures, this section will use the same matrices as used in the examples in the section :ref:`Linear Discriminative Learning`. See the section for more detail about the matrices and how to construct them.


Prediction accuracy
===================

pyldl.performance.accuracy returns how many words are correcly predicted.

.. code-block:: python

    >>> import pandas as pd
    >>> import pyldl.mapping as pmap
    >>> import pyldl.performance as lp
    >>> words = ['walk','walked','walks']
    >>> cmat  = pmap.gen_cmat(words)
    >>> infl = pd.DataFrame({'Word':['walk','walked','walks'], 'Lemma':['walk','walk','walk'], 'Tense':['PRES','PAST','PRES']})
    >>> smat = pmap.gen_smat_sim(infl, dim_size=5)
    >>> shat = pmap.gen_shat(cmat=cmat, smat=smat)
    >>> chat = pmap.gen_chat(smat=smat, cmat=cmat)
    >>> lp.accuracy(chat, cmat)
    1.0
    >>> lp.accuracy(shat, smat)
    1.0

Since this toy example is so tiny, all the words are perfectlly correctly predicted.


Prediction dataframes
=====================

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

The leftmost column "WordDISC" is the target word, or the correct answers. The middle column "pred" is the predictions by the model. The right column "acc" shows is the predictions are correct or not.

You can blend in the model's second guesses and more by setting the argument "max_guess".

.. code-block:: python

    >>> lp.predict_df(chat, cmat, max_guess=2)
      WordDISC   pred1  pred2  acc1   acc2
    0     walk    walk  walks  True  False
    1   walked  walked   walk  True  False
    2    walks   walks   walk  True  False
    >>> lp.predict_df(shat, smat, max_guess=2)
      WordDISC   pred1   pred2  acc1   acc2
    0     walk    walk  walked  True  False
    1   walked  walked   walks  True  False
    2    walks   walks  walked  True  False


Obtain predictions for a particular word
============================================

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
