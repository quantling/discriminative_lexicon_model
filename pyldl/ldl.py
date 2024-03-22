import pyldl.mapping as lm
import pandas as pd
import numpy as np

class LDL:
    def __init__ (self, words=None, embed_or_df=None, noise=False, events=None):
        if (not words is None) and (not embed_or_df is None):
            self.gen_all_matrices(words, embed_or_df, noise=noise, events=events)
        else:
            pass
        return None

    def gen_cmat (self, words=None, noise=False):
        if not (words is None):
            self.words = words
        self.cmat = lm.gen_cmat(self.words, noise=noise)
        return None

    def gen_smat (self, embed_or_df, words=None, noise=False):
        if not (words is None):
            self.words = words
        if isinstance(embed_or_df, pd.core.frame.DataFrame):
            self.smat = lm.gen_smat_from_df(embed_or_df, noise=noise)
        else:
            self.smat = lm.gen_smat(self.words, embed_or_df, noise=noise)
        return None

    def gen_fmat (self, events=None):
        if events is None:
            self.fmat = lm.gen_fmat(cmat=self.cmat, smat=self.smat)
        else:
            isind = all([ isinstance(i, int) for i in events ])
            if isind:
                self.fmat = lm.incremental_learning_byind(events, self.cmat, self.smat)
            else:
                self.fmat = lm.incremental_learning(events, self.cmat, self.smat)
        return None

    def gen_gmat (self, events=None):
        if events is None:
            self.gmat = lm.gen_gmat(cmat=self.cmat, smat=self.smat)
        else:
            isind = all([ isinstance(i, int) for i in events ])
            if isind:
                self.gmat = lm.incremental_learning_byind(events, self.cmat, self.smat)
            else:
                self.gmat = lm.incremental_learning(events, self.cmat, self.smat)
        return None

    def gen_vmat (self, words=None):
        if not (words is None):
            self.words = words
        self.vmat = lm.gen_vmat(self.words)
        return None

    def gen_all_matrices (self, words, embed_or_df, noise=False, events=None):
        self.gen_cmat(words=words, noise=noise)
        self.gen_smat(embed_or_df=embed_or_df, words=None, noise=noise)
        self.gen_fmat(events)
        self.gen_gmat(events)
        self.gen_vmat(words=None)
        return None

    def produce (self, gold, word=False, roundby=10, max_attempt=50, positive=False):
        if not isinstance(gold, np.ndarray):
            gold = np.array(gold)
        p = -1
        xs = []
        vecs = []
        c_comp = np.zeros(self.cmat.cues.values.size)
        for i in range(max_attempt):
            s0 = np.matmul(c_comp, self.fmat.values)
            if positive:
                s0[s0<0] = 0 # It may be necessary for a small lexicon with multi-hot matrices.
            s = gold - s0
            g = np.matmul(self.gmat.values, np.diag(self.vmat.loc[self.vmat.current.values[p],:].values))
            c_prod = np.matmul(s, g)
            # c_prod[c_prod<0] = 0  # It may be necessary for a small lexicon with multi-hot matrices.
            if (c_prod<=0).all():
                break
            else:
                p = np.argmax(c_prod)
                c_comp[p] = c_comp[p] + 1
                xs = xs + [self.cmat.cues.values[p]]
                vecs = vecs + [c_prod.round(roundby)]
            if i==(max_attempt-1):
                print('The maximum number of iterations ({:d}) reached.'.format(max_attempt))
                break
        df = pd.DataFrame(vecs).rename(columns={ i:j for i,j in enumerate(self.cmat.cues.values) })
        hdr = pd.Series(xs).to_frame(name='Selected')
        df = pd.concat([hdr, df], axis=1)
        if word:
            df = concat_cues(df.Selected)
        return df

def concat_cues (a):
    assert is_consecutive(a)
    a = pd.Series(a).str.slice(start=0, stop=1).iloc[:-1].str.cat(sep='') + pd.Series(a).iloc[-1]
    return a

def is_consecutive (a):
    a = pd.Series(a)
    b = a.shift(-1)
    a = a.iloc[:-1]
    b = b.iloc[:-1]
    a = a.str.slice(start=1)
    b = b.str.slice(stop=-1)
    return (a==b).all()


