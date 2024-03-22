import pyldl.mapping as lm
import pandas as pd

class LDL:
    def __init__ (self, words=None, embed=None, noise=False, events=None):
        if (not words is None) and (not embed is None):
            self.gen_matrices(words, embed, noise=noise, events=events)
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

    def gen_all_matrices (self, words, embed, noise=False, events=None):
        self.cmat = self.gen_cmat(words=words, noise=noise)
        self.smat = self.gen_smat(embed=embed, words=None, noise=noise)
        self.fmat = self.gen_fmat(events)
        self.gmat = self.gen_gmat(events)
        self.vmat = self.gen_vmat(words=None)
        return None

