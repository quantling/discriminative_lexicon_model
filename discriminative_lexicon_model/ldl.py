import pandas as pd
import numpy as np
import re
from pathlib import Path

from . import mapping as lm
from . import performance as lp

class LDL:
    def __init__ (self, words=None, embed_or_df=None, cmat=False, smat=False,
            fmat=False, gmat=False, vmat=False, chat=False, shat=False,
            allmatrices=False, noise=False, events=None, freqs=None):
        if not (words is None):
            self.words = words
        if (any([cmat, smat, fmat, gmat, vmat, chat, shat])) or allmatrices:
            self.gen_matrices(words=words, embed_or_df=embed_or_df, cmat=cmat,
                    smat=smat, fmat=fmat, gmat=gmat, vmat=vmat, chat=chat,
                    shat=shat, allmatrices=allmatrices, noise=noise,
                    events=events, freqs=freqs)
        return None

    def gen_cmat (self, words=None, gram=3, count=True, noise=False, freqs=None):
        if not (words is None):
            self.words = words
        self.cmat = lm.gen_cmat(self.words, gram=gram, count=count, noise=noise, freqs=freqs)
        return None

    def gen_smat (self, embed_or_df, words=None, noise=False, freqs=None):
        if not (words is None):
            self.words = words
        if isinstance(embed_or_df, pd.core.frame.DataFrame):
            self.smat = lm.gen_smat_from_df(embed_or_df, noise=noise, freqs=freqs)
        else:
            self.smat = lm.gen_smat(self.words, embed_or_df, noise=noise, freqs=freqs)
        return None

    def gen_fmat (self, events=None):
        if events is None:
            self.fmat = lm.gen_fmat(cmat=self.cmat, smat=self.smat)
        else:
            isind = all([ isinstance(i, (int, np.integer)) for i in events ])
            if isind:
                self.fmat = lm.incremental_learning_byind(events, self.cmat, self.smat)
            else:
                self.fmat = lm.incremental_learning(events, self.cmat, self.smat)
        return None

    def gen_gmat (self, events=None):
        if events is None:
            self.gmat = lm.gen_gmat(cmat=self.cmat, smat=self.smat)
        else:
            isind = all([ isinstance(i, (int, np.integer)) for i in events ])
            if isind:
                self.gmat = lm.incremental_learning_byind(events, self.smat, self.cmat)
            else:
                self.gmat = lm.incremental_learning(events, self.smat, self.cmat)
        return None

    def gen_vmat (self, words=None, gram=3):
        if not (words is None):
            self.words = words
        self.vmat = lm.gen_vmat(self.words, gram=gram)
        return None

    def gen_shat (self):
        self.shat = lm.gen_shat(cmat=self.cmat, fmat=self.fmat)
        return None

    def gen_chat (self):
        self.chat = lm.gen_chat(smat=self.smat, gmat=self.gmat)
        return None

    def gen_matrices (self, words=None, embed_or_df=None, cmat=True, smat=True, fmat=True, gmat=True, vmat=True, chat=True, shat=True, allmatrices=True, noise=False, events=None, freqs=None):
        if cmat or allmatrices:
            self.gen_cmat(words=words, noise=noise, freqs=freqs)
        if smat or allmatrices:
            self.gen_smat(embed_or_df=embed_or_df, words=words, noise=noise, freqs=freqs)
        if fmat or allmatrices:
            self.gen_fmat(events=events)
        if gmat or allmatrices:
            self.gen_gmat(events=events)
        if vmat or allmatrices:
            self.gen_vmat(words=words)
        if chat or allmatrices:
            self.gen_chat()
        if shat or allmatrices:
            self.gen_shat()
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
            is_end = xs[-1][-1] == '#'
            is_max_iter = i==(max_attempt-1)
            if is_end or is_max_iter:
                if is_max_iter:
                    print('The maximum number of iterations ({:d}) reached.'.format(max_attempt))
                break
        df = pd.DataFrame(vecs).rename(columns={ i:j for i,j in enumerate(self.cmat.cues.values) })
        hdr = pd.Series(xs).to_frame(name='Selected')
        df = pd.concat([hdr, df], axis=1)
        if word:
            df = concat_cues(df.Selected)
        return df

    def save_matrices (self, directory, add='', mats=None, compress=True):
        ext = '.csv.gz' if compress else '.csv'
        if mats is None:
            mats = ['cmat','smat','fmat','gmat','vmat','shat','chat']
        for i in mats:
            if hasattr(self, i):
                path = directory + '/' + i + add + ext
                lm.save_mat(getattr(self, i), path)
            else:
                pass
        return None

    def load_matrices (self, directory, add=''):
        mats = ['cmat','smat','fmat','gmat','vmat','shat','chat']
        for i in mats:
            path_gz = Path(directory+'/'+i+add+'.csv.gz')
            path_csv = Path(directory+'/'+i+add+'.csv')
            if path_gz.exists:
                path = path_gz
            elif path_csv.exists:
                path = path_csv
            else:
                continue
            mat = lm.load_mat(str(path))
            path = re.sub(r'\..+$', '', path.name).replace(add, '')
            setattr(self, path, mat)
        return None

    def accuracy (self, method='correlation', print_output=True):
        acc_comp = acc_prod = None
        exist_chat = hasattr(self, 'chat')
        exist_shat = hasattr(self, 'shat')
        if exist_chat:
            acc_prod = lp.accuracy(pred=self.chat, gold=self.cmat, method=method)
        if exist_shat:
            acc_comp = lp.accuracy(pred=self.shat, gold=self.smat, method=method)
        if (acc_comp is None) and (acc_prod is None):
            raise ValueError('No C-hat or S-hat was found.')
        if print_output:
            if (acc_comp is None) and (not acc_prod is None):
                acc_prod = 'Production: {}'.format(acc_prod)
                acc = acc_prod
            elif (not acc_comp is None) and (acc_prod is None):
                acc_comp = 'Comprehension: {}'.format(acc_comp)
                acc = acc_comp
            else:
                acc_prod = 'Production: {}'.format(acc_prod)
                acc_comp = 'Comprehension: {}'.format(acc_comp)
                acc = acc_comp + '\n' + acc_prod
            print(acc)
            acc = None
        else:
            acc = {'Comprehension': acc_comp, 'Production': acc_prod}
            acc = { i:j for i,j in acc.items() if not j is None }
        return acc

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


