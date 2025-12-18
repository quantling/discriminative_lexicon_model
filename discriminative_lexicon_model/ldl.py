import pandas as pd
import numpy as np
import re
from pathlib import Path

from . import mapping as lm
from . import performance as lp

try:
    import torch
except ImportError:
    torch = None

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

    def produce (self, gold, word=False, roundby=10, max_attempt=50, positive=False, apply_vmat=True, backend='auto', device=None):
        """
        Produce output using discriminative learning.

        Parameters
        ----------
        gold : array-like
            The target semantic vector.
        word : bool
            If True, concatenate cues to form words.
        roundby : int
            Number of decimal places to round vectors.
        max_attempt : int
            Maximum number of iterations.
        positive : bool
            If True, set negative values to zero.
        apply_vmat : bool
            If True, apply validity matrix.
        backend : {'numpy', 'torch', 'auto'}
            'numpy' -> NumPy CPU implementation.
            'torch' -> PyTorch implementation (CPU/GPU depending on 'device').
            'auto'  -> Try torch+CUDA if available, else fall back to NumPy.
        device : str or None
            For torch backend: 'cuda', 'cpu', etc. If None and backend is
            'torch' or 'auto', chooses 'cuda' if available, else 'cpu'.
        """
        # Determine which backend to use
        use_torch = False
        if backend == 'torch':
            if torch is None:
                raise ImportError('PyTorch is not installed. Install it to use the "torch" backend.')
            use_torch = True
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif backend == 'auto':
            if torch is not None and (device == 'cuda' or (device is None and torch.cuda.is_available())):
                use_torch = True
                device = device or 'cuda'
        elif backend != 'numpy':
            raise ValueError(f'Unknown backend "{backend}". Use "numpy", "torch", or "auto".')

        if use_torch:
            return self._produce_torch(gold, word, roundby, max_attempt, positive, apply_vmat, device)
        else:
            return self._produce_numpy(gold, word, roundby, max_attempt, positive, apply_vmat)

    def _produce_numpy (self, gold, word, roundby, max_attempt, positive, apply_vmat):
        """NumPy implementation of produce (original CPU version)."""
        if not isinstance(gold, np.ndarray):
            gold = np.array(gold, dtype=np.float32)
        else:
            gold = gold.astype(np.float32)
        p = -1
        xs = []
        vecs = []
        cues_values = self.cmat.cues.values
        fmat_values = self.fmat.values.astype(np.float32)
        gmat_values = self.gmat.values.astype(np.float32)
        if apply_vmat:
            vmat_values = self.vmat.values.astype(np.float32)
            vmat_current_values = self.vmat.current.values
        c_comp = np.zeros(cues_values.size, dtype=np.float32)
        for i in range(max_attempt):
            s0 = np.matmul(c_comp, fmat_values)
            if positive:
                s0[s0<0] = 0 # It may be necessary for a small lexicon with multi-hot matrices.
            s = gold - s0
            if apply_vmat:
                # Element-wise multiplication instead of np.diag() matrix multiplication
                vmat_row = vmat_values[p]
                g = gmat_values * vmat_row
            else:
                g = gmat_values
            c_prod = np.matmul(s, g)
            # c_prod[c_prod<0] = 0  # It may be necessary for a small lexicon with multi-hot matrices.
            if (c_prod<=0).all():
                break
            else:
                p = np.argmax(c_prod)
                c_comp[p] = c_comp[p] + 1
                xs.append(cues_values[p])
                vecs.append(c_prod)
            is_unigram_onset = len(xs)==1 and len(xs[0])==1 and xs[0]=='#'
            is_end = (xs[-1][-1]=='#') and (not is_unigram_onset)
            is_max_iter = i==(max_attempt-1)
            if is_end or is_max_iter:
                if is_max_iter:
                    print('The maximum number of iterations ({:d}) reached.'.format(max_attempt))
                break
        # Round vectors only at the end, not in every iteration
        vecs = [v.round(roundby) for v in vecs]
        df = pd.DataFrame(vecs).rename(columns={ i:j for i,j in enumerate(cues_values) })
        hdr = pd.Series(xs).to_frame(name='Selected')
        df = pd.concat([hdr, df], axis=1)
        if word:
            df = concat_cues(df.Selected)
        return df

    def _produce_torch (self, gold, word, roundby, max_attempt, positive, apply_vmat, device):
        """PyTorch implementation of produce with GPU support."""
        if not isinstance(gold, np.ndarray):
            gold = np.array(gold)

        # Move data to torch tensors on specified device
        gold_tensor = torch.as_tensor(gold, dtype=torch.float32, device=device)
        fmat_tensor = torch.as_tensor(self.fmat.values, dtype=torch.float32, device=device)
        gmat_tensor = torch.as_tensor(self.gmat.values, dtype=torch.float32, device=device)

        if apply_vmat:
            vmat_tensor = torch.as_tensor(self.vmat.values, dtype=torch.float32, device=device)

        cues_values = self.cmat.cues.values
        c_comp = torch.zeros(cues_values.size, dtype=torch.float32, device=device)

        p = -1
        xs = []
        vecs = []

        for i in range(max_attempt):
            s0 = torch.matmul(c_comp, fmat_tensor)
            if positive:
                s0 = torch.clamp(s0, min=0)
            s = gold_tensor - s0

            if apply_vmat:
                # Element-wise multiplication with vmat row
                vmat_row = vmat_tensor[p]
                g = gmat_tensor * vmat_row
            else:
                g = gmat_tensor

            c_prod = torch.matmul(s, g)

            # Check if all values are <= 0
            if (c_prod <= 0).all():
                break
            else:
                p = torch.argmax(c_prod).item()
                c_comp[p] = c_comp[p] + 1
                xs.append(cues_values[p])
                # Convert to numpy for storage
                vecs.append(c_prod.detach().cpu().numpy())

            is_unigram_onset = len(xs)==1 and len(xs[0])==1 and xs[0]=='#'
            is_end = (xs[-1][-1]=='#') and (not is_unigram_onset)
            is_max_iter = i==(max_attempt-1)
            if is_end or is_max_iter:
                if is_max_iter:
                    print('The maximum number of iterations ({:d}) reached.'.format(max_attempt))
                break

        # Round vectors only at the end, not in every iteration
        vecs = [v.round(roundby) for v in vecs]
        df = pd.DataFrame(vecs).rename(columns={ i:j for i,j in enumerate(cues_values) })
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
            if path_gz.exists():
                path = path_gz
            elif path_csv.exists():
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


