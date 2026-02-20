import pandas as pd
import numpy as np
import re
from pathlib import Path

from . import mapping as lm
from . import performance as lp

__all__ = [
    "LDL",
    "concat_cues",
]

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

    def gen_cmat (self, words=None, gram=3, count=True, noise=False, freqs=None, cues=None):
        if not (words is None):
            self.words = words
        self.cmat = lm.gen_cmat(self.words, gram=gram, count=count, noise=noise, freqs=freqs, cues=cues)
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

    def gen_vmat (self, words=None, gram=3, cues=None):
        if not (words is None):
            self.words = words
        if cues is None:
            cues = lm.to_cues(self.words, gram=gram)
        self.vmat = lm.gen_vmat(cues=cues)
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

    def produce (self, gold, word=False, roundby=10, max_attempt=50, positive=False, apply_vmat=True, backend='auto', device=None, stop='convergence', tol=0.0):
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
        stop : {'convergence', 'boundary'}
            'convergence' -> Stop when no cue can improve the semantic
                approximation (all c_prod values <= tol).
            'boundary' -> Also stop when a cue ending with '#' is selected.
        tol : float
            Tolerance for the convergence check. The algorithm stops when all
            c_prod values are <= tol. Default is 0.0.
        """
        return lm.produce(gold, self.cmat, self.fmat, self.gmat, self.vmat,
                          word=word, roundby=roundby, max_attempt=max_attempt,
                          positive=positive, apply_vmat=apply_vmat,
                          backend=backend, device=device, stop=stop, tol=tol)

    def produce_batch (self, gold_batch, word=False, roundby=10, max_attempt=50, positive=False, apply_vmat=True, backend='auto', device=None, batch_size=None):
        """
        Produce output for multiple words using batch processing (Phase 2 optimization).

        Parameters
        ----------
        gold_batch : array-like
            Batch of target semantic vectors, shape (n_samples, semantic_dims).
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
            'numpy' -> Process batch sequentially using NumPy (falls back to single produce).
            'torch' -> PyTorch batch implementation (CPU/GPU depending on 'device').
            'auto'  -> Try torch+CUDA if available, else fall back to sequential NumPy.
        device : str or None
            For torch backend: 'cuda', 'cpu', etc. If None and backend is
            'torch' or 'auto', chooses 'cuda' if available, else 'cpu'.
        batch_size : int or None
            If provided, the batch will be processed in chunks of this size.

        Returns
        -------
        list of DataFrames
            One DataFrame per word in the batch.
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
            return self._produce_batch_torch(gold_batch, word, roundby, max_attempt, positive, apply_vmat, device, batch_size)
        else:
            # Fallback to sequential processing
            results = []
            for gold in gold_batch:
                results.append(self.produce(gold, word=word, roundby=roundby, max_attempt=max_attempt, positive=positive, apply_vmat=apply_vmat, backend='numpy'))
            return results

    def _produce_batch_torch (self, gold_batch, word, roundby, max_attempt, positive, apply_vmat, device, batch_size):
        """
        PyTorch batch implementation of produce with GPU support (Phase 2 optimization).
        Processes multiple words in parallel with masking for variable-length iterations.
        """
        if not isinstance(gold_batch, np.ndarray):
            gold_batch = np.array(gold_batch)

        # Handle batch_size parameter for chunked processing
        if batch_size is not None and batch_size < gold_batch.shape[0]:
            results = []
            for i in range(0, gold_batch.shape[0], batch_size):
                chunk = gold_batch[i:i+batch_size]
                results.extend(self._produce_batch_torch(chunk, word, roundby, max_attempt, positive, apply_vmat, device, None))
            return results

        # Move data to torch tensors on specified device
        gold_tensor = torch.as_tensor(gold_batch, dtype=torch.float32, device=device)
        fmat_tensor = torch.as_tensor(self.fmat.values, dtype=torch.float32, device=device)
        gmat_tensor = torch.as_tensor(self.gmat.values, dtype=torch.float32, device=device)

        if apply_vmat:
            vmat_tensor = torch.as_tensor(self.vmat.values, dtype=torch.float32, device=device)

        cues_values = self.cmat.cues.values
        n_cues = cues_values.size
        batch_size_actual = gold_tensor.shape[0]

        # Pre-allocate tensors on GPU
        c_comp = torch.zeros((batch_size_actual, n_cues), dtype=torch.float32, device=device)
        xs_indices = torch.zeros((batch_size_actual, max_attempt), dtype=torch.long, device=device)
        vecs_gpu = torch.zeros((batch_size_actual, max_attempt, n_cues), dtype=torch.float32, device=device)
        p_tensor = torch.full((batch_size_actual,), -1, dtype=torch.long, device=device)

        # Track which words are still active and their iteration counts
        active_mask = torch.ones(batch_size_actual, dtype=torch.bool, device=device)
        iteration_counts = torch.zeros(batch_size_actual, dtype=torch.long, device=device)

        for i in range(max_attempt):
            if not active_mask.any():
                break  # All words have finished

            # Compute for all words in batch
            s0 = torch.matmul(c_comp, fmat_tensor)  # (batch_size, semantic_dims)
            if positive:
                s0 = torch.clamp(s0, min=0)
            s = gold_tensor - s0

            if apply_vmat:
                # Element-wise multiplication with vmat rows for each word
                # Use advanced indexing to get the right vmat row for each word
                vmat_rows = vmat_tensor[p_tensor]  # (batch_size, n_cues)
                g = gmat_tensor.unsqueeze(0) * vmat_rows.unsqueeze(1)  # (batch_size, semantic_dims, n_cues)
                c_prod = torch.bmm(s.unsqueeze(1), g).squeeze(1)  # (batch_size, n_cues)
            else:
                c_prod = torch.matmul(s, gmat_tensor)  # (batch_size, n_cues)

            # Check which words should stop due to all values <= 0
            all_negative = (c_prod <= 0).all(dim=1)
            active_mask = active_mask & ~all_negative

            # For active words, find argmax and update
            if active_mask.any():
                p_tensor_new = torch.argmax(c_prod, dim=1)
                # Only update active words
                p_tensor = torch.where(active_mask, p_tensor_new, p_tensor)

                # In-place update for active words
                batch_indices = torch.arange(batch_size_actual, device=device)
                c_comp[batch_indices[active_mask], p_tensor[active_mask]] += 1

                # Store results for active words
                active_indices = batch_indices[active_mask]
                for idx in active_indices:
                    iter_idx = iteration_counts[idx]
                    xs_indices[idx, iter_idx] = p_tensor[idx]
                    vecs_gpu[idx, iter_idx] = c_prod[idx]
                    iteration_counts[idx] += 1

            # Check ending conditions (need to sync with CPU for string checks)
            # This is necessary but only happens once per iteration for all words
            p_cpu = p_tensor.cpu().numpy()
            for idx in range(batch_size_actual):
                if not active_mask[idx]:
                    continue

                iter_count = iteration_counts[idx].item()
                if iter_count == 0:
                    continue

                selected_cue = cues_values[p_cpu[idx]]
                is_unigram_onset = iter_count == 1 and len(selected_cue) == 1 and selected_cue == '#'
                is_end = (selected_cue[-1] == '#') and (not is_unigram_onset)
                is_max_iter = i == (max_attempt - 1)

                if is_end or is_max_iter:
                    active_mask[idx] = False
                    if is_max_iter:
                        print(f'Word {idx}: The maximum number of iterations ({max_attempt:d}) reached.')

        # Transfer results to CPU and create DataFrames
        xs_indices_cpu = xs_indices.cpu().numpy()
        vecs_cpu = vecs_gpu.cpu().numpy()
        iteration_counts_cpu = iteration_counts.cpu().numpy()

        results = []
        for idx in range(batch_size_actual):
            n_iters = iteration_counts_cpu[idx]
            xs = [cues_values[xs_indices_cpu[idx, i]] for i in range(n_iters)]
            vecs = [vecs_cpu[idx, i].round(roundby) for i in range(n_iters)]

            df = pd.DataFrame(vecs).rename(columns={ i:j for i,j in enumerate(cues_values) })
            hdr = pd.Series(xs).to_frame(name='Selected')
            df = pd.concat([hdr, df], axis=1)
            if word:
                df = concat_cues(df.Selected)
            results.append(df)

        return results

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


