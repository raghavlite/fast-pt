# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import FairseqDataset, data_utils


def collate(samples, pad_idx, eos_idx, src_domain_idx, fixed_pad_length=None, pad_to_bsz=None):
    if len(samples) == 0:
        return {}

    def merge(key, is_list=False):
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                res.append(
                    data_utils.collate_tokens(
                        [s[key][i] for s in samples],
                        pad_idx,
                        eos_idx,
                        left_pad=False,
                        pad_to_length=fixed_pad_length,
                        pad_to_bsz=pad_to_bsz,
                    )
                )
            return res
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
                pad_to_length=fixed_pad_length,
                pad_to_bsz=pad_to_bsz,
            )

    src_tokens = merge("source")
    if samples[0]["target"] is not None:
        is_target_list = isinstance(samples[0]["target"], list)
        target = merge("target", is_target_list)
    else:
        target = src_tokens
    
    if("IRL_losses" in samples[0]):
        return {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "nsentences": len(samples),
            "ntokens": sum(len(s["source"]) for s in samples),
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
                "src_domain_idx": src_domain_idx
            },
            "target": target,
            "IRL_losses": torch.vstack([s["IRL_losses"] for s in samples]),
        }
    else:
        return {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "nsentences": len(samples),
            "ntokens": sum(len(s["source"]) for s in samples),
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
                "src_domain_idx": src_domain_idx
            },
            "target": target,
        }


class MonodomainDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monodomain data.

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching
            (default: True).
    """

    def __init__(
        self,
        dataset,
        sizes,
        src_vocab,
        tgt_vocab=None,
        add_eos_for_other_targets=False,
        shuffle=False,
        targets=None,
        add_domain_token=False,
        fixed_pad_length=None,
        pad_to_bsz=None,
        src_domain_idx=None,
        tgt_domain_idx=None,
        src_domain_token=None,
        tgt_domain_token=None, truncate_len=None
    ):
        if(truncate_len is not None):
            # import ipdb; ipdb.set_trace()
            # dataset = dataset[:truncate_len]
            sizes = sizes[:truncate_len]

        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = src_vocab
        self.tgt_vocab = tgt_vocab or src_vocab
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle
        self.add_domain_token = add_domain_token
        self.fixed_pad_length = fixed_pad_length
        self.pad_to_bsz = pad_to_bsz
        self.src_domain_idx = src_domain_idx
        self.tgt_domain_idx = tgt_domain_idx
        self.src_domain_token = src_domain_token
        self.tgt_domain_token = tgt_domain_token        
        assert targets is None or all(
            t in {"self", "future", "past"} for t in targets
        ), "targets must be none or one of 'self', 'future', 'past'"
        if targets is not None and len(targets) == 0:
            targets = None
        self.targets = targets
        self.IRL_losses=None

    def set_IRL_losses(self, IRL_inputs_05, IRL_losses):
        try:
            assert len(IRL_losses) == len(self)
            self.IRL_losses = IRL_losses
            assert (torch.sum(IRL_losses[0]>0.0)==torch.sum(self[0]['source']>1))
            assert (torch.sum(IRL_losses[-1]>0.0)==torch.sum(self[-1]['source']>1))

            # for idx1 in range(len(IRL_losses)):
            #     assert (IRL_inputs_05[idx1]==self[idx1]['source'][:5]).all(), (IRL_inputs_05[idx1], self[idx1]['source'][:5])

            # for idx1 in range(len(IRL_losses)):
            #     assert ((IRL_losses[idx1]>-0.00001)==(self[idx1]['source']>1)).all()
            # print("IRL losses loaded", flush=True)
        except:
            print("IRL losses loading failed", flush=True)
            self.IRL_losses = None
            import ipdb; ipdb.set_trace()
        return

    def __getitem__(self, index):
        if self.targets is not None:
            # *future_target* is the original sentence
            # *source* is shifted right by 1 (maybe left-padded with eos)
            # *past_target* is shifted right by 2 (left-padded as needed)
            #
            # Left-to-right language models should condition on *source* and
            # predict *future_target*.
            # Right-to-left language models should condition on *source* and
            # predict *past_target*.
            source, future_target, past_target = self.dataset[index]
            source, target = self._make_source_target(
                source, future_target, past_target
            )
        else:
            source = self.dataset[index]
            target = None
        source, target = self._maybe_add_bos(source, target)
        if(self.IRL_losses is not None):
            res = {"id": index, "source": source, "target": target, "IRL_losses":self.IRL_losses[index]}
        else:
            res = {"id": index, "source": source, "target": target}
        
        return res

    def __len__(self):
        return len(self.sizes)
        # return len(self.dataset)

    def _make_source_target(self, source, future_target, past_target):
        if self.targets is not None:
            target = []

            if (
                self.add_eos_for_other_targets
                and (("self" in self.targets) or ("past" in self.targets))
                and source[-1] != self.vocab.eos()
            ):
                # append eos at the end of source
                source = torch.cat([source, source.new([self.vocab.eos()])])

                if "future" in self.targets:
                    future_target = torch.cat(
                        [future_target, future_target.new([self.vocab.pad()])]
                    )
                if "past" in self.targets:
                    # first token is before the start of sentence which is only used in "none" break mode when
                    # add_eos_for_other_targets is False
                    past_target = torch.cat(
                        [
                            past_target.new([self.vocab.pad()]),
                            past_target[1:],
                            source[-2, None],
                        ]
                    )

            for t in self.targets:
                if t == "self":
                    target.append(source)
                elif t == "future":
                    target.append(future_target)
                elif t == "past":
                    target.append(past_target)
                else:
                    raise Exception("invalid target " + t)

            if len(target) == 1:
                target = target[0]
        else:
            target = future_target

        return source, self._filter_vocab(target)

    def _maybe_add_bos(self, source, target):
        if self.add_domain_token:
            # src_lang_idx and tgt_lang_idx are passed in for multilingual LM, with the
            # first token being an lang_id token.
            bos = self.src_domain_token or self.vocab.bos()
            source = torch.cat([source.new([bos]), source])
            if target is not None:
                tgt_bos = self.tgt_domain_token or self.tgt_vocab.bos()
                target = torch.cat([target.new([tgt_bos]), target])
        return source, target

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        return self.sizes[indices]

    
    def _filter_vocab(self, target):
        if len(self.tgt_vocab) != len(self.vocab):

            def _filter(target):
                mask = target.ge(len(self.tgt_vocab))
                if mask.any():
                    target[mask] = self.tgt_vocab.unk()
                return target

            if isinstance(target, list):
                return [_filter(t) for t in target]
            return _filter(target)
        return target

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        return collate(
            samples, 
            self.vocab.pad(), 
            self.vocab.eos(), 
            [self.src_domain_idx] * len(samples),
            self.fixed_pad_length,
            self.pad_to_bsz
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.RandomState(seed=torch.distributed.get_rank()).permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(indices)
