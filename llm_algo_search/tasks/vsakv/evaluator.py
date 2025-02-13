from functools import partial, reduce

import numpy as np
import torch


class VSAKVEvaluator:
    def evaluate(self, cfg, vsa_cls):
        """
        Evaluate VSA on key/value store performance.
        Test how many key/value pairs can be bundled and recovered.
        """
        all_vec_dims = (256, 512, 1024, 2048)
        all_num_pairs = (32, 64, 128, 256, 512, 1024)
        vsa = vsa_cls()

        all_metrics = []
        for vec_dims in all_vec_dims:
            this_dim_metrics = []
            all_metrics.append(this_dim_metrics)
            for num_pairs in all_num_pairs:
                basis_vecs = vsa.initialize(num_pairs * 2, vec_dims)  # (2 * num_pairs, vec_dims)
                key_bvs, value_bvs = torch.chunk(basis_vecs, 2)  # (num_pairs, vec_dims)
                bound = vsa.bind(key_bvs, value_bvs)  # shape (num_pairs, vec_dims)
                bundled = reduce(vsa.bundle, bound.unsqueeze(1))  # shape (1, vec_dims)
                unbound = torch.stack(list(
                    map(partial(vsa.unbind, bundled), key_bvs.unsqueeze(1))  # shape (num_pairs, vec_dims)
                )).squeeze(1)
                similarity = vsa.similarity(unbound, value_bvs)
                pred_ids = similarity.argmax(-1)
                target_ids = torch.arange(pred_ids.shape[0])
                num_correct = (pred_ids == target_ids).sum()
                num_total = pred_ids.shape[0]
                accuracy = num_correct / num_total
                this_dim_metrics.append(accuracy.item())

        return {
            'vector_dims': all_vec_dims,
            'num_kv_pairs': all_num_pairs,
            'accuracy': all_metrics
        }