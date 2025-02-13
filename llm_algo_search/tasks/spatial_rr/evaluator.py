import torch


class SpatialRREvaluator:
    def evaluate(self, cfg, spatial_rr_cls):
        """
        Evaluate spatial reduced representation.
        Test how accurately distinct points can be bound and recovered from a single vector.
        """
        all_vec_dims = (128, 256, 512, 1024)
        all_num_positions = (32, 64, 128, 256, 512)
        num_samples = 5
        position_dims = 3
        radius = 20

        all_metrics = []
        for vec_dims in all_vec_dims:
            this_dim_metrics = []
            all_metrics.append(this_dim_metrics)
            for num_positions in all_num_positions:
                srr = spatial_rr_cls(num_positions, position_dims, vec_dims)

                errs = []
                for _ in range(num_samples):
                    points = (2 * torch.rand(num_positions, position_dims) - 1) * radius
                    bundled = srr.bundle(points)
                    recons = srr.unbundle(bundled)
                    err = torch.sqrt((points - recons) ** 2)
                    errs.append(err)
                mean_rmserr = torch.mean(torch.stack(errs))
                this_dim_metrics.append(mean_rmserr.item())

        return {
            'vector_dims': all_vec_dims,
            'num_positions': all_num_positions,
            'rms_err': all_metrics,
        }