from torch.utils.data import Dataset
import torch


class LatentDataset(Dataset):
    """
    Simple dataset for loading pre-encoded latent pairs saved by encode_latents.py.
    Each item in meta_data should be a dict with key 'video' pointing to a .pt file path.
    The .pt file should contain a dict with keys 'x0' and 'y' (both tensors).
    """
    def __init__(self, meta_data, squeeze_dim0=False) -> None:
        self.meta_data = meta_data
        self.squeeze_dim0 = squeeze_dim0

    def __len__(self):
        return len(self.meta_data)

    def _getitem_fn(self, idx):
        path = self.meta_data[idx]['video']
        example = torch.load(path, weights_only=False)
        if self.squeeze_dim0:
            for k, v in example.items():
                if isinstance(v, torch.Tensor):
                    example[k] = v.squeeze(0)
        return example

    def __getitem__(self, idx):
        n_tries = 10
        for _ in range(n_tries):
            try:
                return self._getitem_fn(idx)
            except Exception as e:
                print(f"Error loading latent file {self.meta_data[idx]['video']}: {e}")
                idx = torch.randint(0, len(self.meta_data), (1,)).item()
        raise RuntimeError(
            f"Failed to load latent after {n_tries} tries (last tried: {self.meta_data[idx]['video']})"
        )
