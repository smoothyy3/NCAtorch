import os
from itertools import groupby

import numpy as np
import nibabel as nib
import cv2
import torch

from nca.data.datasets.base_dataset import NCADataset


class DecathlonDataset(NCADataset):
    """
    2D slices from Medical Segmentation Decathlon volumes.

    Emits (seed, cond, target) per NCAtorch convention:
        seed (channel_n, H, W): zeros with the image in channel 0
        cond scalar 0.0:   no conditioning
        target (1, H, W): binarised mask

    Slice on one axis, resize (cubic for images, nearest for labels), three intensity normalisations, 
    and label binarisation.
    """

    def __init__(self, root, channel_n, size=(64, 64), slice_axis=2,train=True, train_frac=0.7, cache_volumes=True, device="cpu"):
        self.images_path = os.path.join(root, "imagesTr")
        self.labels_path = os.path.join(root, "labelsTr")
        self.channel_n = channel_n
        self.size = tuple(size)
        self.slice_axis = slice_axis
        # Slice-then-resize is equivalent to Med-NCA resize-then-slice ONLY
        # because rescale3d resizes axes 0 and 1 and leaves axis 2 untouched.
        # It is NOT equivalent for slice_axis 0 or 1.
        if slice_axis != 2:
            raise ValueError(
                "slice-then-resize is only equivalent to Med-NCA's ordering for "
                "slice_axis=2; resize the whole volume first for other axes."
            )
        self.train = train
        self.train_frac = train_frac
        self.device = device

        self._cache = {} if cache_volumes else None
        self.index = self._build_index()

    def _volume_files(self):
        """
        Sorted .nii.gz filenames, excluding hidden entries.
        """
        return sorted(
            f for f in os.listdir(self.images_path)
            if f.endswith(".nii.gz") and not f.startswith(".")
        )

    def _build_index(self):
        """
        Flat list of (filename, slice_idx) for this split.
        """
        files = self._volume_files()
        n_train = int(len(files) * self.train_frac)
        files = files[:n_train] if self.train else files[n_train:]

        index = []
        for f in files:
            n_slices = nib.load(os.path.join(self.images_path, f)).shape[self.slice_axis]
            index.extend((f, s) for s in range(n_slices))
        return index

    def __len__(self):
        return len(self.index)

    def volume_groups(self):
        """Group this split's flat index by source volume, for per-patient eval.

        Returns:
            list of ``(filename, [dataset_index, ...])``, volumes in the same
            order as ``_volume_files()`` and slices in ascending slice order.
        """
        return [
            (fname, [i for i, _ in group])
            for fname, group in groupby(
                enumerate(self.index), key=lambda item: item[1][0]
            )
        ]

    def _load_volume(self, path):
        if self._cache is not None and path in self._cache:
            return self._cache[path]
        vol = nib.load(path).get_fdata()
        if vol.ndim == 4:
            vol = vol[..., 0]
        if self._cache is not None:
            self._cache[path] = vol
        return vol

    def _take_slice(self, vol, slice_idx):
        if self.slice_axis == 0:
            return vol[slice_idx, :, :]
        if self.slice_axis == 1:
            return vol[:, slice_idx, :]
        return vol[:, :, slice_idx]

    @staticmethod
    def _normalise(img):
        """Med-NCA's three intensity normalisations.

        1. cv2 min-max to [0, 1]
        2. z-normalisation (SKIPPED when the slice is all-zero)
        3. percentile clip (0.5, 99.5) then rescale to [0, 1]
        """
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if img.sum() > 0:
            img = (img - img.mean()) / img.std()

        lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
        img = np.clip(img, lo, hi)
        img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)

        return img.astype(np.float32)

    def __getitem__(self, idx):
        fname, slice_idx = self.index[idx]

        img_vol = self._load_volume(os.path.join(self.images_path, fname))
        lbl_vol = self._load_volume(os.path.join(self.labels_path, fname))

        img = self._take_slice(img_vol, slice_idx)
        lbl = self._take_slice(lbl_vol, slice_idx)

        # cv2.resize takes dsize as (width, height)
        dsize = (self.size[1], self.size[0])
        img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        lbl = cv2.resize(lbl, dsize=dsize, interpolation=cv2.INTER_NEAREST)

        img = self._normalise(img)
        lbl = (lbl != 0).astype(np.float32)

        seed = torch.zeros(self.channel_n, *self.size, dtype=torch.float32)
        seed[0] = torch.from_numpy(img)

        target = torch.from_numpy(lbl).unsqueeze(0)

        return seed, torch.tensor(0.0), target