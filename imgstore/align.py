from __future__ import print_function

import itertools
import functools

import numpy as np


class StoreAligner(object):
    MISSING_POLICY_DROP = "drop"
    MISSING_POLICY_HOLD = "zoh"
    MISSING_POLICY_BLACK = "black"

    def __init__(self, *stores):
        self._fns = []
        self._missing_policy = None
        self._stores = stores
        sizes = {s.image_shape for s in stores}
        assert len(sizes) == 1
        self._store_shape = sizes.pop()

    def extract_common_frames(self, policy):
        assert policy in (
            StoreAligner.MISSING_POLICY_DROP,
            StoreAligner.MISSING_POLICY_HOLD,
            StoreAligner.MISSING_POLICY_BLACK,
        )

        fns_arr = [
            s.get_frame_metadata()["frame_number"] for s in self._stores
        ]
        if self._missing_policy == StoreAligner.MISSING_POLICY_DROP:
            self._fns = functools.reduce(np.intersect1d, fns_arr)
        else:
            fns_all = np.fromiter(
                itertools.chain.from_iterable(fns_arr), dtype=int
            )
            self._fns = np.unique(fns_all)

        self._missing_policy = policy
        return self._fns

    def iter_imgs(self, return_times=False):
        assert len(self._fns)
        assert self._missing_policy is not None

        black_img = np.zeros(self._store_shape, dtype=np.uint8)
        last_imgs = [black_img for _ in range(len(self._stores))]

        for fn in self._fns:
            imgs = []
            ts = []
            for i, store in enumerate(self._stores):
                try:
                    img, (_, t) = store.get_image(
                        frame_number=fn, exact_only=True
                    )
                    last_imgs[i] = img
                    imgs.append(img)
                    ts.append(t)
                except ValueError:
                    print("store %d missing frame %s" % (i, fn))
                    ts.append(np.nan)
                    if (
                        self._missing_policy
                        == StoreAligner.MISSING_POLICY_HOLD
                    ):
                        imgs.append(last_imgs[i])
                    elif (
                        self._missing_policy
                        == StoreAligner.MISSING_POLICY_BLACK
                    ):
                        imgs.append(black_img)
                    elif (
                        self._missing_policy
                        == StoreAligner.MISSING_POLICY_DROP
                    ):
                        continue
                    else:
                        raise NotImplementedError

            if len(imgs) != len(self._stores):
                continue

            if return_times:
                yield fn, imgs, ts
            else:
                yield fn, imgs
