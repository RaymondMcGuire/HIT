"""Multiresolution IsoSurface Extraction (MISE), NumPy implementation.

Same interface and refinement rule as ``libmise.MISE`` from Occupancy Networks
(Mescheder et al., CVPR 2019), which upstream HIT took from the LEAP package.
That version is a Cython extension; this one needs no compiler.

The grid has ``resolution + 1`` lattice points per axis, with
``resolution = resolution_0 * 2**depth``. Evaluation starts on the coarse
lattice. A voxel is refined when its eight corners lie on both sides of
``threshold``; refining it adds the lattice points of its eight children.
"""

import numpy as np

_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64)


class MISE:

    def __init__(self, resolution_0, depth, threshold):
        self.resolution_0 = resolution_0
        self.depth = depth
        self.threshold = threshold
        self.voxel_size_0 = 1 << depth
        self.resolution = resolution_0 * self.voxel_size_0

        n = self.resolution + 1
        self._values = np.full((n, n, n), np.nan, dtype=np.float64)

        # Leaf voxels of the current level: origins [M, 3] and their edge length.
        axis = np.arange(resolution_0, dtype=np.int64) * self.voxel_size_0
        self._voxels = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
        self._voxel_size = self.voxel_size_0

        axis = np.arange(resolution_0 + 1, dtype=np.int64) * self.voxel_size_0
        self._pending = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)

    def query(self):
        """Lattice points [N, 3] (int64) whose value is needed next; empty when done."""
        return self._pending.copy()

    def update(self, points, values):
        points = np.asarray(points, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        assert points.shape[0] == values.shape[0]
        self._values[points[:, 0], points[:, 1], points[:, 2]] = values

        s = self._voxel_size
        if s == 1 or self._voxels.shape[0] == 0:
            self._pending = np.zeros((0, 3), dtype=np.int64)
            return

        # Voxels whose corners straddle the threshold
        corners = self._voxels[:, None, :] + s * _CORNERS[None]            # M, 8, 3
        inside = self._values[corners[..., 0], corners[..., 1], corners[..., 2]] >= self.threshold
        active = self._voxels[inside.any(axis=1) & ~inside.all(axis=1)]

        # Subdivide them
        h = s // 2
        self._voxels = (active[:, None, :] + h * _CORNERS[None]).reshape(-1, 3)
        self._voxel_size = h

        new_points = (self._voxels[:, None, :] + h * _CORNERS[None]).reshape(-1, 3)
        new_points = np.unique(new_points, axis=0)
        unknown = np.isnan(self._values[new_points[:, 0], new_points[:, 1], new_points[:, 2]])
        self._pending = new_points[unknown]

    def to_dense(self):
        """Dense [resolution+1]^3 grid; lattice points that were never evaluated
        take the value of their predecessor along x, then y, then z."""
        out = self._values.copy()
        s0 = self.voxel_size_0
        n = out.shape[0]

        for i in range(1, n - 1):
            sub = out[i, ::s0, ::s0]
            nan = np.isnan(sub)
            sub[nan] = out[i - 1, ::s0, ::s0][nan]
        for j in range(1, n - 1):
            sub = out[:, j, ::s0]
            nan = np.isnan(sub)
            sub[nan] = out[:, j - 1, ::s0][nan]
        for k in range(1, n - 1):
            sub = out[:, :, k]
            nan = np.isnan(sub)
            sub[nan] = out[:, :, k - 1][nan]
        return out
