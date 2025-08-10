import unittest
import numpy as np
import ipynb.fs.full.assignment_1 as a1

class TestAssignment1Template(unittest.TestCase):
    def setUp(self):
        # Small test matrix (4x4)
        self.X = np.array([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [2, 2, 2, 3],
            [0, 1, 0, 1]
        ], dtype=float)

        self.X_2 = np.array([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [2, 2, 2, 2],
            [0, 1, 0, 1]
        ], dtype=float)

    def test_normalize_data(self):
        try:
            X_norm = a1.normalize_data(self.X)
        except NotImplementedError:
            self.skipTest('normalize_data not implemented')
        # Each row should have mean ~0 and std ~1
        means = np.mean(X_norm, axis=1)
        stds = np.std(X_norm, axis=1)
        np.testing.assert_allclose(means, 0, atol=1e-7)
        np.testing.assert_allclose(stds, 1, atol=1e-7)

    def test_normalize_data_zero_std(self):
        try:
            X_norm = a1.normalize_data(self.X_2)
        except NotImplementedError:
            self.skipTest('normalize_data not implemented')
        # The third row has zero std, should be all zeros after normalization
        self.assertTrue(np.allclose(X_norm[2], 0))
        # Other rows should have mean ~0 and std ~1
        means = np.mean(X_norm[[0,1,3]], axis=1)
        stds = np.std(X_norm[[0,1,3]], axis=1)
        np.testing.assert_allclose(means, 0, atol=1e-7)
        np.testing.assert_allclose(stds, 1, atol=1e-7)

    def test_euclidean_distance(self):
        try:
            dist = a1.euclidean_distance(self.X)
        except NotImplementedError:
            self.skipTest('euclidean_distance not implemented')
        self.assertEqual(dist.shape, (4, 4))
        self.assertAlmostEqual(dist[0, 1], dist[1, 0])
        self.assertAlmostEqual(dist[0, 0], 0)

    def test_haar_matrix(self):
        try:
            H = a1.haar_matrix(4)
        except NotImplementedError:
            self.skipTest('haar_matrix not implemented')
        self.assertEqual(H.shape, (4, 4))
        # Haar matrix should be orthogonal: H @ H.T = I
        I = np.eye(4)
        np.testing.assert_allclose(H @ H.T, I, atol=1e-7)

    def test_wavelet_transform(self):
        try:
            X_wavelet = a1.wavelet_transform(self.X, 4)
        except NotImplementedError:
            self.skipTest('wavelet_transform not implemented')
        self.assertEqual(X_wavelet.shape, self.X.shape)

    def test_pca(self):
        try:
            X_pca = a1.pca(self.X, 2)
        except NotImplementedError:
            self.skipTest('pca not implemented')
        # Should return shape (n_samples, n_components)
        self.assertEqual(X_pca.shape, (4, 2))

if __name__ == '__main__':
    unittest.main()
