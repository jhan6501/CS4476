import unittest
import numpy as np

from proj3_code import recover_rot_translation

class TestEssentialMatrixDecomposition(unittest.TestCase):

    def setUp(self):
        self.F = np.array([[100, 200, 1],
                   [100, 140, 1],
                   [500, 100, 1]])
        self.I = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])
        self.K = np.array([[100, 0, 240],
                          [0, 100, 320],
                          [0, 0, 1]])
        self.E = np.array([[ 1000000,  2000000,  8800100],
                             [ 1000000,  1400000,  6880100],
                             [ 5650000,  9290000, 43288561]])

    def test_recover_E_from_F(self):
        E = recover_rot_translation.recover_E_from_F(self.F, self.I)
        assert np.array_equal(E, self.F)
        E = recover_rot_translation.recover_E_from_F(self.F, self.K)
        assert np.array_equal(E, self.E)

    def test_recover_rot_translation_from_E(self):
        R1, R2, t = recover_rot_translation.recover_rot_translation_from_E(self.I)
        assert np.allclose([0., 0., 1.57079633], R1) or np.allclose([0., 0., 1.57079633], R2)
        assert np.allclose([-0., -0., -1.57079633], R1) or np.allclose([-0., -0., -1.57079633], R2)
        assert np.array_equal([0., 0., 1.], t)

        R1, R2, t = recover_rot_translation.recover_rot_translation_from_E(self.E)
        assert np.allclose([ 1.42572617, -1.69270725, -2.19161805], R1) or np.allclose([ 1.42572617, -1.69270725, -2.19161805], R2)
        assert np.allclose([-2.13337135,  0.89636235, -2.07923215], R1) or np.allclose([-2.13337135,  0.89636235, -2.07923215], R2)
        assert np.allclose([ 0.54960898, 0.80051726, -0.23896042], t)
