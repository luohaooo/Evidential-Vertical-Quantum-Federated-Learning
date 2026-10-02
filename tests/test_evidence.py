import unittest

import torch

from evidence import CCRServer, TeleportationPauliNoise, evidence2prob


class EvidenceTest(unittest.TestCase):
    def test_plausibility_transform_shape(self):
        self.assertEqual(evidence2prob(2).shape, (2, 4))
        self.assertEqual(evidence2prob(4).shape, (4, 16))

    def test_zero_noise_is_identity(self):
        masses = torch.rand(3, 4)
        self.assertTrue(torch.equal(TeleportationPauliNoise(0.0)(masses), masses))

    def test_epsilon_point_75_fully_randomizes_two_bits(self):
        masses = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        expected = torch.full_like(masses, 0.25)
        self.assertTrue(
            torch.allclose(TeleportationPauliNoise(0.75)(masses), expected)
        )

    def test_ccr_server_returns_normalized_probabilities(self):
        masses = torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.1, 0.4]]]
        )
        probabilities = CCRServer(num_clients=2)(masses)
        self.assertEqual(probabilities.shape, (1, 2))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(1)))


if __name__ == "__main__":
    unittest.main()
