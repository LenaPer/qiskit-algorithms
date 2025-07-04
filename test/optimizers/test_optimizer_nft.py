# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test of NFT optimizer"""

import unittest
from test import QiskitAlgorithmsTestCase
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qiskit_algorithms.optimizers import NFT
from qiskit_algorithms.minimum_eigensolvers import VQE


class TestOptimizerNFT(QiskitAlgorithmsTestCase):
    """Test NFT optimizer using RY with VQE"""

    def setUp(self):
        super().setUp()
        self.qubit_op = SparsePauliOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )

    def test_nft(self):
        """Test NFT optimizer by using it"""

        vqe = VQE(StatevectorEstimator(), ansatz=RealAmplitudes(), optimizer=NFT())

        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)

        self.assertAlmostEqual(result.eigenvalue.real, -1.857275, places=6)


if __name__ == "__main__":
    unittest.main()
