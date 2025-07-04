# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# =============================================================================

"""Test QFI."""

import unittest
from test import QiskitAlgorithmsTestCase

from ddt import ddt, data
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.primitives import StatevectorEstimator

from qiskit_algorithms.gradients import LinCombQGT, ReverseQGT, QFI, DerivativeType


@ddt
class TestQFI(QiskitAlgorithmsTestCase):
    """Test QFI"""

    def setUp(self):
        super().setUp()
        self.estimator = StatevectorEstimator()
        self.lcu_qgt = LinCombQGT(self.estimator, derivative_type=DerivativeType.REAL)
        self.reverse_qgt = ReverseQGT(derivative_type=DerivativeType.REAL)

    def test_qfi(self):
        """Test if the quantum fisher information calculation is correct for a simple test case.
        QFI = [[1, 0], [0, 1]] - [[0, 0], [0, cos^2(a)]]
        """
        # create the circuit
        a, b = Parameter("a"), Parameter("b")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(a, 0)
        qc.rx(b, 0)

        param_list = [[np.pi / 4, 0.1], [np.pi, 0.1], [np.pi / 2, 0.1]]
        correct_values = [[[1, 0], [0, 0.5]], [[1, 0], [0, 0]], [[1, 0], [0, 1]]]

        qfi = QFI(self.lcu_qgt)
        for i, param in enumerate(param_list):
            qfis = qfi.run([qc], [param]).result().qfis
            np.testing.assert_allclose(qfis[0], correct_values[i], atol=1e-3)

    def test_qfi_phase_fix(self):
        """Test the phase-fix argument in the QFI calculation"""
        # create the circuit
        a, b = Parameter("a"), Parameter("b")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(a, 0)
        qc.rx(b, 0)

        param = [np.pi / 4, 0.1]
        # test for different values
        correct_values = [[1, 0], [0, 1]]
        qgt = LinCombQGT(self.estimator, phase_fix=False)
        qfi = QFI(qgt)
        qfis = qfi.run([qc], [param]).result().qfis
        np.testing.assert_allclose(qfis[0], correct_values, atol=1e-3)

    @data("lcu", "reverse")
    def test_qfi_maxcut(self, qgt_kind):
        """Test the QFI for a simple MaxCut problem.

        This is interesting because it contains the same parameters in different gates.
        """
        # create maxcut circuit for the hamiltonian
        # H = (I ^ I ^ Z ^ Z) + (I ^ Z ^ I ^ Z) + (Z ^ I ^ I ^ Z) + (I ^ Z ^ Z ^ I)

        x = ParameterVector("x", 2)
        ansatz = QuantumCircuit(4)

        # initial hadamard layer
        ansatz.h(ansatz.qubits)

        # e^{iZZ} layers
        def expiz(qubit0, qubit1):
            ansatz.cx(qubit0, qubit1)
            ansatz.rz(2 * x[0], qubit1)
            ansatz.cx(qubit0, qubit1)

        expiz(2, 1)
        expiz(3, 0)
        expiz(2, 0)
        expiz(1, 0)

        # mixer layer with RX gates
        for i in range(ansatz.num_qubits):
            ansatz.rx(2 * x[1], i)

        reference = np.array([[16.0, -5.551], [-5.551, 18.497]])
        param = [0.4, 0.69]

        qgt = self.lcu_qgt if qgt_kind == "lcu" else self.reverse_qgt
        qfi = QFI(qgt)
        qfi_result = qfi.run([ansatz], [param]).result().qfis
        np.testing.assert_array_almost_equal(qfi_result[0], reference, decimal=3)

    def test_precision(self):
        """Test QFI's precision option"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qgt = LinCombQGT(estimator=self.estimator, precision=0.1)

        with self.subTest("QGT"):
            qfi = QFI(qgt=qgt)
            precision = qfi.precision
            result = qfi.run([qc], [[1]]).result()
            self.assertEqual(result.precision, 0.1)
            self.assertEqual(precision, None)

        with self.subTest("QFI init"):
            qfi = QFI(qgt=qgt, precision=0.2)
            result = qfi.run([qc], [[1]]).result()
            precision = qfi.precision
            self.assertEqual(result.precision, 0.2)
            self.assertEqual(precision, 0.2)

        with self.subTest("QFI update"):
            qfi = QFI(qgt, precision=0.2)
            qfi.precision = 0.1
            precision = qfi.precision
            result = qfi.run([qc], [[1]]).result()
            self.assertEqual(result.precision, 0.1)
            self.assertEqual(precision, 0.1)

        with self.subTest("QFI run"):
            qfi = QFI(qgt=qgt, precision=0.2)
            result = qfi.run([qc], [[0]], precision=0.3).result()
            precision = qfi.precision
            self.assertEqual(result.precision, 0.3)
            self.assertEqual(precision, 0.2)


if __name__ == "__main__":
    unittest.main()
