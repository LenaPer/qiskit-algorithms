# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# =============================================================================

"""Test Estimator Gradients"""

import unittest
from test import QiskitAlgorithmsTestCase

import numpy as np
from ddt import ddt, data, unpack
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.circuit.library.standard_gates import RXXGate, RYYGate, RZXGate, RZZGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.quantum_info.random import random_pauli_list
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2


from qiskit_algorithms.gradients import (
    FiniteDiffEstimatorGradient,
    LinCombEstimatorGradient,
    ParamShiftEstimatorGradient,
    SPSAEstimatorGradient,
    ReverseEstimatorGradient,
    DerivativeType,
)

from .logging_primitives import LoggingEstimator

gradient_factories = [
    lambda estimator: FiniteDiffEstimatorGradient(estimator, epsilon=1e-6, method="central"),
    lambda estimator: FiniteDiffEstimatorGradient(estimator, epsilon=1e-6, method="forward"),
    lambda estimator: FiniteDiffEstimatorGradient(estimator, epsilon=1e-6, method="backward"),
    ParamShiftEstimatorGradient,
    LinCombEstimatorGradient,
    lambda estimator: ReverseEstimatorGradient(),  # does not take an estimator!
]

THREE_QUBITS_BACKEND = GenericBackendV2(num_qubits=3, coupling_map=[[0, 1], [1, 2]], seed=54)


@ddt
class TestEstimatorGradient(QiskitAlgorithmsTestCase):
    """Test Estimator Gradient"""

    @data(*gradient_factories)
    def test_gradient_operators(self, grad):
        """Test the estimator gradient for different operators"""
        estimator = StatevectorEstimator()
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        gradient = grad(estimator)
        op = SparsePauliOp.from_list([("Z", 1)])
        correct_result = -1 / np.sqrt(2)
        param = [np.pi / 4]
        value = gradient.run([qc], [op], [param]).result().gradients[0]
        self.assertAlmostEqual(value[0], correct_result, 3)
        op = SparsePauliOp.from_list([("Z", 1)])
        value = gradient.run([qc], [op], [param]).result().gradients[0]
        self.assertAlmostEqual(value[0], correct_result, 3)

    @data(*gradient_factories)
    def test_single_circuit_observable(self, grad):
        """Test the estimator gradient for a single circuit and observable"""
        estimator = StatevectorEstimator()
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        gradient = grad(estimator)
        op = SparsePauliOp.from_list([("Z", 1)])
        correct_result = -1 / np.sqrt(2)
        param = [np.pi / 4]
        value = gradient.run(qc, op, [param]).result().gradients[0]
        self.assertAlmostEqual(value[0], correct_result, 3)

    @data(*gradient_factories)
    def test_gradient_p(self, grad):
        """Test the estimator gradient for p"""
        estimator = StatevectorEstimator()
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.p(a, 0)
        qc.h(0)
        gradient = grad(estimator)
        op = SparsePauliOp.from_list([("Z", 1)])
        param_list = [[np.pi / 4], [0], [np.pi / 2]]
        correct_results = [[-1 / np.sqrt(2)], [0], [-1]]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            for j, value in enumerate(gradients):
                self.assertAlmostEqual(value, correct_results[i][j], 3)

    @data(*gradient_factories)
    def test_gradient_u(self, grad):
        """Test the estimator gradient for u"""
        estimator = StatevectorEstimator()
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.u(a, b, c, 0)
        qc.h(0)
        gradient = grad(estimator)
        op = SparsePauliOp.from_list([("Z", 1)])

        param_list = [[np.pi / 4, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]]
        correct_results = [[-0.70710678, 0.0, 0.0], [-0.35355339, -0.85355339, -0.85355339]]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            for j, value in enumerate(gradients):
                self.assertAlmostEqual(value, correct_results[i][j], 3)

    @data(*gradient_factories)
    def test_gradient_efficient_su2(self, grad):
        """Test the estimator gradient for EfficientSU2"""
        estimator = StatevectorEstimator()
        qc = EfficientSU2(2, reps=1)
        op = SparsePauliOp.from_list([("ZI", 1)])
        gradient = grad(estimator)
        param_list = [
            [np.pi / 4 for param in qc.parameters],
            [np.pi / 2 for param in qc.parameters],
        ]
        correct_results = [
            [
                -0.35355339,
                -0.70710678,
                0,
                0.35355339,
                0,
                -0.70710678,
                0,
                0,
            ],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_2qubit_gate(self, grad):
        """Test the estimator gradient for 2 qubit gates"""
        estimator = StatevectorEstimator()
        for gate in [RXXGate, RYYGate, RZZGate, RZXGate]:
            param_list = [[np.pi / 4], [np.pi / 2]]
            correct_results = [
                [-0.70710678],
                [-1],
            ]
            op = SparsePauliOp.from_list([("ZI", 1)])
            for i, param in enumerate(param_list):
                a = Parameter("a")
                qc = QuantumCircuit(2)
                gradient = grad(estimator)

                if gate is RZZGate:
                    qc.h([0, 1])
                    qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                    qc.h([0, 1])
                else:
                    qc.append(gate(a), [qc.qubits[0], qc.qubits[1]], [])
                gradients = gradient.run([qc], [op], [param]).result().gradients[0]
                np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_parameter_coefficient(self, grad):
        """Test the estimator gradient for parameter variables with coefficients"""
        estimator = StatevectorEstimator()
        qc = RealAmplitudes(num_qubits=2, reps=1)
        qc.rz(qc.parameters[0].exp() + 2 * qc.parameters[1], 0)
        qc.rx(3.0 * qc.parameters[0] + qc.parameters[1].sin(), 1)
        qc.u(qc.parameters[0], qc.parameters[1], qc.parameters[3], 1)
        qc.p(2 * qc.parameters[0] + 1, 0)
        qc.rxx(qc.parameters[0] + 2, 0, 1)
        gradient = grad(estimator)
        param_list = [[np.pi / 4 for _ in qc.parameters], [np.pi / 2 for _ in qc.parameters]]
        correct_results = [
            [-0.7266653, -0.4905135, -0.0068606, -0.9228880],
            [-3.5972095, 0.10237173, -0.3117748, 0],
        ]
        op = SparsePauliOp.from_list([("ZI", 1)])
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param]).result().gradients[0]
            np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_parameters(self, grad):
        """Test the estimator gradient for parameters"""
        estimator = StatevectorEstimator()
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc.rx(b, 0)
        gradient = grad(estimator)
        param_list = [[np.pi / 4, np.pi / 2]]
        correct_results = [
            [-0.70710678],
        ]
        op = SparsePauliOp.from_list([("Z", 1)])
        for i, param in enumerate(param_list):
            gradients = gradient.run([qc], [op], [param], parameters=[[a]]).result().gradients[0]
            np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            c = Parameter("c")
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)

            param_list = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            correct_results = [
                [-0.35355339, 0.61237244, -0.61237244],
                [-0.61237244, 0.61237244, -0.35355339],
                [-0.35355339, -0.61237244],
                [-0.61237244, -0.35355339],
            ]
            param = [[a, b, c], [c, b, a], [a, c], [c, a]]
            op = SparsePauliOp.from_list([("Z", 1)])
            for i, p in enumerate(param):  # pylint: disable=invalid-name
                gradient = grad(estimator)
                gradients = (
                    gradient.run([qc], [op], param_list, parameters=[p]).result().gradients[0]
                )
                np.testing.assert_allclose(gradients, correct_results[i], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_multi_arguments(self, grad):
        """Test the estimator gradient for multiple arguments"""
        estimator = StatevectorEstimator()
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        qc2 = QuantumCircuit(1)
        qc2.rx(b, 0)
        gradient = grad(estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        correct_results = [
            [-0.70710678],
            [-1],
        ]
        op = SparsePauliOp.from_list([("Z", 1)])
        gradients = gradient.run([qc, qc2], [op] * 2, param_list).result().gradients
        np.testing.assert_allclose(gradients, correct_results, atol=1e-3)

        c = Parameter("c")
        qc3 = QuantumCircuit(1)
        qc3.rx(c, 0)
        qc3.ry(a, 0)
        param_list2 = [[np.pi / 4], [np.pi / 4, np.pi / 4], [np.pi / 4, np.pi / 4]]
        correct_results2 = [
            [-0.70710678],
            [-0.5],
            [-0.5, -0.5],
        ]
        gradients2 = (
            gradient.run([qc, qc3, qc3], [op] * 3, param_list2, parameters=[[a], [c], None])
            .result()
            .gradients
        )
        np.testing.assert_allclose(gradients2[0], correct_results2[0], atol=1e-3)
        np.testing.assert_allclose(gradients2[1], correct_results2[1], atol=1e-3)
        np.testing.assert_allclose(gradients2[2], correct_results2[2], atol=1e-3)

    @data(*gradient_factories)
    def test_gradient_validation(self, grad):
        """Test estimator gradient's validation"""
        estimator = StatevectorEstimator()
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        gradient = grad(estimator)
        param_list = [[np.pi / 4], [np.pi / 2]]
        op = SparsePauliOp.from_list([("Z", 1)])
        with self.assertRaises(ValueError):
            gradient.run([qc], [op], param_list)
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], [op, op], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc, qc], [op], param_list, parameters=[[a]])
        with self.assertRaises(ValueError):
            gradient.run([qc], [op], [[np.pi / 4, np.pi / 4]])

    def test_spsa_gradient(self):
        """Test the SPSA estimator gradient"""
        estimator = StatevectorEstimator()
        with self.assertRaises(ValueError):
            _ = SPSAEstimatorGradient(estimator, epsilon=-0.1)
        a = Parameter("a")
        b = Parameter("b")
        qc = QuantumCircuit(2)
        qc.rx(b, 0)
        qc.rx(a, 1)
        param_list = [[1, 1]]
        correct_results = [[-0.84147098, 0.84147098]]
        op = SparsePauliOp.from_list([("ZI", 1)])
        gradient = SPSAEstimatorGradient(estimator, epsilon=1e-6, seed=123)
        gradients = gradient.run([qc], [op], param_list).result().gradients
        np.testing.assert_allclose(gradients, correct_results, atol=1e-3)

        # multi parameters
        with self.subTest(msg="Multiple parameters"):
            gradient = SPSAEstimatorGradient(estimator, epsilon=1e-6, seed=123)
            param_list2 = [[1, 1], [1, 1], [3, 3]]
            gradients2 = (
                gradient.run([qc] * 3, [op] * 3, param_list2, parameters=[None, [b], None])
                .result()
                .gradients
            )
            correct_results2 = [[-0.84147098, 0.84147098], [0.84147098], [-0.14112001, 0.14112001]]
            for grad, correct in zip(gradients2, correct_results2):
                np.testing.assert_allclose(grad, correct, atol=1e-3)

        # batch size
        with self.subTest(msg="Batch size"):
            correct_results = [[-0.84147098, 0.1682942]]
            gradient = SPSAEstimatorGradient(estimator, epsilon=1e-6, batch_size=5, seed=123)
            gradients = gradient.run([qc], [op], param_list).result().gradients
            np.testing.assert_allclose(gradients, correct_results, atol=1e-3)

        # parameter order
        with self.subTest(msg="The order of gradients"):
            gradient = SPSAEstimatorGradient(estimator, epsilon=1e-6, seed=123)
            c = Parameter("c")
            qc = QuantumCircuit(1)
            qc.rx(a, 0)
            qc.rz(b, 0)
            qc.rx(c, 0)
            op = SparsePauliOp.from_list([("Z", 1)])
            param_list3 = [[np.pi / 4, np.pi / 2, np.pi / 3]]
            param = [[a, b, c], [c, b, a], [a, c], [c, a]]
            expected = [
                [-0.3535525, 0.3535525, 0.3535525],
                [0.3535525, 0.3535525, -0.3535525],
                [-0.3535525, 0.3535525],
                [0.3535525, -0.3535525],
            ]
            for i, p in enumerate(param):  # pylint: disable=invalid-name
                gradient = SPSAEstimatorGradient(estimator, epsilon=1e-6, seed=123)
                gradients = (
                    gradient.run([qc], [op], param_list3, parameters=[p]).result().gradients[0]
                )
                np.testing.assert_allclose(gradients, expected[i], atol=1e-3)

    @data(ParamShiftEstimatorGradient, LinCombEstimatorGradient)
    def test_gradient_random_parameters(self, grad):
        """Test param shift and lin comb w/ random parameters"""
        rng = np.random.default_rng(123)
        qc = RealAmplitudes(num_qubits=3, reps=1)
        params = qc.parameters
        qc.rx(3.0 * params[0] + params[1].sin(), 0)
        qc.ry(params[0].exp() + 2 * params[1], 1)
        qc.rz(params[0] * params[1] - params[2], 2)
        qc.p(2 * params[0] + 1, 0)
        qc.u(params[0].sin(), params[1] - 2, params[2] * params[3], 1)
        qc.sx(2)
        qc.rxx(params[0].sin(), 1, 2)
        qc.ryy(params[1].cos(), 2, 0)
        qc.rzz(params[2] * 2, 0, 1)
        qc.crx(params[0].exp(), 1, 2)
        qc.cry(params[1].arctan(), 2, 0)
        qc.crz(params[2] * -2, 0, 1)
        qc.dcx(0, 1)
        qc.csdg(0, 1)
        qc.ccx(0, 1, 2)
        qc.iswap(0, 2)
        qc.swap(1, 2)
        qc.global_phase = params[0] * params[1] + params[2].cos().exp()

        size = 10
        op = SparsePauliOp(random_pauli_list(num_qubits=qc.num_qubits, size=size, seed=rng))
        op.coeffs = rng.normal(0, 10, size)

        estimator = StatevectorEstimator()
        findiff = FiniteDiffEstimatorGradient(estimator, 1e-6)
        gradient = grad(estimator)

        num_tries = 10
        param_values = rng.normal(0, 2, (num_tries, qc.num_parameters)).tolist()
        np.testing.assert_allclose(
            findiff.run([qc] * num_tries, [op] * num_tries, param_values).result().gradients,
            gradient.run([qc] * num_tries, [op] * num_tries, param_values).result().gradients,
            rtol=1e-4,
        )

    @data((DerivativeType.IMAG, -1.0), (DerivativeType.COMPLEX, -1.0j))
    @unpack
    def test_complex_gradient(self, derivative_type, expected_gradient_value):
        """Tests if the ``LinCombEstimatorGradient`` has the correct value."""
        estimator = StatevectorEstimator()
        lcu = LinCombEstimatorGradient(estimator, derivative_type=derivative_type)
        reverse = ReverseEstimatorGradient(derivative_type=derivative_type)

        for gradient in [lcu, reverse]:
            with self.subTest(gradient=gradient):
                c = QuantumCircuit(1)
                c.rz(Parameter("p"), 0)
                result = gradient.run([c], [Pauli("I")], [[0.0]]).result()
                self.assertAlmostEqual(result.gradients[0][0], expected_gradient_value)

    @data(
        FiniteDiffEstimatorGradient,
        ParamShiftEstimatorGradient,
        LinCombEstimatorGradient,
        SPSAEstimatorGradient,
    )
    def test_precision(self, grad):
        """Test estimator gradient's precision"""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        op = SparsePauliOp.from_list([("Z", 1)])
        estimator = StatevectorEstimator(default_precision=0.2)
        with self.subTest("estimator"):
            if grad is FiniteDiffEstimatorGradient or grad is SPSAEstimatorGradient:
                gradient = grad(estimator, epsilon=1e-6)
            else:
                gradient = grad(estimator)
            precision = gradient.precision
            result = gradient.run([qc], [op], [[1]]).result()
            self.assertEqual(result.precision, 0.2)
            self.assertEqual(precision, None)

        with self.subTest("gradient init"):
            if grad is FiniteDiffEstimatorGradient or grad is SPSAEstimatorGradient:
                gradient = grad(estimator, epsilon=1e-6, precision=0.3)
            else:
                gradient = grad(estimator, precision=0.3)
            precision = gradient.precision
            result = gradient.run([qc], [op], [[1]]).result()
            self.assertEqual(result.precision, 0.3)
            self.assertEqual(precision, 0.3)

        with self.subTest("gradient update"):
            if grad is FiniteDiffEstimatorGradient or grad is SPSAEstimatorGradient:
                gradient = grad(estimator, epsilon=1e-6, precision=0.4)
            else:
                gradient = grad(estimator, precision=0.4)
            gradient.precision = 0.5
            precision = gradient.precision
            result = gradient.run([qc], [op], [[1]]).result()
            self.assertEqual(result.precision, 0.5)
            self.assertEqual(precision, 0.5)

        with self.subTest("gradient run"):
            if grad is FiniteDiffEstimatorGradient or grad is SPSAEstimatorGradient:
                gradient = grad(estimator, epsilon=1e-6, precision=0.6)
            else:
                gradient = grad(estimator, precision=0.6)
            precision = gradient.precision
            result = gradient.run([qc], [op], [[1]], precision=0.7).result()
            self.assertEqual(result.precision, 0.7)
            # Only default + estimator options. Not run.
            self.assertEqual(precision, 0.6)

    @data(
        FiniteDiffEstimatorGradient,
        ParamShiftEstimatorGradient,
        LinCombEstimatorGradient,
        SPSAEstimatorGradient,
    )
    def test_operations_preserved(self, gradient_cls):
        """Test non-parameterized instructions are preserved and not unrolled."""
        x = Parameter("x")
        circuit = QuantumCircuit(2)
        circuit.initialize([0.5, 0.5, 0.5, 0.5])  # this should remain as initialize
        circuit.crx(x, 0, 1)  # this should get unrolled

        values = [np.pi / 2]
        expect = -1 / (2 * np.sqrt(2))

        observable = SparsePauliOp(["XX"])

        ops = []

        def operations_callback(op):
            ops.append(op)

        estimator = LoggingEstimator(operations_callback=operations_callback)

        if gradient_cls in [SPSAEstimatorGradient, FiniteDiffEstimatorGradient]:
            gradient = gradient_cls(estimator, epsilon=0.01)
        else:
            gradient = gradient_cls(estimator)

        job = gradient.run([circuit], [observable], [values])
        result = job.result()

        with self.subTest(msg="assert initialize is preserved"):
            self.assertTrue(all("initialize" in ops_i[0].keys() for ops_i in ops))

        with self.subTest(msg="assert result is correct"):
            self.assertAlmostEqual(result.gradients[0].item(), expect, places=5)

    def test_product_rule_check(self):
        """Test product rule check."""
        p = Parameter("p")
        qc = QuantumCircuit(1)
        qc.rx(p, 0)
        qc.ry(p, 0)

        from qiskit_algorithms.gradients.reverse.derive_circuit import derive_circuit

        with self.assertRaises(NotImplementedError):
            _ = derive_circuit(qc, p)

    @data(
        FiniteDiffEstimatorGradient,
        ParamShiftEstimatorGradient,
        LinCombEstimatorGradient,
        SPSAEstimatorGradient,
    )
    def test_transpiler(self, gradient_cls):
        """Test that the transpiler is called for the LinCombEstimatorGradient"""
        pass_manager = generate_preset_pass_manager(
            backend=THREE_QUBITS_BACKEND, optimization_level=1, seed_transpiler=42
        )
        counts = [0]

        def callback(**kwargs):
            counts[0] = kwargs["count"]

        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(a, 0)
        op = SparsePauliOp.from_list([("Z", 1)])
        estimator = StatevectorEstimator(default_precision=0.2)

        # Test transpiler without options
        if gradient_cls in [SPSAEstimatorGradient, FiniteDiffEstimatorGradient]:
            gradient = gradient_cls(estimator, epsilon=0.01, transpiler=pass_manager)
        else:
            gradient = gradient_cls(estimator, transpiler=pass_manager)

        gradient.run([qc], [op], [[1]]).result()

        # Test that transpiler is called using callback function
        if gradient_cls in [SPSAEstimatorGradient, FiniteDiffEstimatorGradient]:
            gradient = gradient_cls(
                estimator,
                epsilon=0.01,
                transpiler=pass_manager,
                transpiler_options={"callback": callback},
            )
        else:
            gradient = gradient_cls(
                estimator, transpiler=pass_manager, transpiler_options={"callback": callback}
            )

        gradient.run([qc], [op], [[1]]).result()

        self.assertGreater(counts[0], 0)


if __name__ == "__main__":
    unittest.main()
