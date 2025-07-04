# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Grover's algorithm."""

import unittest
from itertools import product
from test import QiskitAlgorithmsTestCase

import numpy as np
from ddt import data, ddt
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, PhaseOracle
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_algorithms import AmplificationProblem, Grover
from qiskit_algorithms.utils.optionals import CAN_USE_PHASE_ORACLE


@ddt
class TestAmplificationProblem(QiskitAlgorithmsTestCase):
    """Test the amplification problem."""

    def setUp(self):
        super().setUp()
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        self._expected_grover_op = GroverOperator(oracle=oracle)

    @data("oracle_only", "oracle_and_stateprep")
    def test_groverop_getter(self, kind):
        """Test the default construction of the Grover operator."""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)

        if kind == "oracle_only":
            problem = AmplificationProblem(oracle, is_good_state=["11"])
            expected = GroverOperator(oracle)
        else:
            stateprep = QuantumCircuit(2)
            stateprep.ry(0.2, [0, 1])
            problem = AmplificationProblem(
                oracle, state_preparation=stateprep, is_good_state=["11"]
            )
            expected = GroverOperator(oracle, stateprep)

        self.assertEqual(Operator(expected), Operator(problem.grover_operator))

    @data("list_str", "list_int", "statevector", "callable")
    def test_is_good_state(self, kind):
        """Test is_good_state works on different input types."""
        if kind == "list_str":
            is_good_state = ["01", "11"]
        elif kind == "list_int":
            is_good_state = [1]  # means bitstr[1] == '1'
        elif kind == "statevector":
            is_good_state = Statevector(np.array([0, 1, 0, 1]) / np.sqrt(2))
        else:

            def is_good_state(bitstr):
                # same as ``bitstr in ['01', '11']``
                return bitstr[1] == "1"

        possible_states = ["".join(list(map(str, item))) for item in product([0, 1], repeat=2)]

        oracle = QuantumCircuit(2)
        problem = AmplificationProblem(oracle, is_good_state=is_good_state)

        expected = [state in ["01", "11"] for state in possible_states]
        actual = [problem.is_good_state(state) for state in possible_states]

        self.assertListEqual(expected, actual)


@ddt
class TestGrover(QiskitAlgorithmsTestCase):
    """Test for the functionality of Grover"""

    def setUp(self):
        super().setUp()
        self._sampler = StatevectorSampler(seed=123)

    @unittest.skipUnless(
        CAN_USE_PHASE_ORACLE, "tweedledum or qiskit >= 2.0.0 required for this test"
    )
    def test_implicit_phase_oracle_is_good_state(self):
        """Test implicit default for is_good_state with PhaseOracle."""
        grover = self._prepare_grover()
        oracle = PhaseOracle("x & y")
        problem = AmplificationProblem(oracle)
        result = grover.amplify(problem)
        self.assertEqual(result.top_measurement, "11")

    @data([1, 2, 3], None, 2)
    def test_iterations_with_good_state(self, iterations):
        """Test the algorithm with different iteration types and with good state"""
        grover = self._prepare_grover(iterations)
        problem = AmplificationProblem(Statevector.from_label("111"), is_good_state=["111"])
        result = grover.amplify(problem)
        self.assertEqual(result.top_measurement, "111")

    @unittest.skip(
        "Skipped until "
        "https://github.com/qiskit-community/qiskit-algorithms/issues/136#issuecomment-2291169158 is "
        "resolved"
    )
    @data([1, 2, 3], None, 2)
    def test_iterations_with_good_state_sample_from_iterations(self, iterations):
        """Test the algorithm with different iteration types and with good state"""
        grover = self._prepare_grover(iterations, sample_from_iterations=True)
        problem = AmplificationProblem(Statevector.from_label("111"), is_good_state=["111"])
        result = grover.amplify(problem)
        self.assertEqual(result.top_measurement, "111")

    def test_fixed_iterations_without_good_state(self):
        """Test the algorithm with iterations as an int and without good state"""
        grover = self._prepare_grover(iterations=2)
        problem = AmplificationProblem(Statevector.from_label("111"))
        result = grover.amplify(problem)
        self.assertEqual(result.top_measurement, "111")

    @data([1, 2, 3], None)
    def test_iterations_without_good_state(self, iterations):
        """Test the correct error is thrown for none/list of iterations and without good state"""
        grover = self._prepare_grover(iterations=iterations)
        problem = AmplificationProblem(Statevector.from_label("111"))

        with self.assertRaisesRegex(
            TypeError, "An is_good_state function is required with the provided oracle"
        ):
            grover.amplify(problem)

    def test_iterator(self):
        """Test running the algorithm on an iterator."""

        # step-function iterator
        def iterator():
            wait, value, count = 3, 1, 0
            while True:
                yield value
                count += 1
                if count % wait == 0:
                    value += 1

        grover = self._prepare_grover(iterations=iterator())
        problem = AmplificationProblem(Statevector.from_label("111"), is_good_state=["111"])
        result = grover.amplify(problem)
        self.assertEqual(result.top_measurement, "111")

    def test_growth_rate(self):
        """Test running the algorithm on a growth rate"""
        grover = self._prepare_grover(growth_rate=8 / 7)
        problem = AmplificationProblem(Statevector.from_label("111"), is_good_state=["111"])
        result = grover.amplify(problem)
        self.assertEqual(result.top_measurement, "111")

    def test_max_num_iterations(self):
        """Test the iteration stops when the maximum number of iterations is reached."""

        def zero():
            while True:
                yield 0

        grover = self._prepare_grover(iterations=zero())
        n = 5
        problem = AmplificationProblem(Statevector.from_label("1" * n), is_good_state=["1" * n])
        result = grover.amplify(problem)
        self.assertEqual(len(result.iterations), 2**n)

    def test_max_power(self):
        """Test the iteration stops when the maximum power is reached."""
        lam = 10.0
        grover = self._prepare_grover(growth_rate=lam)
        problem = AmplificationProblem(Statevector.from_label("111"), is_good_state=["111"])
        result = grover.amplify(problem)
        self.assertEqual(len(result.iterations), 0)

    def test_run_circuit_oracle(self):
        """Test execution with a quantum circuit oracle"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        problem = AmplificationProblem(oracle, is_good_state=["11"])
        grover = self._prepare_grover()
        result = grover.amplify(problem)
        self.assertIn(result.top_measurement, ["11"])

    def test_run_state_vector_oracle(self):
        """Test execution with a state vector oracle"""
        mark_state = Statevector.from_label("11")
        problem = AmplificationProblem(mark_state, is_good_state=["11"])
        grover = self._prepare_grover()
        result = grover.amplify(problem)
        self.assertIn(result.top_measurement, ["11"])

    def test_run_custom_grover_operator(self):
        """Test execution with a grover operator oracle"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover_op = GroverOperator(oracle)
        problem = AmplificationProblem(
            oracle=oracle, grover_operator=grover_op, is_good_state=["11"]
        )
        grover = self._prepare_grover()
        result = grover.amplify(problem)
        self.assertIn(result.top_measurement, ["11"])

    def test_optimal_num_iterations(self):
        """Test optimal_num_iterations"""
        num_qubits = 7
        for num_solutions in range(1, 2**num_qubits):
            amplitude = np.sqrt(num_solutions / 2**num_qubits)
            expected = round(np.arccos(amplitude) / (2 * np.arcsin(amplitude)))
            actual = Grover.optimal_num_iterations(num_solutions, num_qubits)
            self.assertEqual(actual, expected)

    def test_construct_circuit(self):
        """Test construct_circuit"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        problem = AmplificationProblem(oracle, is_good_state=["11"])
        grover = Grover()
        constructed = grover.construct_circuit(problem, 2, measurement=False)

        grover_op = GroverOperator(oracle)
        expected = QuantumCircuit(2)
        expected.h([0, 1])
        expected.compose(grover_op.power(2), inplace=True)

        self.assertTrue(Operator(constructed).equiv(Operator(expected)))

    def test_circuit_result(self):
        """Test circuit_result"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        # is_good_state=['00'] is intentionally selected to obtain a list of results
        problem = AmplificationProblem(oracle, is_good_state=["00"])
        grover = self._prepare_grover(iterations=[1, 2, 3, 4])

        result = grover.amplify(problem)

        for i, dist in enumerate(result.circuit_results):
            keys, values = zip(*sorted(dist.items()))
            if i in (0, 3):
                self.assertTupleEqual(keys, ("11",))
                np.testing.assert_allclose(values, [1], atol=0.2)
            else:
                self.assertTupleEqual(keys, ("00", "01", "10", "11"))
                np.testing.assert_allclose(values, [0.25, 0.25, 0.25, 0.25], atol=0.2)

    def test_max_probability(self):
        """Test max_probability"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        problem = AmplificationProblem(oracle, is_good_state=["11"])
        grover = self._prepare_grover()
        result = grover.amplify(problem)
        self.assertAlmostEqual(result.max_probability, 1.0)

    @unittest.skipUnless(
        CAN_USE_PHASE_ORACLE, "tweedledum or qiskit >= 2.0.0 required for this test"
    )
    def test_oracle_evaluation(self):
        """Test oracle_evaluation for PhaseOracle"""
        oracle = PhaseOracle("x1 & x2 & (not x3)")
        problem = AmplificationProblem(oracle, is_good_state=oracle.evaluate_bitstring)
        grover = self._prepare_grover()
        result = grover.amplify(problem)
        self.assertTrue(result.oracle_evaluation)
        self.assertEqual("011", result.top_measurement)

    def test_sampler_setter(self):
        """Test sampler setter"""
        grover = Grover()
        grover.sampler = self._sampler
        self.assertEqual(grover.sampler, self._sampler)

    def test_transpiler(self):
        """Test that the transpiler is called"""
        pass_manager = generate_preset_pass_manager(optimization_level=1, seed_transpiler=42)
        counts = [0]

        def callback(**kwargs):
            counts[0] = kwargs["count"]

        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        # is_good_state=['00'] is intentionally selected to obtain a list of results
        problem = AmplificationProblem(oracle)

        # Test transpilation without setting options
        Grover(
            iterations=1,
            sampler=StatevectorSampler(seed=42),
            transpiler=pass_manager,
        ).amplify(problem)

        # Test that transpiler is called using callback function
        Grover(
            iterations=1,
            sampler=StatevectorSampler(seed=42),
            transpiler=pass_manager,
            transpiler_options={"callback": callback},
        ).amplify(problem)

        self.assertGreater(counts[0], 0)

    def _prepare_grover(
        self,
        iterations=None,
        growth_rate=None,
        sample_from_iterations=False,
    ):
        """Prepare Grover instance for test"""
        return Grover(
            sampler=self._sampler,
            iterations=iterations,
            growth_rate=growth_rate,
            sample_from_iterations=sample_from_iterations,
        )


if __name__ == "__main__":
    unittest.main()
