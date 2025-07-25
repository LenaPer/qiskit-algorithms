# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Variational Quantum Imaginary Time Evolution algorithm."""

import unittest
from test import QiskitAlgorithmsTestCase
from ddt import ddt
import numpy as np

from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector
from qiskit.providers.fake_provider import GenericBackendV2

from qiskit_algorithms.gradients import LinCombQGT, LinCombEstimatorGradient
from qiskit_algorithms import TimeEvolutionProblem, VarQITE
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit_algorithms.utils import algorithm_globals


@ddt
class TestVarQITE(QiskitAlgorithmsTestCase):
    """Test Variational Quantum Imaginary Time Evolution algorithm."""

    def setUp(self):
        super().setUp()
        self.seed = 11
        np.random.seed(self.seed)

    def test_run_d_1_with_aux_ops(self):
        """Test VarQITE for d = 1 and t = 1 with evaluating auxiliary operator and the Forward
        Euler solver."""

        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )
        aux_ops = [Pauli("XX"), Pauli("YZ")]
        d = 1  # pylint: disable=invalid-name
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1
        time = 1

        evolution_problem = TimeEvolutionProblem(observable, time, aux_operators=aux_ops)

        thetas_expected = [
            0.87984606025879,
            2.04681975664763,
            2.68980594039104,
            2.75915988512186,
            2.38796546567171,
            1.78144857115127,
            2.13109162826101,
            1.9259609596416,
        ]

        thetas_expected_shots = [
            0.87665726,
            2.04313234,
            2.67702257,
            2.74971934,
            2.38728532,
            1.78404205,
            2.11388396,
            1.92959433,
        ]

        # Should be roughly the same in both Exact and shot-based backends
        expected_aux_ops = (-0.2177982985749799, 0.2556790598588627)

        with self.subTest(msg="Test exact backend"):
            algorithm_globals.random_seed = self.seed

            estimator = StatevectorEstimator(seed=self.seed)
            qgt = LinCombQGT(estimator)
            gradient = LinCombEstimatorGradient(estimator)
            var_principle = ImaginaryMcLachlanPrinciple(qgt, gradient)

            var_qite = VarQITE(
                ansatz, init_param_values, var_principle, estimator, num_timesteps=25
            )
            evolution_result = var_qite.evolve(evolution_problem)

            aux_ops = evolution_result.aux_ops_evaluated

            parameter_values = evolution_result.parameter_values[-1]

            for i, parameter_value in enumerate(parameter_values):
                np.testing.assert_almost_equal(
                    float(parameter_value), thetas_expected[i], decimal=2
                )

            np.testing.assert_array_almost_equal(
                [result[0] for result in aux_ops], expected_aux_ops
            )

        with self.subTest(msg="Test non-zero precision backend."):
            algorithm_globals.random_seed = self.seed

            # A precision of pow(2, -6) roughly corresponds to 4096 shots
            estimator = StatevectorEstimator(default_precision=pow(2, -6), seed=self.seed)
            qgt = LinCombQGT(estimator)
            gradient = LinCombEstimatorGradient(estimator)
            var_principle = ImaginaryMcLachlanPrinciple(qgt, gradient)

            var_qite = VarQITE(
                ansatz, init_param_values, var_principle, estimator, num_timesteps=25
            )
            evolution_result = var_qite.evolve(evolution_problem)

            aux_ops = evolution_result.aux_ops_evaluated

            parameter_values = evolution_result.parameter_values[-1]

            for i, parameter_value in enumerate(parameter_values):
                np.testing.assert_almost_equal(
                    float(parameter_value), thetas_expected_shots[i], decimal=2
                )

            np.testing.assert_array_almost_equal(
                [result[0] for result in aux_ops], expected_aux_ops, decimal=1
            )

    def test_run_d_1_t_7(self):
        """Test VarQITE for d = 1 and t = 7 with RK45 ODE solver."""

        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )

        d = 1  # pylint: disable=invalid-name
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1
        var_principle = ImaginaryMcLachlanPrinciple()

        time = 7
        var_qite = VarQITE(
            ansatz, init_param_values, var_principle, ode_solver="RK45", num_timesteps=25
        )

        thetas_expected = [
            0.828917365718767,
            1.88481074798033,
            3.14111335991238,
            3.14125849601269,
            2.33768562678401,
            1.78670990729437,
            2.04214275514208,
            2.04009918594422,
        ]

        self._test_helper(observable, thetas_expected, time, var_qite, 2)

    def test_run_d_2(self):
        """Test VarQITE for d = 2 and t = 1 with RK45 ODE solver."""

        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )
        d = 2  # pylint: disable=invalid-name
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 4

        var_principle = ImaginaryMcLachlanPrinciple()

        time = 1
        var_qite = VarQITE(
            ansatz, init_param_values, var_principle, ode_solver="RK45", num_timesteps=25
        )

        thetas_expected = [
            1.29495364023786,
            1.08970061333559,
            0.667488228710748,
            0.500122687902944,
            1.4377736672043,
            1.22881086103085,
            0.729773048146251,
            1.01698854755226,
            0.050807780587492,
            0.294828474947149,
            0.839305697704923,
            0.663689581255428,
        ]

        self._test_helper(observable, thetas_expected, time, var_qite, 4)

    def test_run_d_1_time_dependent(self):
        """Test VarQITE for d = 1 and a time-dependent Hamiltonian with the Forward Euler solver."""
        t_param = Parameter("t")
        time = 1
        observable = SparsePauliOp(["I", "Z"], np.array([0, t_param]))

        x, y, z = [Parameter(s) for s in "xyz"]
        ansatz = QuantumCircuit(1)
        ansatz.rz(x, 0)
        ansatz.ry(y, 0)
        ansatz.rz(z, 0)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        x_val = 0
        y_val = np.pi / 2
        z_val = 0

        init_param_values[0] = x_val
        init_param_values[1] = y_val
        init_param_values[2] = z_val

        evolution_problem = TimeEvolutionProblem(observable, time, t_param=t_param)

        thetas_expected = [1.83881002737137e-18, 2.43224994794434, -3.05311331771918e-18]

        state_expected = Statevector([0.34849948 + 0.0j, 0.93730897 + 0.0j]).to_dict()
        # the expected final state is Statevector([0.34849948+0.j, 0.93730897+0.j])

        with self.subTest(msg="Test exact backend."):
            algorithm_globals.random_seed = self.seed

            estimator = StatevectorEstimator(seed=self.seed)
            var_principle = ImaginaryMcLachlanPrinciple()

            var_qite = VarQITE(
                ansatz, init_param_values, var_principle, estimator, num_timesteps=100
            )

            evolution_result = var_qite.evolve(evolution_problem)

            evolved_state = evolution_result.evolved_state

            parameter_values = evolution_result.parameter_values[-1]

            for key, evolved_value in Statevector(evolved_state).to_dict().items():
                # np.allclose works with complex numbers
                self.assertTrue(np.allclose(evolved_value, state_expected[key], 1e-02))

            for i, parameter_value in enumerate(parameter_values):
                np.testing.assert_almost_equal(
                    float(parameter_value), thetas_expected[i], decimal=2
                )

        with self.subTest(msg="Test non-zero precision backend."):
            algorithm_globals.random_seed = self.seed

            # A precision of pow(2, -6) roughly corresponds to 4096 shots
            estimator = StatevectorEstimator(default_precision=pow(2, -6), seed=self.seed)
            var_principle = ImaginaryMcLachlanPrinciple()

            var_qite = VarQITE(
                ansatz, init_param_values, var_principle, estimator, num_timesteps=100
            )

            evolution_result = var_qite.evolve(evolution_problem)

            evolved_state = evolution_result.evolved_state

            parameter_values = evolution_result.parameter_values[-1]

            for key, evolved_value in Statevector(evolved_state).to_dict().items():
                # np.allclose works with complex numbers
                self.assertTrue(np.allclose(evolved_value, state_expected[key], 1e-02))

            for i, parameter_value in enumerate(parameter_values):
                np.testing.assert_almost_equal(
                    float(parameter_value), thetas_expected[i], decimal=2
                )

    # pylint: disable=too-many-positional-arguments
    def _test_helper(self, observable, thetas_expected, time, var_qite, decimal):
        evolution_problem = TimeEvolutionProblem(observable, time)
        evolution_result = var_qite.evolve(evolution_problem)
        parameter_values = evolution_result.parameter_values[-1]

        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(
                float(parameter_value), thetas_expected[i], decimal=decimal
            )

    def test_transpiler(self):
        """Test VarQITE for d = 1 and t = 1 with evaluating auxiliary operator and the Forward
        Euler solver."""

        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )
        aux_ops = [Pauli("XX"), Pauli("YZ")]
        d = 1  # pylint: disable=invalid-name
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1
        time = 1

        evolution_problem = TimeEvolutionProblem(observable, time, aux_operators=aux_ops)

        estimator = StatevectorEstimator(seed=self.seed)
        qgt = LinCombQGT(estimator)
        gradient = LinCombEstimatorGradient(estimator)
        var_principle = ImaginaryMcLachlanPrinciple(qgt, gradient)

        pass_manager = generate_preset_pass_manager(
            backend=GenericBackendV2(num_qubits=3, coupling_map=[[0, 1], [1, 2]], seed=54),
            optimization_level=1,
            seed_transpiler=42,
        )
        counts = [0]

        def callback(**kwargs):
            counts[0] = kwargs["count"]

        var_qite = VarQITE(
            ansatz,
            init_param_values,
            var_principle,
            estimator,
            num_timesteps=25,
            transpiler=pass_manager,
            transpiler_options={"callback": callback},
        )
        var_qite.evolve(evolution_problem)

        self.assertGreater(counts[0], 0)


if __name__ == "__main__":
    unittest.main()
