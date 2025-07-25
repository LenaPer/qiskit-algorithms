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

"""Test Variational Quantum Real Time Evolution algorithm."""

import unittest
from test import QiskitAlgorithmsTestCase

from ddt import ddt
import numpy as np

from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector
from qiskit.providers.fake_provider import GenericBackendV2

from qiskit_algorithms.gradients import LinCombQGT, DerivativeType, LinCombEstimatorGradient
from qiskit_algorithms import TimeEvolutionProblem, VarQRTE
from qiskit_algorithms.time_evolvers.variational import RealMcLachlanPrinciple
from qiskit_algorithms.utils import algorithm_globals


@ddt
class TestVarQRTE(QiskitAlgorithmsTestCase):
    """Test Variational Quantum Real Time Evolution algorithm."""

    def setUp(self):
        super().setUp()
        self.seed = 11
        np.random.seed(self.seed)

    def test_time_dependent_hamiltonian(self):
        """Simple test case with a time dependent Hamiltonian."""
        t_param = Parameter("t")
        hamiltonian = SparsePauliOp(["Z"], np.array(t_param))

        x = ParameterVector("x", 3)
        circuit = QuantumCircuit(1)
        circuit.rz(x[0], 0)
        circuit.ry(x[1], 0)
        circuit.rz(x[2], 0)

        initial_parameters = np.array([0, np.pi / 2, 0])

        def expected_state(time):
            # possible with pen and paper as the Hamiltonian is diagonal
            return 1 / np.sqrt(2) * np.array([np.exp(-0.5j * time**2), np.exp(0.5j * time**2)])

        final_time = 0.75
        evolution_problem = TimeEvolutionProblem(hamiltonian, t_param=t_param, time=final_time)
        estimator = StatevectorEstimator()
        varqrte = VarQRTE(circuit, initial_parameters, estimator=estimator)

        result = varqrte.evolve(evolution_problem)

        final_parameters = result.parameter_values[-1]
        final_state = Statevector(circuit.assign_parameters(final_parameters)).to_dict()
        final_expected_state = expected_state(final_time)

        for key, expected_value in final_state.items():
            self.assertTrue(np.allclose(final_expected_state[int(key)], expected_value, 1e-02))

    def test_run_d_1_with_aux_ops(self):
        """Test VarQRTE for d = 1 and t = 0.1 with evaluating auxiliary operators and the Forward
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

        time = 0.1

        evolution_problem = TimeEvolutionProblem(observable, time, aux_operators=aux_ops)

        thetas_expected = [
            0.886841151529636,
            1.53852629218265,
            1.57099556659882,
            1.5889216657174,
            1.5996487153364,
            1.57018939515742,
            1.63950719260698,
            1.53853696496673,
        ]

        expected_aux_ops = [0.06836996703935797, 0.7711574493422457]

        with self.subTest(msg="Test exact backend."):
            algorithm_globals.random_seed = self.seed
            estimator = StatevectorEstimator(seed=self.seed)
            qgt = LinCombQGT(estimator)
            gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.IMAG)
            var_principle = RealMcLachlanPrinciple(qgt, gradient)

            var_qrte = VarQRTE(
                ansatz, init_param_values, var_principle, estimator, num_timesteps=25
            )
            evolution_result = var_qrte.evolve(evolution_problem)

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
            # A precision of pow(2, -7) roughly corresponds to 4 * 4096 shots
            estimator = StatevectorEstimator(seed=self.seed, default_precision=pow(2, -7))
            qgt = LinCombQGT(estimator)
            gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.IMAG)
            var_principle = RealMcLachlanPrinciple(qgt, gradient)

            var_qrte = VarQRTE(
                ansatz, init_param_values, var_principle, estimator, num_timesteps=25
            )
            evolution_result = var_qrte.evolve(evolution_problem)

            aux_ops = evolution_result.aux_ops_evaluated

            parameter_values = evolution_result.parameter_values[-1]

            for i, parameter_value in enumerate(parameter_values):
                np.testing.assert_almost_equal(
                    float(parameter_value), thetas_expected[i], decimal=2
                )

            np.testing.assert_array_almost_equal(
                [result[0] for result in aux_ops], expected_aux_ops, decimal=2
            )

    def test_run_d_2(self):
        """Test VarQRTE for d = 2 and t = 1 with RK45 ODE solver."""

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
        estimator = StatevectorEstimator()
        qgt = LinCombQGT(estimator)
        gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.IMAG)

        var_principle = RealMcLachlanPrinciple(qgt, gradient)

        param_dict = dict(zip(parameters, init_param_values))

        time = 1
        var_qrte = VarQRTE(ansatz, param_dict, var_principle, ode_solver="RK45", num_timesteps=25)

        thetas_expected = [
            0.348407744196573,
            0.919404626262464,
            1.18189219371626,
            0.771011177789998,
            0.734384256533924,
            0.965289520781899,
            1.14441687204195,
            1.17231927568571,
            1.03014771379412,
            0.867266309056347,
            0.699606368428206,
            0.610788576398685,
        ]

        self._test_helper(observable, thetas_expected, time, var_qrte)

    def test_run_d_1_time_dependent(self):
        """Test VarQRTE for d = 1 and a time-dependent Hamiltonian with the Forward Euler solver."""
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

        thetas_expected = [0.0, 1.5707963267949, 0.99]

        # the expected final state is Statevector([0.62289306-0.33467034j, 0.62289306+0.33467034j])

        algorithm_globals.random_seed = self.seed
        estimator = StatevectorEstimator()
        qgt = LinCombQGT(estimator)
        gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.IMAG)
        var_principle = RealMcLachlanPrinciple(qgt, gradient)

        var_qrte = VarQRTE(ansatz, init_param_values, var_principle, estimator, num_timesteps=100)
        evolution_result = var_qrte.evolve(evolution_problem)

        parameter_values = evolution_result.parameter_values[-1]

        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=2)

    def _test_helper(self, observable, thetas_expected, time, var_qrte):
        evolution_problem = TimeEvolutionProblem(observable, time)
        evolution_result = var_qrte.evolve(evolution_problem)

        parameter_values = evolution_result.parameter_values[-1]

        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=4)

    def test_transpiler(self):
        """Test VarQRTE for d = 1 and t = 1 with evaluating auxiliary operator and the Forward
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
        gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.IMAG)
        var_principle = RealMcLachlanPrinciple(qgt, gradient)

        pass_manager = generate_preset_pass_manager(
            backend=GenericBackendV2(num_qubits=3, coupling_map=[[0, 1], [1, 2]], seed=54),
            optimization_level=1,
            seed_transpiler=42,
        )
        counts = [0]

        def callback(**kwargs):
            counts[0] = kwargs["count"]

        var_qrte = VarQRTE(
            ansatz,
            init_param_values,
            var_principle,
            estimator,
            num_timesteps=25,
            transpiler=pass_manager,
            transpiler_options={"callback": callback},
        )
        var_qrte.evolve(evolution_problem)

        self.assertGreater(counts[0], 0)


if __name__ == "__main__":
    unittest.main()
