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

"""Test the variational quantum eigensolver algorithm."""

import unittest
from test import QiskitAlgorithmsTestCase

from functools import partial
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from ddt import data, ddt

from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit.library import RealAmplitudes, TwoLocal
from qiskit.quantum_info import SparsePauliOp, Operator, Pauli
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from qiskit_algorithms import AlgorithmError
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import (
    CG,
    COBYLA,
    GradientDescent,
    L_BFGS_B,
    OptimizerResult,
    P_BFGS,
    QNSPSA,
    SLSQP,
    SPSA,
    TNC,
)
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_algorithms.utils import algorithm_globals


THREE_QUBITS_BACKEND = GenericBackendV2(num_qubits=3, coupling_map=[[0, 1], [1, 2]], seed=54)


# pylint: disable=invalid-name
def _mock_optimizer(fun, x0, jac=None, bounds=None, inputs=None) -> OptimizerResult:
    """A mock of a callable that can be used as minimizer in the VQE."""
    result = OptimizerResult()
    result.x = np.zeros_like(x0)
    result.fun = fun(result.x)
    result.nit = 0

    if inputs is not None:
        inputs.update({"fun": fun, "x0": x0, "jac": jac, "bounds": bounds})
    return result


@ddt
class TestVQE(QiskitAlgorithmsTestCase):
    """Test VQE"""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        self.h2_op = SparsePauliOp(
            ["II", "IZ", "ZI", "ZZ", "XX"],
            coeffs=[
                -1.052373245772859,
                0.39793742484318045,
                -0.39793742484318045,
                -0.01128010425623538,
                0.18093119978423156,
            ],
        )
        self.h2_energy = -1.85727503

        self.ryrz_wavefunction = TwoLocal(rotation_blocks=["ry", "rz"], entanglement_blocks="cz")
        self.ry_wavefunction = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

    @data(L_BFGS_B(), COBYLA())
    def test_using_ref_estimator(self, optimizer):
        """Test VQE using reference Estimator."""
        vqe = VQE(StatevectorEstimator(seed=42), self.ryrz_wavefunction, optimizer)

        result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        with self.subTest(msg="test eigenvalue"):
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)

        with self.subTest(msg="test optimal_value"):
            self.assertAlmostEqual(result.optimal_value, self.h2_energy)

        with self.subTest(msg="test dimension of optimal point"):
            self.assertEqual(len(result.optimal_point), 16)

        with self.subTest(msg="assert cost_function_evals is set"):
            self.assertIsNotNone(result.cost_function_evals)

        with self.subTest(msg="assert optimizer_time is set"):
            self.assertIsNotNone(result.optimizer_time)

        with self.subTest(msg="assert optimizer_result is set"):
            self.assertIsNotNone(result.optimizer_result)

        with self.subTest(msg="assert optimizer_result."):
            self.assertAlmostEqual(result.optimizer_result.fun, self.h2_energy, places=5)

        with self.subTest(msg="assert return ansatz is set"):
            estimator = StatevectorEstimator(seed=42)
            job = estimator.run([(result.optimal_circuit, self.h2_op, result.optimal_point)])
            np.testing.assert_array_almost_equal(job.result()[0].data.evs, result.eigenvalue, 6)

    def test_invalid_initial_point(self):
        """Test the proper error is raised when the initial point has the wrong size."""
        ansatz = self.ryrz_wavefunction
        initial_point = np.array([1])

        vqe = VQE(
            StatevectorEstimator(seed=42),
            ansatz,
            SLSQP(),
            initial_point=initial_point,
        )

        with self.assertRaises(ValueError):
            _ = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

    def test_ansatz_resize(self):
        """Test the ansatz is properly resized if it's a blueprint circuit."""
        ansatz = RealAmplitudes(1, reps=1)
        vqe = VQE(StatevectorEstimator(seed=42), ansatz, SLSQP())
        result = vqe.compute_minimum_eigenvalue(self.h2_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)

    def test_invalid_ansatz_size(self):
        """Test an error is raised if the ansatz has the wrong number of qubits."""
        ansatz = QuantumCircuit(1)
        ansatz.compose(RealAmplitudes(1, reps=2))
        vqe = VQE(StatevectorEstimator(), ansatz, SLSQP())

        with self.assertRaises(AlgorithmError):
            _ = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

    def test_missing_ansatz_params(self):
        """Test specifying an ansatz with no parameters raises an error."""
        ansatz = QuantumCircuit(self.h2_op.num_qubits)
        vqe = VQE(StatevectorEstimator(), ansatz, SLSQP())
        with self.assertRaises(AlgorithmError):
            vqe.compute_minimum_eigenvalue(operator=self.h2_op)

    def test_max_evals_grouped(self):
        """Test with SLSQP with max_evals_grouped."""
        optimizer = SLSQP(maxiter=50, max_evals_grouped=5)
        vqe = VQE(
            StatevectorEstimator(seed=42),
            self.ryrz_wavefunction,
            optimizer,
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)

    @data(
        CG(),
        L_BFGS_B(),
        P_BFGS(),
        SLSQP(),
        TNC(),
    )
    def test_with_gradient(self, optimizer):
        """Test VQE using gradient primitive."""
        estimator = StatevectorEstimator(seed=42)
        vqe = VQE(
            estimator,
            self.ry_wavefunction,
            optimizer,
            gradient=ParamShiftEstimatorGradient(estimator),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)

    def test_gradient_passed(self):
        """Test the gradient is properly passed into the optimizer."""
        inputs = {}
        estimator = StatevectorEstimator(seed=42)
        vqe = VQE(
            estimator,
            RealAmplitudes(),
            partial(_mock_optimizer, inputs=inputs),
            gradient=ParamShiftEstimatorGradient(estimator),
        )
        _ = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertIsNotNone(inputs["jac"])

    def test_gradient_run(self):
        """Test using the gradient to calculate the minimum."""
        estimator = StatevectorEstimator(seed=42)
        vqe = VQE(
            estimator,
            RealAmplitudes(),
            GradientDescent(maxiter=200, learning_rate=0.1),
            gradient=ParamShiftEstimatorGradient(estimator),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)

    def test_callback(self):
        """Test the callback on VQE."""
        history = {"eval_count": [], "parameters": [], "mean": [], "metadata": []}

        def store_intermediate_result(eval_count, parameters, mean, metadata):
            history["eval_count"].append(eval_count)
            history["parameters"].append(parameters)
            history["mean"].append(mean)
            history["metadata"].append(metadata)

        optimizer = COBYLA(maxiter=3)
        wavefunction = self.ry_wavefunction

        estimator = StatevectorEstimator(seed=42)

        vqe = VQE(
            estimator,
            wavefunction,
            optimizer,
            callback=store_intermediate_result,
        )
        vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertTrue(all(isinstance(count, int) for count in history["eval_count"]))
        self.assertTrue(all(isinstance(mean, float) for mean in history["mean"]))
        self.assertTrue(all(isinstance(metadata, dict) for metadata in history["metadata"]))
        for params in history["parameters"]:
            self.assertTrue(all(isinstance(param, float) for param in params))

    def test_reuse(self):
        """Test re-using a VQE algorithm instance."""
        ansatz = TwoLocal(rotation_blocks=["ry", "rz"], entanglement_blocks="cz")
        vqe = VQE(StatevectorEstimator(seed=42), ansatz, SLSQP(maxiter=300))
        with self.subTest(msg="assert VQE works once all info is available"):
            result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)

        operator = Operator(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]]))
        operator = SparsePauliOp.from_operator(operator)

        with self.subTest(msg="assert vqe works on re-use."):
            result = vqe.compute_minimum_eigenvalue(operator=operator)
            self.assertAlmostEqual(result.eigenvalue.real, -1.0, places=5)

    def test_vqe_optimizer_reuse(self):
        """Test running same VQE twice to re-use optimizer, then switch optimizer"""
        vqe = VQE(
            StatevectorEstimator(seed=42),
            self.ryrz_wavefunction,
            SLSQP(),
        )

        def run_check():
            result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)

        run_check()

        with self.subTest("Optimizer re-use."):
            run_check()

        with self.subTest("Optimizer replace."):
            vqe.optimizer = L_BFGS_B()
            run_check()

    def test_default_batch_evaluation_on_spsa(self):
        """Test the default batching works."""
        ansatz = TwoLocal(2, rotation_blocks=["ry", "rz"], entanglement_blocks="cz")

        wrapped_estimator = StatevectorEstimator(seed=42)
        inner_estimator = StatevectorEstimator(seed=43)

        callcount = {"estimator": 0}

        def wrapped_estimator_run(*args, **kwargs):
            kwargs["callcount"]["estimator"] += 1
            return inner_estimator.run(*args)

        wrapped_estimator.run = partial(wrapped_estimator_run, callcount=callcount)

        spsa = SPSA(maxiter=5)

        vqe = VQE(wrapped_estimator, ansatz, spsa)
        _ = vqe.compute_minimum_eigenvalue(Pauli("ZZ"))

        # 1 calibration + 5 loss + 1 return loss
        expected_estimator_runs = 1 + 5 + 1

        with self.subTest(msg="check callcount"):
            self.assertEqual(callcount["estimator"], expected_estimator_runs)

        with self.subTest(msg="check reset to original max evals grouped"):
            self.assertIsNone(spsa._max_evals_grouped)

    def test_batch_evaluate_with_qnspsa(self):
        """Test batch evaluating with QNSPSA works."""
        ansatz = TwoLocal(2, rotation_blocks=["ry", "rz"], entanglement_blocks="cz")

        wrapped_sampler = StatevectorSampler(seed=42)
        inner_sampler = StatevectorSampler(seed=43)

        wrapped_estimator = StatevectorEstimator(seed=44)
        inner_estimator = StatevectorEstimator(seed=45)

        callcount = {"sampler": 0, "estimator": 0}

        def wrapped_estimator_run(*args):
            callcount["estimator"] += 1
            return inner_estimator.run(*args)

        def wrapped_sampler_run(*args):
            callcount["sampler"] += 1
            return inner_sampler.run(*args)

        wrapped_estimator.run = wrapped_estimator_run
        wrapped_sampler.run = wrapped_sampler_run

        fidelity = ComputeUncompute(wrapped_sampler)

        def fidelity_callable(left, right):
            batchsize = np.asarray(left).shape[0]
            job = fidelity.run(batchsize * [ansatz], batchsize * [ansatz], left, right)
            return job.result().fidelities

        qnspsa = QNSPSA(fidelity_callable, maxiter=5)
        qnspsa.set_max_evals_grouped(100)

        vqe = VQE(
            wrapped_estimator,
            ansatz,
            qnspsa,
        )
        _ = vqe.compute_minimum_eigenvalue(Pauli("ZZ"))

        # 5 (fidelity)
        expected_sampler_runs = 5
        # 1 calibration + 1 stddev estimation + 1 initial blocking
        # + 5 (1 loss + 1 blocking) + 1 return loss
        expected_estimator_runs = 1 + 1 + 1 + 5 * 2 + 1

        self.assertEqual(callcount["sampler"], expected_sampler_runs)
        self.assertEqual(callcount["estimator"], expected_estimator_runs)

    def test_optimizer_scipy_callable(self):
        """Test passing a SciPy optimizer directly as callable."""
        vqe = VQE(
            StatevectorEstimator(seed=42),
            self.ryrz_wavefunction,
            partial(scipy_minimize, method="L-BFGS-B", options={"maxiter": 10}),
        )
        result = vqe.compute_minimum_eigenvalue(self.h2_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=2)

    def test_optimizer_callable(self):
        """Test passing a optimizer directly as callable."""
        ansatz = RealAmplitudes(1, reps=1)
        vqe = VQE(StatevectorEstimator(seed=42), ansatz, _mock_optimizer)
        result = vqe.compute_minimum_eigenvalue(SparsePauliOp("Z"))
        self.assertTrue(np.all(result.optimal_point == np.zeros(ansatz.num_parameters)))

    # Since we perform actions on the aux_operators when a transpiler is set, we have to check that it
    # doesn't affect the final result
    @data(
        None,
        generate_preset_pass_manager(
            backend=THREE_QUBITS_BACKEND, optimization_level=1, seed_transpiler=42
        ),
    )
    def test_aux_operators_list(self, transpiler):
        """Test list-based aux_operators."""
        wavefunction = self.ry_wavefunction
        wavefunction.num_qubits = 2
        vqe = VQE(
            StatevectorEstimator(seed=42), wavefunction, SLSQP(maxiter=300), transpiler=transpiler
        )

        with self.subTest("Test with an empty list."):
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=[])
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)
            self.assertIsInstance(result.aux_operators_evaluated, list)
            self.assertEqual(len(result.aux_operators_evaluated), 0)

        with self.subTest("Test with two auxiliary operators."):
            aux_op1 = SparsePauliOp.from_list([("II", 2.0)])
            aux_op2 = SparsePauliOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
            aux_ops = [aux_op1, aux_op2]
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=aux_ops)

            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)
            self.assertEqual(len(result.aux_operators_evaluated), 2)
            # expectation values
            self.assertAlmostEqual(result.aux_operators_evaluated[0][0], 2.0, places=6)
            self.assertAlmostEqual(result.aux_operators_evaluated[1][0], 0.0, places=6)
            # metadata
            self.assertIsInstance(result.aux_operators_evaluated[0][1], dict)
            self.assertIsInstance(result.aux_operators_evaluated[1][1], dict)

        with self.subTest("Test with additional zero operator."):
            extra_ops = [*aux_ops, 0]
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=extra_ops)
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)
            self.assertEqual(len(result.aux_operators_evaluated), 3)
            # expectation values
            self.assertAlmostEqual(result.aux_operators_evaluated[0][0], 2.0, places=6)
            self.assertAlmostEqual(result.aux_operators_evaluated[1][0], 0.0, places=6)
            self.assertAlmostEqual(result.aux_operators_evaluated[2][0], 0.0)
            # metadata
            self.assertIsInstance(result.aux_operators_evaluated[0][1], dict)
            self.assertIsInstance(result.aux_operators_evaluated[1][1], dict)
            self.assertIsInstance(result.aux_operators_evaluated[2][1], dict)

    # Since we perform actions on the aux_operators when a transpiler is set, we have to check that it
    # doesn't affect the final result
    @data(
        None,
        generate_preset_pass_manager(
            backend=THREE_QUBITS_BACKEND, optimization_level=1, seed_transpiler=42
        ),
    )
    def test_aux_operators_dict(self, transpiler):
        """Test dictionary compatibility of aux_operators"""
        wavefunction = self.ry_wavefunction
        wavefunction.num_qubits = 2
        vqe = VQE(
            StatevectorEstimator(seed=42), wavefunction, SLSQP(maxiter=300), transpiler=transpiler
        )

        with self.subTest("Test with an empty dictionary."):
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators={})
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)
            self.assertIsInstance(result.aux_operators_evaluated, dict)
            self.assertEqual(len(result.aux_operators_evaluated), 0)

        with self.subTest("Test with two auxiliary operators."):
            aux_op1 = SparsePauliOp.from_list([("II", 2.0)])
            aux_op2 = SparsePauliOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
            aux_ops = {"aux_op1": aux_op1, "aux_op2": aux_op2}
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=aux_ops)
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)
            self.assertEqual(len(result.aux_operators_evaluated), 2)

            # expectation values
            self.assertAlmostEqual(result.aux_operators_evaluated["aux_op1"][0], 2.0, places=5)
            self.assertAlmostEqual(result.aux_operators_evaluated["aux_op2"][0], 0.0, places=5)
            # metadata
            self.assertIsInstance(result.aux_operators_evaluated["aux_op1"][1], dict)
            self.assertIsInstance(result.aux_operators_evaluated["aux_op2"][1], dict)

        with self.subTest("Test with additional zero operator."):
            extra_ops = {**aux_ops, "zero_operator": 0}
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=extra_ops)
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)
            self.assertEqual(len(result.aux_operators_evaluated), 3)
            # expectation values
            self.assertAlmostEqual(result.aux_operators_evaluated["aux_op1"][0], 2.0, places=5)
            self.assertAlmostEqual(result.aux_operators_evaluated["aux_op2"][0], 0.0, places=5)
            self.assertAlmostEqual(result.aux_operators_evaluated["zero_operator"][0], 0.0)
            # metadata
            self.assertIsInstance(result.aux_operators_evaluated["aux_op1"][1], dict)
            self.assertIsInstance(result.aux_operators_evaluated["aux_op2"][1], dict)
            self.assertIsInstance(result.aux_operators_evaluated["zero_operator"][1], dict)

    @data(None, THREE_QUBITS_BACKEND)
    def test_transpiler(self, backend):
        """Test that the transpiler is called"""
        pass_manager = generate_preset_pass_manager(
            backend=backend, optimization_level=1, seed_transpiler=42
        )
        counts = [0]

        def callback(**kwargs):
            counts[0] = kwargs["count"]

        wavefunction = self.ryrz_wavefunction
        wavefunction.num_qubits = 2
        vqe = VQE(
            estimator=StatevectorEstimator(),
            ansatz=wavefunction,
            optimizer=COBYLA(),
            transpiler=pass_manager,
            transpiler_options={"callback": callback},
        )

        vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertGreater(counts[0], 0)


if __name__ == "__main__":
    unittest.main()
