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

"""The variational quantum eigensolver algorithm."""

from __future__ import annotations

import logging
from time import time
from collections.abc import Callable
from typing import Any, Iterable

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit_algorithms.gradients import BaseEstimatorGradient
from ..custom_types import Transpiler

from ..exceptions import AlgorithmError
from ..list_or_dict import ListOrDict
from ..optimizers import Optimizer, Minimizer, OptimizerResult
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .minimum_eigensolver import MinimumEigensolver, MinimumEigensolverResult
from ..observables_evaluator import estimate_observables
from ..utils import validate_initial_point, validate_bounds

# private function as we expect this to be updated in the next released
from ..utils.set_batching import _set_default_batchsize

logger = logging.getLogger(__name__)


class VQE(VariationalAlgorithm, MinimumEigensolver):
    r"""The Variational Quantum Eigensolver (VQE) algorithm.

    VQE is a hybrid quantum-classical algorithm that uses a variational technique to find the
    minimum eigenvalue of a given Hamiltonian operator :math:`H`.

    The ``VQE`` algorithm is executed using an :attr:`estimator` primitive, which computes
    expectation values of operators (observables).

    An instance of ``VQE`` also requires an :attr:`ansatz`, a parameterized
    :class:`.QuantumCircuit`, to prepare the trial state :math:`|\psi(\vec\theta)\rangle`. It also
    needs a classical :attr:`optimizer` which varies the circuit parameters :math:`\vec\theta` such
    that the expectation value of the operator on the corresponding state approaches a minimum,

    .. math::

        \min_{\vec\theta} \langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle.

    The :attr:`estimator` is used to compute this expectation value for every optimization step.

    The optimizer can either be one of Qiskit's optimizers, such as
    :class:`~qiskit_algorithms.optimizers.SPSA` or a callable with the following signature:

    .. code-block:: python

        from qiskit_algorithms.optimizers import OptimizerResult

        def my_minimizer(fun, x0, jac=None, bounds=None) -> OptimizerResult:
            # Note that the callable *must* have these argument names!
            # Args:
            #     fun (callable): the function to minimize
            #     x0 (np.ndarray): the initial point for the optimization
            #     jac (callable, optional): the gradient of the objective function
            #     bounds (list, optional): a list of tuples specifying the parameter bounds

            result = OptimizerResult()
            result.x = # optimal parameters
            result.fun = # optimal function value
            return result

    The above signature also allows one to use any SciPy minimizer, for instance as

    .. code-block:: python

        from functools import partial
        from scipy.optimize import minimize

        optimizer = partial(minimize, method="L-BFGS-B")

    The following attributes can be set via the initializer but can also be read and updated once
    the VQE object has been constructed.

    Attributes:
        estimator (BaseEstimatorV2): The estimator primitive to compute the expectation value of the
            Hamiltonian operator.
        optimizer (Optimizer | Minimizer): A classical optimizer to find the minimum energy. This
            can either be a Qiskit :class:`.Optimizer` or a callable implementing the
            :class:`.Minimizer` protocol.
        gradient (BaseEstimatorGradient | None): An optional estimator gradient to be used with the
            optimizer.
        callback (Callable[[int, np.ndarray, float, dict[str, Any]], None] | None): A callback that
            can access the intermediate data at each optimization step. These data are: the
            evaluation count, the optimizer parameters for the ansatz, the evaluated mean, and the
            metadata dictionary.

    References:
        [1]: Peruzzo, A., et al, "A variational eigenvalue solver on a quantum processor"
            `arXiv:1304.3061 <https://arxiv.org/abs/1304.3061>`__
    """

    def __init__(
        self,
        estimator: BaseEstimatorV2,
        ansatz: QuantumCircuit,
        optimizer: Optimizer | Minimizer,
        *,
        gradient: BaseEstimatorGradient | None = None,
        initial_point: np.ndarray | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ) -> None:
        r"""
        Args:
            estimator: The estimator primitive to compute the expectation value of the
                Hamiltonian operator.
            ansatz: A parameterized quantum circuit to prepare the trial state.
            optimizer: A classical optimizer to find the minimum energy. This can either be a
                Qiskit :class:`.Optimizer` or a callable implementing the :class:`.Minimizer`
                protocol.
            gradient: An optional estimator gradient to be used with the optimizer.
            initial_point: An optional initial point (i.e. initial parameter values) for the
                optimizer. The length of the initial point must match the number of :attr:`ansatz`
                parameters. If ``None``, a random point will be generated within certain parameter
                bounds. ``VQE`` will look to the ansatz for these bounds. If the ansatz does not
                specify bounds, bounds of :math:`-2\pi`, :math:`2\pi` will be used.
            callback: A callback that can access the intermediate data at each optimization step.
                These data are: the evaluation count, the optimizer parameters for the ansatz, the
                estimated value, and the metadata dictionary.
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are run when using this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.
        """
        super().__init__()

        self.estimator = estimator
        self._ansatz = ansatz
        self.optimizer = optimizer
        self.gradient = gradient
        # this has to go via getters and setters due to the VariationalAlgorithm interface
        self.initial_point = initial_point
        self.callback = callback

        self._transpiler = transpiler
        self._transpiler_options = transpiler_options if transpiler_options is not None else {}

        if self._transpiler is not None:
            self.ansatz = ansatz

    @property
    def initial_point(self) -> np.ndarray | None:
        return self._initial_point

    @initial_point.setter
    def initial_point(self, value: np.ndarray | None) -> None:
        self._initial_point = value

    @property
    def ansatz(self) -> QuantumCircuit:
        """
        A parameterized quantum circuit to prepare the trial state. If a transpiler has been
        provided, the ansatz will be automatically transpiled upon being set.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, value: QuantumCircuit | None) -> None:
        if self._transpiler is not None:
            self._ansatz = self._transpiler.run(value, **self._transpiler_options)
        else:
            self._ansatz = value

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: ListOrDict[BaseOperator] | None = None,
    ) -> VQEResult:
        if self.ansatz.layout is not None:
            operator = operator.apply_layout(self.ansatz.layout)

        self._check_operator_ansatz(operator)

        initial_point = validate_initial_point(self.initial_point, self.ansatz)

        bounds = validate_bounds(self.ansatz)

        start_time = time()

        evaluate_energy = self._get_evaluate_energy(self.ansatz, operator)

        if self.gradient is not None:
            evaluate_gradient = self._get_evaluate_gradient(self.ansatz, operator)
        else:
            evaluate_gradient = None

        # perform optimization
        if callable(self.optimizer):
            optimizer_result = self.optimizer(
                fun=evaluate_energy,  # type: ignore[arg-type]
                x0=initial_point,
                jac=evaluate_gradient,
                bounds=bounds,
            )
        else:
            # we always want to submit as many estimations per job as possible for minimal
            # overhead on the hardware
            was_updated = _set_default_batchsize(self.optimizer)

            optimizer_result = self.optimizer.minimize(
                fun=evaluate_energy,  # type: ignore[arg-type]
                x0=initial_point,
                jac=evaluate_gradient,  # type: ignore[arg-type]
                bounds=bounds,
            )

            # reset to original value
            if was_updated:
                self.optimizer.set_max_evals_grouped(None)

        optimizer_time = time() - start_time

        logger.info(
            "Optimization complete in %s seconds.\nFound optimal point %s",
            optimizer_time,
            optimizer_result.x,
        )

        if aux_operators is not None:
            if self.ansatz.layout is not None:
                # We need to handle the array entries being zero or Optional i.e. having value None
                # len(self.ansatz.layout.final_index_layout()) is the original number of qubits in the
                # ansatz, before transpilation
                zero_op = SparsePauliOp.from_list(
                    [("I" * len(self.ansatz.layout.final_index_layout()), 0)]
                )
                key_op_iterator: Iterable[tuple[str | int, BaseOperator]]
                if isinstance(aux_operators, list):
                    key_op_iterator = enumerate(aux_operators)
                    converted: ListOrDict[BaseOperator] = [zero_op] * len(aux_operators)
                else:
                    key_op_iterator = aux_operators.items()
                    converted = {}
                for key, op in key_op_iterator:
                    if op is not None:
                        converted[key] = (
                            zero_op.apply_layout(self.ansatz.layout)
                            if op == 0
                            else op.apply_layout(self.ansatz.layout)
                        )

                aux_operators = converted
            aux_operators_evaluated = estimate_observables(
                self.estimator,
                self.ansatz,
                aux_operators,
                optimizer_result.x,  # type: ignore[arg-type]
            )
        else:
            aux_operators_evaluated = None

        return self._build_vqe_result(
            self.ansatz,
            optimizer_result,
            aux_operators_evaluated,  # type: ignore[arg-type]
            optimizer_time,
        )

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def _get_evaluate_energy(
        self,
        ansatz: QuantumCircuit,
        operator: BaseOperator,
    ) -> Callable[[np.ndarray], np.ndarray | float]:
        """Returns a function handle to evaluate the energy at given parameters for the ansatz.
        This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            ansatz: The ansatz preparing the quantum state.
            operator: The operator whose energy to evaluate.

        Returns:
            A callable that computes and returns the energy of the hamiltonian of each parameter.

        Raises:
            AlgorithmError: If the primitive job to evaluate the energy fails.
        """
        num_parameters = ansatz.num_parameters

        # avoid creating an instance variable to remain stateless regarding results
        eval_count = 0

        def evaluate_energy(parameters: np.ndarray) -> np.ndarray | float:
            nonlocal eval_count

            # handle broadcasting: ensure parameters is of shape [array, array, ...]
            parameters = np.reshape(parameters, (-1, num_parameters))

            try:
                job = self.estimator.run([(ansatz, operator, parameters)])
                estimator_result = job.result()[0]
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

            values = estimator_result.data.evs

            if not values.shape:
                values = values.reshape(1)

            if self.callback is not None:
                for params, value in zip(parameters.reshape(-1, 1), values):
                    eval_count += 1
                    self.callback(eval_count, params, value, estimator_result.metadata)

            energy = values[0] if len(values) == 1 else values

            return energy

        return evaluate_energy

    def _get_evaluate_gradient(
        self,
        ansatz: QuantumCircuit,
        operator: BaseOperator,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Get a function handle to evaluate the gradient at given parameters for the ansatz.

        Args:
            ansatz: The ansatz preparing the quantum state.
            operator: The operator whose energy to evaluate.

        Returns:
            A function handle to evaluate the gradient at given parameters for the ansatz.

        Raises:
            AlgorithmError: If the primitive job to evaluate the gradient fails.
        """

        def evaluate_gradient(parameters: np.ndarray) -> np.ndarray:
            # broadcasting not required for the estimator gradients
            try:
                job = self.gradient.run(
                    [ansatz], [operator], [parameters]  # type: ignore[list-item]
                )
                gradients = job.result().gradients
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the gradient failed!") from exc

            return gradients[0]

        return evaluate_gradient

    def _check_operator_ansatz(self, operator: BaseOperator):
        """Check that the number of qubits of operator and ansatz match and that the ansatz is
        parameterized.
        """
        if operator.num_qubits != self.ansatz.num_qubits:
            try:
                logger.info(
                    "Trying to resize ansatz to match operator on %s qubits.", operator.num_qubits
                )
                self.ansatz.num_qubits = operator.num_qubits
            except AttributeError as error:
                raise AlgorithmError(
                    "The number of qubits of the ansatz does not match the "
                    "operator, and the ansatz does not allow setting the "
                    "number of qubits using `num_qubits`."
                ) from error

        if self.ansatz.num_parameters == 0:
            raise AlgorithmError("The ansatz must be parameterized, but has no free parameters.")

    def _build_vqe_result(
        self,
        ansatz: QuantumCircuit,
        optimizer_result: OptimizerResult,
        aux_operators_evaluated: ListOrDict[tuple[complex, tuple[complex, int]]],
        optimizer_time: float,
    ) -> VQEResult:
        result = VQEResult()
        result.optimal_circuit = ansatz.copy()
        result.eigenvalue = optimizer_result.fun
        result.cost_function_evals = optimizer_result.nfev
        result.optimal_point = optimizer_result.x  # type: ignore[assignment]
        result.optimal_parameters = dict(
            zip(self.ansatz.parameters, optimizer_result.x)  # type: ignore[arg-type]
        )
        result.optimal_value = optimizer_result.fun
        result.optimizer_time = optimizer_time
        result.aux_operators_evaluated = aux_operators_evaluated  # type: ignore[assignment]
        result.optimizer_result = optimizer_result
        return result


class VQEResult(VariationalResult, MinimumEigensolverResult):
    """The Variational Quantum Eigensolver (VQE) result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals: int | None = None

    @property
    def cost_function_evals(self) -> int | None:
        """The number of cost optimizer evaluations."""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        self._cost_function_evals = value
