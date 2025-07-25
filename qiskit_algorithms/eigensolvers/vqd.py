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

"""The Variational Quantum Deflation Algorithm for computing higher energy states.

See https://arxiv.org/abs/1805.08138.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence, Iterable
from time import time
from typing import Any, cast

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit_algorithms.state_fidelities import BaseStateFidelity
from .eigensolver import Eigensolver, EigensolverResult
from ..custom_types import Transpiler
from ..exceptions import AlgorithmError
from ..list_or_dict import ListOrDict
from ..observables_evaluator import estimate_observables
from ..optimizers import Optimizer, Minimizer, OptimizerResult
from ..utils import validate_bounds, validate_initial_point

# private function as we expect this to be updated in the next release
from ..utils.set_batching import _set_default_batchsize
from ..variational_algorithm import VariationalAlgorithm

logger = logging.getLogger(__name__)


class VQD(VariationalAlgorithm, Eigensolver):
    r"""The Variational Quantum Deflation algorithm. Implementation using primitives.

    `VQD <https://arxiv.org/abs/1805.08138>`__ is a quantum algorithm that uses a
    variational technique to find
    the k lowest eigenvalues of the Hamiltonian :math:`H` of a given system.

    The algorithm computes excited state energies of generalised hamiltonians
    by optimizing over a modified cost function where each successive eigenvalue
    is calculated iteratively by introducing an overlap term with all
    the previously computed eigenstates that must be minimised, thus ensuring
    higher energy eigenstates are found.

    An instance of VQD requires defining three algorithmic subcomponents:
    an integer k denoting the number of eigenstates to calculate, a trial
    state (a.k.a. ansatz) which is a :class:`QuantumCircuit`,
    and one instance (or list of) classical :mod:`~qiskit_algorithms.optimizers`.
    The optimizer varies the circuit parameters
    The trial state :math:`|\psi(\vec\theta)\rangle` is varied by the optimizer,
    which modifies the set of ansatz parameters :math:`\vec\theta`
    such that the expectation value of the operator on the corresponding
    state approaches a minimum. The algorithm does this by iteratively refining
    each excited state to be orthogonal to all the previous excited states.

    An optional array of parameter values, via the *initial_point*, may be provided
    as the starting point for the search of the minimum eigenvalue. This feature is
    particularly useful when there are reasons to believe that the solution point
    is close to a particular point.

    The length of the *initial_point* list value must match the number of the parameters
    expected by the ansatz. If the *initial_point* is left at the default
    of ``None``, then VQD will look to the ansatz for a preferred value, based on its
    given initial state. If the ansatz returns ``None``,
    then a random point will be generated within the parameter bounds set, as per above.
    It is also possible to give a list of initial points, one for every kth eigenvalue.
    If the ansatz provides ``None`` as the lower bound, then VQD
    will default it to :math:`-2\pi`; similarly, if the ansatz returns ``None``
    as the upper bound, the default value will be :math:`2\pi`.

    The following attributes can be set via the initializer but can also be read and
    updated once the VQD object has been constructed.

    Attributes:
            estimator (BaseEstimatorV2): The primitive instance used to perform the expectation
                estimation as indicated in the VQD paper.
            fidelity (BaseStateFidelity): The fidelity class instance used to compute the
                overlap estimation as indicated in the VQD paper.
            optimizer(Optimizer | Sequence[Optimizer]): A classical optimizer or a list of optimizers,
                one for every k-th eigenvalue. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            k (int): the number of eigenvalues to return. Returns the lowest k eigenvalues.
            betas (list[float]): Beta parameters in the VQD paper.
                Should have length k - 1, with k the number of excited states.
                These hyper-parameters balance the contribution of each overlap term to the cost
                function and have a default value computed as the mean square sum of the
                coefficients of the observable.
            callback (Callable[[int, np.ndarray, float, dict[str, Any]], None] | None):
                A callback that can access the intermediate data
                during the optimization. Four parameter values are passed to the callback as
                follows during each evaluation by the optimizer: the evaluation count,
                the optimizer parameters for the ansatz, the estimated value, the estimation
                metadata, and the current step.
            convergence_threshold: A threshold under which the algorithm is considered to have
                converged. It corresponds to the maximal average fidelity an eigenstate is allowed
                to have with the previous eigenstates. If set to None, no check is performed.
    """

    def __init__(
        self,
        estimator: BaseEstimatorV2,
        fidelity: BaseStateFidelity,
        ansatz: QuantumCircuit,
        optimizer: Optimizer | Minimizer | Sequence[Optimizer | Minimizer],
        *,
        k: int = 2,
        betas: np.ndarray | None = None,
        initial_point: np.ndarray | list[np.ndarray] | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any], int], None] | None = None,
        convergence_threshold: float | None = None,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ) -> None:
        """

        Args:
            estimator: The estimator primitive.
            fidelity: The fidelity class using primitives.
            ansatz: A parameterized circuit used as ansatz for the wave function.
            optimizer: A classical optimizer or a list of optimizers, one for every k-th eigenvalue.
                Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            k: The number of eigenvalues to return. Returns the lowest k eigenvalues.
            betas: Beta parameters in the VQD paper.
                Should have length k - 1, with k the number of excited states.
                These hyperparameters balance the contribution of each overlap term to the cost
                function and have a default value computed as the mean square sum of the
                coefficients of the observable.
            initial_point: An optional initial point (i.e. initial parameter values)
                or a list of initial points (one for every k-th eigenvalue)
                for the optimizer.
                If ``None`` then VQD will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            callback: A callback that can access the intermediate data
                during the optimization. Five parameter values are passed to the callback as
                follows during each evaluation by the optimizer: the evaluation count,
                the optimizer parameters for the ansatz, the estimated value,
                the estimation metadata, and the current step.
            convergence_threshold: A threshold under which the algorithm is considered to have
                converged. It corresponds to the maximal average fidelity an eigenstate is allowed
                to have with the previous eigenstates. If set to None, no check is performed.
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are run when using this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.
        """
        super().__init__()

        self.estimator = estimator
        self.fidelity = fidelity
        self._ansatz = ansatz
        self.optimizer = optimizer
        self.k = k
        self.betas = betas
        # this has to go via getters and setters due to the VariationalAlgorithm interface
        self.initial_point = initial_point
        self.callback = callback
        self.convergence_threshold = convergence_threshold

        self._transpiler = transpiler
        self._transpiler_options = transpiler_options if transpiler_options is not None else {}

        if self._transpiler is not None:
            self.ansatz = ansatz

        self._eval_count = 0

    @property
    def initial_point(self) -> np.ndarray | list[np.ndarray] | None:
        """Returns initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: np.ndarray | list[np.ndarray] | None):
        """Sets initial point"""
        self._initial_point = initial_point

    @property
    def ansatz(self) -> QuantumCircuit:
        """
        A parameterized circuit used as ansatz for the wave function. If a transpiler has been
        provided, the ansatz will be automatically transpiled upon being set.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, value: QuantumCircuit | None) -> None:
        if self._transpiler is not None:
            self._ansatz = self._transpiler.run(value, **self._transpiler_options)
        else:
            self._ansatz = value

    def _check_operator_ansatz(self, operator: BaseOperator):
        """Check that the number of qubits of operator and ansatz match."""
        if operator is not None and self.ansatz is not None:
            if operator.num_qubits != self.ansatz.num_qubits:
                # try to set the number of qubits on the ansatz, if possible
                try:
                    self.ansatz.num_qubits = operator.num_qubits
                except AttributeError as exc:
                    raise AlgorithmError(
                        "The number of qubits of the ansatz does not match the "
                        "operator, and the ansatz does not allow setting the "
                        "number of qubits using `num_qubits`."
                    ) from exc

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def compute_eigenvalues(
        self,
        operator: BaseOperator,
        aux_operators: ListOrDict[BaseOperator] | None = None,
    ) -> VQDResult:
        super().compute_eigenvalues(operator, aux_operators)

        if self.ansatz.layout is not None:
            operator = operator.apply_layout(self.ansatz.layout)

        # this sets the size of the ansatz, so it must be called before the initial point
        # validation
        self._check_operator_ansatz(operator)

        bounds = validate_bounds(self.ansatz)

        # We need to handle the array entries being zero or Optional i.e. having value None
        if aux_operators:
            if self.ansatz.layout is not None:
                # len(self.ansatz.layout.final_index_layout()) is the original number of qubits in the
                # ansatz, before transpilation
                zero_op = SparsePauliOp.from_list(
                    [("I" * len(self.ansatz.layout.final_index_layout()), 0)]
                )
            else:
                zero_op = SparsePauliOp.from_list([("I" * self.ansatz.num_qubits, 0)])

            # Convert the None and zero values when aux_operators is a list.
            # Drop None and convert zero values when aux_operators is a dict.
            key_op_iterator: Iterable[tuple[str | int, BaseOperator]]
            if isinstance(aux_operators, list):
                aux_operators = [op if op is not None else zero_op for op in aux_operators]
                key_op_iterator = enumerate(aux_operators)
                converted: ListOrDict[BaseOperator] = [zero_op] * len(aux_operators)
            else:
                key_op_iterator = aux_operators.items()
                converted = {}
            for key, op in key_op_iterator:
                if op is not None:
                    converted[key] = zero_op if op == 0 else op  # type: ignore[index]

                    if self.ansatz.layout is not None:
                        converted[key] = converted[key].apply_layout(self.ansatz.layout)

            aux_operators = converted

        else:
            aux_operators = None

        betas = self.betas
        if self.betas is None:
            try:
                upper_bound = sum(np.abs(operator.coeffs))

            except Exception as exc:
                raise NotImplementedError(
                    r"Beta autoevaluation is not supported for operators"
                    f"of type {type(operator)}."
                ) from exc

            betas = np.asarray([upper_bound * 10] * self.k)
            logger.info("beta autoevaluated to %s", betas[0])

        result = self._build_vqd_result()

        if aux_operators is not None:
            aux_values = []

        # We keep a list of the bound circuits with optimal parameters, to avoid re-binding
        # the same parameters to the ansatz if we do multiple steps
        prev_states = []

        # These two variables are defined inside if statements and static analysis, e.g. lint can
        # see this as a potential error of them not being defined before use. Following the logic
        # they do end up being defined before use so the setting of these here, these values would
        # not be used in practice.
        initial_point = np.asarray([])
        initial_points = np.asarray([])

        num_initial_points = 0
        if self.initial_point is not None:
            initial_points = np.reshape(self.initial_point, (-1, self.ansatz.num_parameters))
            num_initial_points = len(initial_points)

        # 0 just means the initial point is ``None`` and ``validate_initial_point``
        # will select a random point
        if num_initial_points <= 1:
            initial_point = validate_initial_point(
                self.initial_point, self.ansatz  # type: ignore[arg-type]
            )

        current_optimal_point: dict[str, Any] = {"optimal_value": float("inf")}

        for step in range(1, self.k + 1):
            current_optimal_point["optimal_value"] = float("inf")

            if num_initial_points > 1:
                initial_point = validate_initial_point(initial_points[step - 1], self.ansatz)

            if step > 1:
                prev_states.append(self.ansatz.assign_parameters(current_optimal_point["x"]))

            self._eval_count = 0
            energy_evaluation = self._get_evaluate_energy(
                step,
                operator,
                betas,
                prev_states=prev_states,
                current_optimal_point=current_optimal_point,
            )

            start_time = time()

            # TODO: add gradient support after FidelityGradients are implemented
            if isinstance(self.optimizer, Sequence):
                optimizer = self.optimizer[step - 1]
            else:
                optimizer = self.optimizer  # fall back to single optimizer if not list

            if callable(optimizer):
                opt_result = optimizer(  # pylint: disable=not-callable
                    fun=energy_evaluation,  # type: ignore[arg-type]
                    x0=initial_point,
                    jac=None,
                    bounds=bounds,
                )
            else:
                # we always want to submit as many estimations per job as possible for minimal
                # overhead on the hardware
                was_updated = _set_default_batchsize(optimizer)

                opt_result = optimizer.minimize(
                    fun=energy_evaluation, x0=initial_point, bounds=bounds  # type: ignore[arg-type]
                )

                # reset to original value
                if was_updated:
                    optimizer.set_max_evals_grouped(None)

            eval_time = time() - start_time

            self._update_vqd_result(
                result, opt_result, eval_time, self.ansatz.copy(), current_optimal_point
            )

            if aux_operators is not None:
                aux_value = estimate_observables(
                    self.estimator, self.ansatz, aux_operators, current_optimal_point["x"]
                )
                aux_values.append(aux_value)

            if step == 1:
                logger.info(
                    "Ground state optimization complete in %s seconds.\n"
                    "Found opt_params %s in %s evals",
                    eval_time,
                    result.optimal_points,
                    self._eval_count,
                )
            else:
                average_fidelity = current_optimal_point["total_fidelity"][0] / (step - 1)

                if (
                    self.convergence_threshold is not None
                    and average_fidelity > self.convergence_threshold
                ):
                    last_digit = step % 10

                    if last_digit == 1 and step % 100 != 11:
                        suffix = "st"
                    elif last_digit == 2:
                        suffix = "nd"
                    elif last_digit == 3:
                        suffix = "rd"
                    else:
                        suffix = "th"

                    raise AlgorithmError(
                        f"Convergence threshold is set to {self.convergence_threshold} but an "
                        f"average (weighted by the betas) fidelity of {average_fidelity:.5f} with "
                        f"the previous eigenstates has been observed during the evaluation of the "
                        f"{step}{suffix} lowest eigenvalue."
                    )
                logger.info(
                    (
                        "%s excited state optimization complete in %s s.\n"
                        "Found opt_params %s in %s evals"
                    ),
                    str(step - 1),
                    eval_time,
                    result.optimal_points,
                    self._eval_count,
                )

        # To match the signature of EigensolverResult
        result.eigenvalues = np.array(result.eigenvalues)

        if aux_operators is not None:
            result.aux_operators_evaluated = aux_values

        return result

    def _get_evaluate_energy(  # pylint: disable=too-many-positional-arguments
        self,
        step: int,
        operator: BaseOperator,
        betas: np.ndarray,
        current_optimal_point: dict["str", Any],
        prev_states: list[QuantumCircuit] | None = None,
    ) -> Callable[[np.ndarray], float | np.ndarray]:
        """Returns a function handle to evaluate the ansatz's energy for any given parameters.
            This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            step: level of energy being calculated. 1 for ground, 2 for first excited state...
            operator: The operator whose energy to evaluate.
            betas: Beta parameters in the VQD paper.
            prev_states: List of optimal circuits from previous rounds of optimization.
            current_optimal_point: A dict to keep track of the current optimal point, which is used
                to check the algorithm's convergence.

        Returns:
            A callable that computes and returns the energy of the hamiltonian
            of each parameter.

        Raises:
            AlgorithmError: If the circuit is not parameterized (i.e. has 0 free parameters).
            AlgorithmError: If operator was not provided.
            RuntimeError: If the previous states array is of the wrong size.
        """

        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise AlgorithmError("The ansatz must be parameterized, but has no free parameters.")

        if step > 1 and (len(prev_states) + 1) != step:
            raise RuntimeError(
                f"Passed previous states of the wrong size."
                f"Passed array has length {str(len(prev_states))}"
            )

        def evaluate_energy(parameters: np.ndarray) -> float | np.ndarray:
            # handle broadcasting: ensure parameters is of shape [array, array, ...]
            if len(parameters.shape) == 1:
                parameters = np.reshape(parameters, (-1, num_parameters))
            batch_size = len(parameters)

            estimator_job = self.estimator.run([(self.ansatz, operator, parameters)])

            total_cost = np.zeros(batch_size)

            if step > 1:
                # compute overlap cost
                batched_prev_states = [state for state in prev_states for _ in range(batch_size)]
                fidelity_job = self.fidelity.run(
                    batch_size * [self.ansatz] * (step - 1),
                    batched_prev_states,
                    np.tile(parameters, (step - 1, 1)),  # type: ignore[arg-type]
                )
                costs = fidelity_job.result().fidelities

                costs = np.reshape(costs, (step - 1, -1))
                for state, cost in enumerate(costs):
                    total_cost += np.real(betas[state] * cost)

            try:
                estimator_result = estimator_job.result()[0]

            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

            values = estimator_result.data.evs + total_cost

            if self.callback is not None:
                for params, value in zip(parameters, values):
                    self._eval_count += 1
                    self.callback(self._eval_count, params, value, estimator_result.metadata, step)
            else:
                self._eval_count += len(values)

            for param, value in zip(parameters, values):
                if value < current_optimal_point["optimal_value"]:
                    current_optimal_point["optimal_value"] = value
                    current_optimal_point["x"] = param

                    if step > 1:
                        current_optimal_point["total_fidelity"] = total_cost
                        current_optimal_point["eigenvalue"] = (value - total_cost)[0]
                    else:
                        current_optimal_point["eigenvalue"] = value

            return values if len(values) > 1 else values[0]

        return evaluate_energy

    @staticmethod
    def _build_vqd_result() -> VQDResult:
        result = VQDResult()
        result.optimal_points = np.array([])
        result.optimal_parameters = []
        result.optimal_values = np.array([])
        result.cost_function_evals = np.array([], dtype=int)
        result.optimizer_times = np.array([])
        result.eigenvalues = []  # type: ignore[assignment]
        result.optimizer_results = []
        result.optimal_circuits = []
        return result

    @staticmethod
    def _update_vqd_result(
        result: VQDResult, opt_result: OptimizerResult, eval_time, ansatz, optimal_point
    ) -> VQDResult:
        result.optimal_points = (
            np.concatenate([result.optimal_points, [optimal_point["x"]]])
            if len(result.optimal_points) > 0
            else np.array([optimal_point["x"]])
        )
        result.optimal_parameters.append(
            dict(zip(ansatz.parameters, cast(np.ndarray, optimal_point["x"])))
        )
        result.optimal_values = np.concatenate(
            [result.optimal_values, [optimal_point["optimal_value"]]]
        )
        result.cost_function_evals = np.concatenate([result.cost_function_evals, [opt_result.nfev]])
        result.optimizer_times = np.concatenate([result.optimizer_times, [eval_time]])
        result.eigenvalues.append(optimal_point["eigenvalue"] + 0j)  # type: ignore[attr-defined]
        result.optimizer_results.append(opt_result)
        result.optimal_circuits.append(ansatz)
        return result


class VQDResult(EigensolverResult):
    """VQD Result."""

    def __init__(self) -> None:
        super().__init__()

        self._cost_function_evals: np.ndarray | None = None
        self._optimizer_times: np.ndarray | None = None
        self._optimal_values: np.ndarray | None = None
        self._optimal_points: np.ndarray | None = None
        self._optimal_parameters: list[dict] | None = None
        self._optimizer_results: list[OptimizerResult] | None = None
        self._optimal_circuits: list[QuantumCircuit] | None = None

    @property
    def cost_function_evals(self) -> np.ndarray | None:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: np.ndarray) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value

    @property
    def optimizer_times(self) -> np.ndarray | None:
        """Returns time taken for optimization for each step"""
        return self._optimizer_times

    @optimizer_times.setter
    def optimizer_times(self, value: np.ndarray) -> None:
        """Sets time taken for optimization for each step"""
        self._optimizer_times = value

    @property
    def optimal_values(self) -> np.ndarray | None:
        """Returns optimal value for each step"""
        return self._optimal_values

    @optimal_values.setter
    def optimal_values(self, value: np.ndarray) -> None:
        """Sets optimal values"""
        self._optimal_values = value

    @property
    def optimal_points(self) -> np.ndarray | None:
        """Returns optimal point for each step"""
        return self._optimal_points

    @optimal_points.setter
    def optimal_points(self, value: np.ndarray) -> None:
        """Sets optimal points"""
        self._optimal_points = value

    @property
    def optimal_parameters(self) -> list[dict] | None:
        """Returns the optimal parameters for each step"""
        return self._optimal_parameters

    @optimal_parameters.setter
    def optimal_parameters(self, value: list[dict]) -> None:
        """Sets optimal parameters"""
        self._optimal_parameters = value

    @property
    def optimizer_results(self) -> list[OptimizerResult] | None:
        """Returns the optimizer results for each step"""
        return self._optimizer_results

    @optimizer_results.setter
    def optimizer_results(self, value: list[OptimizerResult]) -> None:
        """Sets optimizer results"""
        self._optimizer_results = value

    @property
    def optimal_circuits(self) -> list[QuantumCircuit] | None:
        """The optimal circuits. Along with the optimal parameters,
        these can be used to retrieve the different eigenstates."""
        return self._optimal_circuits

    @optimal_circuits.setter
    def optimal_circuits(self, optimal_circuits: list[QuantumCircuit]) -> None:
        self._optimal_circuits = optimal_circuits
