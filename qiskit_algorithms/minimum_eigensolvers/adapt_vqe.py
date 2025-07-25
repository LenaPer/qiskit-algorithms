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

"""An implementation of the AdaptVQE algorithm."""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Iterable

import numpy as np
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit_algorithms.utils.validation import validate_min
from .minimum_eigensolver import MinimumEigensolver
from .vqe import VQE, VQEResult
from ..observables_evaluator import estimate_observables
from ..variational_algorithm import VariationalAlgorithm

logger = logging.getLogger(__name__)


class TerminationCriterion(Enum):
    """A class enumerating the various finishing criteria."""

    CONVERGED = "Threshold converged"
    CYCLICITY = "Aborted due to a cyclic selection of evolution operators"
    MAXIMUM = "Maximum number of iterations reached"


class AdaptVQE(VariationalAlgorithm, MinimumEigensolver):
    """The Adaptive Variational Quantum Eigensolver algorithm.

    `AdaptVQE <https://arxiv.org/abs/1812.11173>`__ is a quantum algorithm which creates a compact
    ansatz from a set of evolution operators. It iteratively extends the ansatz circuit, by
    selecting the building block that leads to the largest gradient from a set of candidates. In
    chemistry, this is usually a list of orbital excitations. Thus, a common choice of ansatz to be
    used with this algorithm is the Unitary Coupled Cluster ansatz implemented in Qiskit Nature.
    This results in a wavefunction ansatz which is uniquely adapted to the operator whose minimum
    eigenvalue is being determined. This class relies on a supplied instance of :class:`~.VQE` to
    find the minimum eigenvalue. The performance of AdaptVQE significantly depends on the
    minimization routine.

    .. code-block:: python

      from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
      from qiskit_algorithms.optimizers import SLSQP
      from qiskit.primitives import StatevectorEstimator
      from qiskit.circuit.library import EvolvedOperatorAnsatz

      # get your Hamiltonian
      hamiltonian = ...

      # construct your ansatz
      ansatz = EvolvedOperatorAnsatz(...)

      vqe = VQE(StatevectorEstimator(), ansatz, SLSQP())

      adapt_vqe = AdaptVQE(vqe)

      eigenvalue, _ = adapt_vqe.compute_minimum_eigenvalue(hamiltonian)

    The following attributes can be set via the initializer but can also be read and updated once
    the AdaptVQE object has been constructed.

    Attributes:
        solver: a :class:`~.VQE` instance used internally to compute the minimum eigenvalues.
            It is a requirement that the :attr:`~.VQE.ansatz` of this solver is of type
            :class:`~qiskit.circuit.library.EvolvedOperatorAnsatz`.
        gradient_threshold: once all gradients have an absolute value smaller than this threshold,
            the algorithm has converged and terminates.
        eigenvalue_threshold: once the eigenvalue has changed by less than this threshold from one
            iteration to the next, the algorithm has converged and terminates. When this case
            occurs, the excitation included in the final iteration did not result in a significant
            improvement of the eigenvalue and, thus, the results from this iteration are not
            considered.
        max_iterations: the maximum number of iterations for the adaptive loop. If ``None``, the
            algorithm is not bound in its number of iterations.
    """

    def __init__(
        self,
        solver: VQE,
        *,
        gradient_threshold: float = 1e-5,
        eigenvalue_threshold: float = 1e-5,
        max_iterations: int | None = None,
    ) -> None:
        """
        Args:
            solver: a :class:`~.VQE` instance used internally to compute the minimum eigenvalues.
                It is a requirement that the :attr:`~.VQE.ansatz` of this solver is of type
                :class:`~qiskit.circuit.library.EvolvedOperatorAnsatz`.
            gradient_threshold: once all gradients have an absolute value smaller than this
                threshold, the algorithm has converged and terminates. Defaults to ``1e-5``.
            eigenvalue_threshold: once the eigenvalue has changed by less than this threshold from
                one iteration to the next, the algorithm has converged and terminates. When this
                case occurs, the excitation included in the final iteration did not result in a
                significant improvement of the eigenvalue and, thus, the results from this iteration
                are not considered.
            max_iterations: the maximum number of iterations for the adaptive loop. If ``None``, the
                algorithm is not bound in its number of iterations.
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are run when using this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.
        """
        validate_min("gradient_threshold", gradient_threshold, 1e-15)
        validate_min("eigenvalue_threshold", eigenvalue_threshold, 1e-15)

        self.solver = solver
        self.gradient_threshold = gradient_threshold
        self.eigenvalue_threshold = eigenvalue_threshold
        self.max_iterations = max_iterations
        self._tmp_ansatz: EvolvedOperatorAnsatz | None = None
        self._excitation_pool: list[BaseOperator] = []
        self._excitation_list: list[BaseOperator] = []

    @property
    def initial_point(self) -> np.ndarray | None:
        """Returns the initial point of the internal :class:`~.VQE` solver."""
        return self.solver.initial_point

    @initial_point.setter
    def initial_point(self, value: np.ndarray | None) -> None:
        """Sets the initial point of the internal :class:`~.VQE` solver."""
        self.solver.initial_point = value

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def _compute_gradients(
        self,
        theta: list[float],
        operator: SparsePauliOp,
    ) -> ListOrDict[tuple[float, dict[str, BaseOperator]]]:
        """
        Computes the gradients for all available excitation operators.

        Args:
            theta: List of (up to now) optimal parameters.
            operator: operator whose gradient needs to be computed.
        Returns:
            List of pairs consisting of the computed gradient and excitation operator.
        """
        # The excitations operators are applied later as exp(i*theta*excitation).
        # For this commutator, we need to explicitly pull in the imaginary phase.
        commutators = [1j * (operator @ exc - exc @ operator) for exc in self._excitation_pool]
        # We have to call simplify on it since Qiskit doesn't do so for now, see
        # Qiskit/qiskit/issues/14567
        # TODO: Remove the below line once the aforementioned issue is fixed to avoid unnecessary
        #  overhead. The issue will be present in Qiskit 2.1 however
        commutators = [obs.simplify() for obs in commutators]

        # If the ansatz has been transpiled
        if self.solver.ansatz.layout:
            commutators = [
                commutator.apply_layout(self.solver.ansatz.layout) for commutator in commutators
            ]
        res = estimate_observables(self.solver.estimator, self.solver.ansatz, commutators, theta)
        return res

    @staticmethod
    def _check_cyclicity(indices: list[int]) -> bool:
        """
        Auxiliary function to check for cycles in the indices of the selected excitations.

        Args:
            indices: The list of chosen gradient indices.

        Returns:
            Whether repeating sequences of indices have been detected.
        """
        cycle_regex = re.compile(r"(\b.+ .+\b)( \b\1\b)+")
        # reg-ex explanation:
        # 1. (\b.+ .+\b) will match at least two numbers and try to match as many as possible. The
        #    word boundaries in the beginning and end ensure that now numbers are split into digits.
        # 2. the match of this part is placed into capture group 1
        # 3. ( \b\1\b)+ will match a space followed by the contents of capture group 1 (again
        #    delimited by word boundaries to avoid separation into digits).
        # -> this results in any sequence of at least two numbers being detected
        match = cycle_regex.search(" ".join(map(str, indices)))
        logger.debug("Cycle detected: %s", match)
        # Additionally we also need to check whether the last two numbers are identical, because the
        # reg-ex above will only find cycles of at least two consecutive numbers.
        # It is sufficient to assert that the last two numbers are different due to the iterative
        # nature of the algorithm.
        return match is not None or (len(indices) > 1 and indices[-2] == indices[-1])

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: ListOrDict[BaseOperator] | None = None,
    ) -> AdaptVQEResult:
        """Computes the minimum eigenvalue.

        Args:
            operator: Operator whose minimum eigenvalue we want to find.
            aux_operators: Additional auxiliary operators to evaluate.

        Raises:
            TypeError: If an ansatz other than :class:`~.EvolvedOperatorAnsatz` is provided.
            AlgorithmError: If all evaluated gradients lie below the convergence threshold in
                the first iteration of the algorithm.

        Returns:
            An :class:`~.AdaptVQEResult` which is a :class:`~.VQEResult` but also but also
            includes runtime information about the AdaptVQE algorithm like the number of iterations,
            termination criterion, and the final maximum gradient.
        """
        if not isinstance(self.solver.ansatz, EvolvedOperatorAnsatz):
            raise TypeError("The AdaptVQE ansatz must be of the EvolvedOperatorAnsatz type.")

        # Overwrite the solver's ansatz with the initial state
        self._tmp_ansatz = self.solver.ansatz
        self._excitation_pool = self._tmp_ansatz.operators
        # This will transpile the initial state if the solver has a transpiler that is set
        self.solver.ansatz = self._tmp_ansatz.initial_state

        prev_op_indices: list[int] = []
        raw_vqe_result: VQEResult | None = None
        theta: list[float] = []
        max_grad: tuple[float, dict[str, BaseOperator] | None] = (0.0, None)
        self._excitation_list = []
        history: list[complex] = []
        iteration = 0
        while self.max_iterations is None or iteration < self.max_iterations:
            iteration += 1
            logger.info("--- Iteration #%s ---", str(iteration))
            # compute gradients
            logger.debug("Computing gradients")
            cur_grads = self._compute_gradients(theta, operator)
            # pick maximum gradient
            max_grad_index, max_grad = max(  # type: ignore[assignment]
                enumerate(cur_grads),
                # mypy <= 1.10 needs call-overload, for 1.11 its arg-type to suppress the error
                # below then the other is seen as unused hence the additional unused-ignore
                key=lambda item: np.abs(
                    item[1][0]  # type: ignore[arg-type, unused-ignore]
                ),  # type: ignore[call-overload, unused-ignore]
            )
            logger.info(
                "Found maximum gradient %s at index %s",
                str(np.abs(max_grad[0])),
                str(max_grad_index),
            )
            # log gradients
            if np.abs(max_grad[0]) < self.gradient_threshold:
                if iteration == 1:
                    raise AlgorithmError(
                        "All gradients have been evaluated to lie below the convergence threshold "
                        "during the first iteration of the algorithm. Try to either tighten the "
                        "convergence threshold or pick a different ansatz."
                    )
                logger.info(
                    "AdaptVQE terminated successfully with a final maximum gradient: %s",
                    str(np.abs(max_grad[0])),
                )
                termination_criterion = TerminationCriterion.CONVERGED
                break
            # store maximum gradient's index for cycle detection
            prev_op_indices.append(max_grad_index)
            # check indices of picked gradients for cycles
            if self._check_cyclicity(prev_op_indices):
                logger.info("Alternating sequence found. Finishing.")
                logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))
                termination_criterion = TerminationCriterion.CYCLICITY
                break
            # add new excitation to self._ansatz
            logger.info(
                "Adding new operator to the ansatz: %s", str(self._excitation_pool[max_grad_index])
            )
            self._excitation_list.append(self._excitation_pool[max_grad_index])
            theta.append(0.0)
            # setting up the ansatz for the VQE iteration
            self._tmp_ansatz.operators = self._excitation_list
            self.solver.ansatz = self._tmp_ansatz
            self.solver.initial_point = np.asarray(theta)
            # evaluating the eigenvalue with the internal VQE
            prev_raw_vqe_result = raw_vqe_result
            raw_vqe_result = self.solver.compute_minimum_eigenvalue(operator)
            theta = raw_vqe_result.optimal_point.tolist()
            # checking convergence based on the change in eigenvalue
            if iteration > 1:
                eigenvalue_diff = np.abs(raw_vqe_result.eigenvalue - history[-1])
                if eigenvalue_diff < self.eigenvalue_threshold:
                    logger.info(
                        "AdaptVQE terminated successfully with a final change in eigenvalue: %s",
                        str(eigenvalue_diff),
                    )
                    termination_criterion = TerminationCriterion.CONVERGED
                    logger.debug(
                        "Reverting the addition of the last excitation to the ansatz since it "
                        "resulted in a change of the eigenvalue below the configured threshold."
                    )
                    self._excitation_list.pop()
                    theta.pop()
                    self._tmp_ansatz.operators = self._excitation_list
                    self.solver.ansatz = self._tmp_ansatz
                    self.solver.initial_point = np.asarray(theta)
                    raw_vqe_result = prev_raw_vqe_result
                    break
            # appending the computed eigenvalue to the tracking history
            history.append(raw_vqe_result.eigenvalue)
            logger.info("Current eigenvalue: %s", str(raw_vqe_result.eigenvalue))
        else:
            # reached maximum number of iterations
            termination_criterion = TerminationCriterion.MAXIMUM
            logger.info("Maximum number of iterations reached. Finishing.")
            logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))

        result = AdaptVQEResult()
        result.combine(raw_vqe_result)
        result.num_iterations = iteration
        result.final_max_gradient = max_grad[0]
        result.termination_criterion = termination_criterion  # type: ignore[assignment]
        result.eigenvalue_history = history

        # once finished evaluate auxiliary operators if any
        if aux_operators is not None:
            if self.solver.ansatz.layout is not None:
                key_op_iterator: Iterable[tuple[str | int, BaseOperator]]
                if isinstance(aux_operators, list):
                    key_op_iterator = enumerate(aux_operators)
                    # Dummy placeholder
                    converted: ListOrDict[BaseOperator] = [
                        SparsePauliOp.from_list([("I", 0)])
                    ] * len(aux_operators)
                else:
                    key_op_iterator = aux_operators.items()
                    converted = {}
                for key, op in key_op_iterator:
                    if op is not None:
                        converted[key] = op.apply_layout(self.solver.ansatz.layout)

                aux_operators = converted
            aux_values = estimate_observables(
                self.solver.estimator,
                self.solver.ansatz,
                aux_operators,
                result.optimal_point,  # type: ignore[arg-type]
            )
            result.aux_operators_evaluated = aux_values  # type: ignore[assignment]

        logger.info("The final eigenvalue is: %s", str(result.eigenvalue))
        self.solver.ansatz.operators = self._excitation_pool
        return result


class AdaptVQEResult(VQEResult):
    """AdaptVQE Result."""

    def __init__(self) -> None:
        super().__init__()
        self._num_iterations: int | None = None
        self._final_max_gradient: float | None = None
        self._termination_criterion: str = ""
        self._eigenvalue_history: list[complex] | None = None

    @property
    def num_iterations(self) -> int:
        """Returns the number of iterations."""
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, value: int) -> None:
        """Sets the number of iterations."""
        self._num_iterations = value

    @property
    def final_max_gradient(self) -> float:
        """Returns the final maximum gradient."""
        return self._final_max_gradient

    @final_max_gradient.setter
    def final_max_gradient(self, value: float) -> None:
        """Sets the final maximum gradient."""
        self._final_max_gradient = value

    @property
    def termination_criterion(self) -> str:
        """Returns the termination criterion."""
        return self._termination_criterion

    @termination_criterion.setter
    def termination_criterion(self, value: str) -> None:
        """Sets the termination criterion."""
        self._termination_criterion = value

    @property
    def eigenvalue_history(self) -> list[complex]:
        """Returns the history of computed eigenvalues.

        The history's length matches the number of iterations and includes the final computed value.
        """
        return self._eigenvalue_history

    @eigenvalue_history.setter
    def eigenvalue_history(self, eigenvalue_history: list[complex]) -> None:
        """Sets the history of computed eigenvalues."""
        self._eigenvalue_history = eigenvalue_history
