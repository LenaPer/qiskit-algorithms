# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An algorithm to implement a Trotterization real time-evolution."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.parametertable import ParameterView
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.synthesis import ProductFormula, LieTrotter

from qiskit_algorithms.custom_types import Transpiler
from qiskit_algorithms.time_evolvers.time_evolution_problem import TimeEvolutionProblem
from qiskit_algorithms.time_evolvers.time_evolution_result import TimeEvolutionResult
from qiskit_algorithms.time_evolvers.real_time_evolver import RealTimeEvolver
from qiskit_algorithms.observables_evaluator import estimate_observables


class TrotterQRTE(RealTimeEvolver):
    """Quantum Real Time Evolution using Trotterization.

    The type of Trotterization is defined by the :class:`~qiskit.synthesis.ProductFormula`
    provided to the algorithm.

    Examples:

        .. code-block:: python

            from qiskit.quantum_info import Pauli, SparsePauliOp
            from qiskit import QuantumCircuit
            from qiskit_algorithms import TrotterQRTE, TimeEvolutionProblem
            from qiskit.primitives import StatevectorEstimator

            operator = SparsePauliOp([Pauli("X"), Pauli("Z")])
            initial_state = QuantumCircuit(1)
            time = 1
            evolution_problem = TimeEvolutionProblem(operator, time, initial_state)
            # LieTrotter with 1 rep
            estimator = StatevectorEstimator()
            trotter_qrte = TrotterQRTE(estimator=estimator)
            evolved_state = trotter_qrte.evolve(evolution_problem).evolved_state
    """

    def __init__(
        self,
        product_formula: ProductFormula | None = None,
        estimator: BaseEstimatorV2 | None = None,
        num_timesteps: int = 1,
        *,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
        insert_barriers: bool = False,
    ) -> None:
        """
        Args:
            product_formula: A Lie-Trotter-Suzuki product formula. If ``None`` provided (default),
                the :class:`~qiskit.synthesis.LieTrotter` first order product formula with a single
                repetition is used. ``reps`` should be 1 to obtain a number of time-steps equal to
                ``num_timesteps`` and an evaluation of :attr:`.TimeEvolutionProblem.aux_operators`
                at every time-step. If ``reps`` is larger than 1, the true number of time-steps will
                be ``num_timesteps * reps``.
            estimator: An estimator primitive used for calculating expectation values of
                ``TimeEvolutionProblem.aux_operators``.
            num_timesteps: The number of time-steps the full evolution time is divided into
                (repetitions of ``product_formula``).
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are produced within this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.
            insert_barriers: If True, insert a barrier after the initial state and after each Trotter
                step.
        """

        self.product_formula = product_formula
        self.num_timesteps = num_timesteps
        self.estimator = estimator
        self._insert_barriers = insert_barriers
        self._transpiler = transpiler
        self._transpiler_options = transpiler_options if transpiler_options is not None else {}

    @property
    def product_formula(self) -> ProductFormula:
        """Returns a product formula."""
        return self._product_formula

    @product_formula.setter
    def product_formula(self, product_formula: ProductFormula | None):
        """Sets a product formula. If ``None`` provided, sets the Lie-Trotter first order product
        formula with a single repetition."""
        if product_formula is None:
            product_formula = LieTrotter(reps=1)
        self._product_formula = product_formula

    @property
    def estimator(self) -> BaseEstimatorV2 | None:
        """
        Returns an estimator.
        """
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimatorV2) -> None:
        """
        Sets an estimator.
        """
        self._estimator = estimator

    @property
    def num_timesteps(self) -> int:
        """Returns the number of timesteps."""
        return self._num_timesteps

    @num_timesteps.setter
    def num_timesteps(self, num_timesteps: int) -> None:
        """
        Sets the number of time-steps.

        Raises:
            ValueError: If num_timesteps is not positive.
        """
        if num_timesteps <= 0:
            raise ValueError(
                f"Number of time steps must be positive integer, {num_timesteps} provided"
            )
        self._num_timesteps = num_timesteps

    @classmethod
    def supports_aux_operators(cls) -> bool:
        """
        Whether computing the expectation value of auxiliary operators is supported.

        Returns:
            ``True`` if ``aux_operators`` expectations in the ``TimeEvolutionProblem`` can be
            evaluated, ``False`` otherwise.
        """
        return True

    def evolve(self, evolution_problem: TimeEvolutionProblem) -> TimeEvolutionResult:
        """
        Evolves a quantum state for a given time using the Trotterization method
        based on a product formula provided. The result is provided in the form of a quantum
        circuit. If auxiliary operators are included in the ``evolution_problem``, they are
        evaluated on the ``init_state`` and on the evolved state at every step (``num_timesteps``
        times) using an estimator primitive provided.

        Args:
            evolution_problem: Instance defining evolution problem. For the included Hamiltonian,
                ``Pauli`` or ``SparsePauliOp`` are supported by TrotterQRTE.

        Returns:
            Evolution result that includes an evolved state as a quantum circuit and, optionally,
            auxiliary operators evaluated for a resulting state on an estimator primitive.

        Raises:
            ValueError: If ``t_param`` is not set to ``None`` in the ``TimeEvolutionProblem``
                (feature not currently supported).
            ValueError: If ``aux_operators`` provided in the time evolution problem but no estimator
                provided to the algorithm.
            ValueError: If the ``initial_state`` is not provided in the ``TimeEvolutionProblem``.
            ValueError: If an unsupported Hamiltonian type is provided.
        """
        if evolution_problem.aux_operators is not None and self.estimator is None:
            raise ValueError(
                "The time evolution problem contained ``aux_operators`` but no estimator was "
                "provided. The algorithm continues without calculating these quantities. "
            )

        # ensure the hamiltonian is a sparse pauli op
        hamiltonian = evolution_problem.hamiltonian
        if not isinstance(hamiltonian, (Pauli, SparsePauliOp)):
            raise ValueError(
                f"TrotterQRTE only accepts Pauli | SparsePauliOp, {type(hamiltonian)} " "provided."
            )

        if isinstance(hamiltonian, Pauli):
            hamiltonian = SparsePauliOp(hamiltonian)

        t_param = evolution_problem.t_param
        free_parameters = hamiltonian.parameters
        if t_param is not None and free_parameters != ParameterView([t_param]):
            raise ValueError(
                f"Hamiltonian time parameters ({free_parameters}) do not match "
                f"evolution_problem.t_param ({t_param})."
            )

        # make sure PauliEvolutionGate does not implement more than one Trotter step
        dt = evolution_problem.time / self.num_timesteps  # pylint: disable=invalid-name

        if evolution_problem.initial_state is not None:
            initial_state = evolution_problem.initial_state
        else:
            raise ValueError("``initial_state`` must be provided in the ``TimeEvolutionProblem``.")

        evolved_state = QuantumCircuit(initial_state.num_qubits)
        evolved_state.append(initial_state, evolved_state.qubits)

        if self._transpiler is not None:
            evolved_state = self._transpiler.run(evolved_state, **self._transpiler_options)

        if self._insert_barriers:
            evolved_state.barrier()

        if evolution_problem.aux_operators is not None:
            observables = []
            observables.append(
                estimate_observables(
                    self.estimator,
                    evolved_state,
                    evolution_problem.aux_operators,
                    None,
                    evolution_problem.truncation_threshold,
                )
            )
        else:
            observables = None

        # Empty define to avoid possibly undefined lint error later here
        single_step_evolution_gate = None

        if t_param is None:
            # the evolution gate
            single_step_evolution_gate = PauliEvolutionGate(
                hamiltonian, dt, synthesis=self.product_formula
            )

        for n in range(self.num_timesteps):
            # if hamiltonian is time-dependent, bind new time-value at every step to construct
            # evolution for next step
            if t_param is not None:
                time_value = (n + 1) * dt
                bound_hamiltonian = hamiltonian.assign_parameters([time_value])
                single_step_evolution_gate = PauliEvolutionGate(
                    bound_hamiltonian,
                    dt,
                    synthesis=self.product_formula,
                )
            evolved_state.append(single_step_evolution_gate, evolved_state.qubits)

            if self._transpiler is not None:
                evolved_state = self._transpiler.run(evolved_state, **self._transpiler_options)

            if self._insert_barriers:
                evolved_state.barrier()

            if evolution_problem.aux_operators is not None:
                observables.append(
                    estimate_observables(
                        self.estimator,
                        evolved_state,
                        evolution_problem.aux_operators,
                        None,
                        evolution_problem.truncation_threshold,
                    )
                )

        evaluated_aux_ops = None
        if evolution_problem.aux_operators is not None:
            evaluated_aux_ops = observables[-1]

        return TimeEvolutionResult(
            evolved_state, evaluated_aux_ops, observables  # type: ignore[arg-type]
        )
