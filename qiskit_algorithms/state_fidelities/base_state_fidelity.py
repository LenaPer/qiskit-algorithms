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
"""
Base state fidelity interface
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import cast, Sequence, List, Any
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from ..algorithm_job import AlgorithmJob
from ..custom_types import Transpiler
from ..utils.circuit_key import _circuit_key


class BaseStateFidelity(ABC):
    r"""
    An interface to calculate state fidelities (state overlaps) for pairs of
    (parametrized) quantum circuits. The calculation depends on the particular
    fidelity method implementation, but can be always defined as the state overlap:

    .. math::

            |\langle\psi(x)|\phi(y)\rangle|^2

    where :math:`x` and :math:`y` are optional parametrizations of the
    states :math:`\psi` and :math:`\phi` prepared by the circuits
    ``circuit_1`` and ``circuit_2``, respectively.

    """

    def __init__(
        self,
        *,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ) -> None:
        r"""
        Args:
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are produced within this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.
        """
        # use cache for preventing unnecessary circuit compositions
        self._circuit_cache: MutableMapping[tuple[int, int], QuantumCircuit] = {}
        self._transpiler = transpiler
        self._transpiler_options = transpiler_options if transpiler_options is not None else {}

    @staticmethod
    def _preprocess_values(
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        values: Sequence[float] | Sequence[Sequence[float]] | None = None,
    ) -> Sequence[list[float]]:
        """
        Checks whether the passed values match the shape of the parameters
        of the corresponding circuits and formats values to 2D list.

        Args:
            circuits: List of circuits to be checked.
            values: Parameter values corresponding to the circuits to be checked.

        Returns:
            A 2D value list if the values match the circuits, or an empty 2D list
            if values is None.

        Raises:
            ValueError: if the number of parameter values doesn't match the number of
                        circuit parameters
            TypeError: if the input values are not a sequence.
        """

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        if values is None:
            for circuit in circuits:
                if circuit.num_parameters != 0:
                    raise ValueError(
                        f"`values` cannot be `None` because circuit <{circuit.name}> has "
                        f"{circuit.num_parameters} free parameters."
                    )
            return [[]]
        else:

            # Support ndarray
            if isinstance(values, np.ndarray):
                values = values.tolist()
            if len(values) > 0 and isinstance(values[0], np.ndarray):
                values = [v.tolist() for v in values]

            if not isinstance(values, Sequence):
                raise TypeError(
                    f"Expected a sequence of numerical parameter values, "
                    f"but got input type {type(values)} instead."
                )

            # ensure 2d
            if len(values) > 0 and not isinstance(values[0], Sequence) or len(values) == 0:
                values = [cast(List[float], values)]

            # we explicitly cast the type here because mypy appears to be unable to understand the
            # above few lines where we ensure that values are 2d
            return cast(Sequence[List[float]], values)

    def _check_qubits_match(self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit) -> None:
        """
        Checks that the number of qubits of 2 circuits matches.
        Args:
            circuit_1: (Parametrized) quantum circuit.
            circuit_2: (Parametrized) quantum circuit.

        Raises:
            ValueError: when ``circuit_1`` and ``circuit_2`` don't have the
            same number of qubits.
        """

        if circuit_1.num_qubits != circuit_2.num_qubits:
            raise ValueError(
                f"The number of qubits for the first circuit ({circuit_1.num_qubits}) "
                f"and second circuit ({circuit_2.num_qubits}) are not the same."
            )

    @abstractmethod
    def create_fidelity_circuit(
        self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit
    ) -> QuantumCircuit:
        """
        Implementation-dependent method to create a fidelity circuit
        from 2 circuit inputs.

        Args:
            circuit_1: (Parametrized) quantum circuit.
            circuit_2: (Parametrized) quantum circuit.

        Returns:
            The fidelity quantum circuit corresponding to ``circuit_1`` and ``circuit_2``.
        """
        raise NotImplementedError

    def _construct_circuits(
        self,
        circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
        circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
    ) -> Sequence[QuantumCircuit]:
        """
        Constructs the list of fidelity circuits to be evaluated.
        These circuits represent the state overlap between pairs of input circuits,
        and their construction depends on the fidelity method implementations.

        Args:
            circuits_1: (Parametrized) quantum circuits.
            circuits_2: (Parametrized) quantum circuits.

        Returns:
            List of constructed fidelity circuits.

        Raises:
            ValueError: if the length of the input circuit lists doesn't match.
        """

        if isinstance(circuits_1, QuantumCircuit):
            circuits_1 = [circuits_1]
        if isinstance(circuits_2, QuantumCircuit):
            circuits_2 = [circuits_2]

        if len(circuits_1) != len(circuits_2):
            raise ValueError(
                f"The length of the first circuit list({len(circuits_1)}) "
                f"and second circuit list ({len(circuits_2)}) is not the same."
            )

        circuits = []
        for circuit_1, circuit_2 in zip(circuits_1, circuits_2):

            # Use the same key for circuits as qiskit.primitives use.
            circuit = self._circuit_cache.get((_circuit_key(circuit_1), _circuit_key(circuit_2)))

            if circuit is not None:
                circuits.append(circuit)
            else:
                self._check_qubits_match(circuit_1, circuit_2)

                # re-parametrize input circuits
                # TODO: make smarter checks to avoid unnecessary re-parametrizations
                parameters_1 = ParameterVector("x", circuit_1.num_parameters)
                parametrized_circuit_1 = circuit_1.assign_parameters(parameters_1)
                parameters_2 = ParameterVector("y", circuit_2.num_parameters)
                parametrized_circuit_2 = circuit_2.assign_parameters(parameters_2)

                circuit = self.create_fidelity_circuit(
                    parametrized_circuit_1, parametrized_circuit_2
                )
                circuits.append(circuit)
                # update cache
                self._circuit_cache[_circuit_key(circuit_1), _circuit_key(circuit_2)] = circuit

        if self._transpiler is not None:
            return self._transpiler.run(circuits, **self._transpiler_options)

        return circuits

    def _construct_value_list(
        self,
        circuits_1: Sequence[QuantumCircuit],
        circuits_2: Sequence[QuantumCircuit],
        values_1: Sequence[float] | Sequence[Sequence[float]] | None = None,
        values_2: Sequence[float] | Sequence[Sequence[float]] | None = None,
    ) -> list[list[float]]:
        """
        Preprocesses input parameter values to match the fidelity
        circuit parametrization, and return in list format.

        Args:
           circuits_1: (Parametrized) quantum circuits preparing the
                        first list of quantum states.
           circuits_2: (Parametrized) quantum circuits preparing the
                        second list of quantum states.
           values_1: Numerical parameters to be bound to the first circuits.
           values_2: Numerical parameters to be bound to the second circuits.

        Returns:
             List of lists of parameter values for fidelity circuit.

        """
        values_1 = self._preprocess_values(circuits_1, values_1)
        values_2 = self._preprocess_values(circuits_2, values_2)
        # now, values_1 and values_2 are explicitly made 2d lists

        values = []
        if len(values_2[0]) == 0:
            values = list(values_1)
        elif len(values_1[0]) == 0:
            values = list(values_2)
        else:
            for val_1, val_2 in zip(values_1, values_2):
                # the `+` operation concatenates the lists
                # and then this new list gets appended to the values list
                values.append(val_1 + val_2)

        # values is guaranteed to be 2d
        return values

    @abstractmethod
    def _run(
        self,
        circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
        circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
        values_1: Sequence[float] | Sequence[Sequence[float]] | None = None,
        values_2: Sequence[float] | Sequence[Sequence[float]] | None = None,
        *,
        shots: int | Sequence[int] | None = None,
    ) -> AlgorithmJob:
        r"""
        Computes the state overlap (fidelity) calculation between two
        (parametrized) circuits (first and second) for a specific set of parameter
        values (first and second).

        Args:
            circuits_1: (Parametrized) quantum circuits preparing :math:`|\psi\rangle`.
            circuits_2: (Parametrized) quantum circuits preparing :math:`|\phi\rangle`.
            values_1: Numerical parameters to be bound to the first set of circuits
            values_2: Numerical parameters to be bound to the second set of circuits.
            shots: Number of shots to be used by the underlying Sampler. If a single integer is
                provided, this number will be used for all circuits. If a sequence of integers is
                provided, they will be used on a per-circuit basis. If not set, the fidelity's default
                number of shots will be used for all circuits, and if that is None (not set) then the
                underlying primitive's default number of shots will be used for all circuits.

        Returns:
            A newly constructed algorithm job instance to get the fidelity result.
        """
        raise NotImplementedError

    def run(
        self,
        circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
        circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
        values_1: Sequence[float] | Sequence[Sequence[float]] | None = None,
        values_2: Sequence[float] | Sequence[Sequence[float]] | None = None,
        *,
        shots: int | Sequence[int] | None = None,
    ) -> AlgorithmJob:
        r"""
        Runs asynchronously the state overlap (fidelity) calculation between two
        (parametrized) circuits (first and second) for a specific set of parameter
        values (first and second). This calculation depends on the particular
        fidelity method implementation.

        Args:
            circuits_1: (Parametrized) quantum circuits preparing :math:`|\psi\rangle`.
            circuits_2: (Parametrized) quantum circuits preparing :math:`|\phi\rangle`.
            values_1: Numerical parameters to be bound to the first set of circuits.
            values_2: Numerical parameters to be bound to the second set of circuits.
            shots: Number of shots to be used by the underlying sampler. If a single integer is
                provided, this number will be used for all circuits. If a sequence of integers is
                provided, they will be used on a per-circuit basis. If none is provided, the
                fidelity's default number of shots will be used for all circuits. If this number is
                also set to None, the underlying primitive's default number of shots will be used
                for all circuits.

        Returns:
            Primitive job for the fidelity calculation.
            The job's result is an instance of :class:`.StateFidelityResult`.
        """
        job = self._run(circuits_1, circuits_2, values_1, values_2, shots=shots)

        job._submit()
        return job

    @staticmethod
    def _truncate_fidelities(fidelities: Sequence[float]) -> Sequence[float]:
        """
        Ensures fidelity result in [0,1].

        Args:
           fidelities: Sequence of raw fidelity results.

        Returns:
             List of truncated fidelities.

        """
        return np.clip(fidelities, 0, 1).tolist()
