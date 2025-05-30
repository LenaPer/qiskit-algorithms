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
from typing import Any, Union, Iterable

import numpy as np
from numpy._typing import ArrayLike
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import BindingsArrayLike, PrimitiveResult

from . import StateFidelityResult
from ..algorithm_job import AlgorithmJob
from ..custom_types import Transpiler

# FIXME: should be placed in custom_types.py?
StateFidelityPubLike = Union[
    QuantumCircuit,
    tuple[QuantumCircuit],
    tuple[QuantumCircuit, BindingsArrayLike],
]


class BaseStateFidelity(ABC):
    r"""
    An interface to calculate state fidelities (state overlaps) for a pair of
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
        self._transpiler = transpiler
        self._transpiler_options = transpiler_options if transpiler_options is not None else {}

    @staticmethod
    def _preprocess_values(
        circuit: QuantumCircuit,
        values: ArrayLike | None = None,
    ) -> np.ndarray:
        """
        Checks whether the passed values match the shape of the parameters
        of the corresponding circuits and formats values to 2D array.

        Args:
            circuit: List of circuits to be checked.
            values: Parameter values corresponding to the circuits to be checked.

        Returns:
            A 2D value list if the values match the circuits, or an empty 2D list
            if values is None.

        Raises:
            ValueError: if the number of parameter values doesn't match the number of
                        circuit parameters, or if values is a three or more dimensional array
            TypeError: if the input values are not a sequence.
        """

        if values is None:
            if circuit.num_parameters != 0:
                raise ValueError(
                    f"`values` cannot be `None` because circuit <{circuit.name}> has "
                    f"{circuit.num_parameters} free parameters."
                )
            return np.array([[]])

        values = np.atleast_2d(values)

        if len(values.shape) > 2:
            raise ValueError(
                f"values must be a two or less dimensional array, but its shape is {values.shape}."
            )

        if circuit.num_parameters != values.shape[1]:
            raise ValueError(
                f"Circuit {circuit.name} has {circuit.num_parameters} parameters, but the parameter"
                f" values array represents {values.shape[1]} parameters."
            )

        return values

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
        circuits_1: list[QuantumCircuit],
        circuits_2: list[QuantumCircuit],
    ) -> list[QuantumCircuit]:
        """
        Constructs the fidelity circuits to be evaluated.
        These circuits represent the state overlap between a pair of input circuits,
        and its construction depends on the fidelity method implementation.

        Args:
            circuits_1: (Parametrized) quantum circuits.
            circuits_2: (Parametrized) quantum circuits.

        Returns:
            Constructed fidelity circuit.

        Raises:
            ValueError: if the length of the input circuit lists doesn't match.
        """

        if len(circuits_1) != len(circuits_2):
            raise ValueError(
                f"The length of the first circuit list({len(circuits_1)}) "
                f"and second circuit list ({len(circuits_2)}) is not the same."
            )

        circuits: list[QuantumCircuit] = []

        for circuit_1, circuit_2 in zip(circuits_1, circuits_2):
            self._check_qubits_match(circuit_1, circuit_2)

            # re-parametrize input circuits
            # TODO: make smarter checks to avoid unnecessary re-parametrizations
            parameters_1 = ParameterVector("x", circuit_1.num_parameters)
            parametrized_circuit_1 = circuit_1.assign_parameters(parameters_1)
            parameters_2 = ParameterVector("y", circuit_2.num_parameters)
            parametrized_circuit_2 = circuit_2.assign_parameters(parameters_2)

            circuits.append(self.create_fidelity_circuit(
                parametrized_circuit_1, parametrized_circuit_2
            ))

        if self._transpiler is not None:
            return self._transpiler.run(circuits, **self._transpiler_options)

        return circuits

    def _construct_value_list(
        self,
        circuit_1: QuantumCircuit,
        circuit_2: QuantumCircuit,
        values_1: ArrayLike | None,
        values_2: ArrayLike | None,
    ) -> np.ndarray:
        """
        Preprocesses input parameter values to match the fidelity
        circuit parametrization, and return in list format.

        Args:
           circuit_1: (Parametrized) quantum circuit preparing the first quantum state.
           circuit_2: (Parametrized) quantum circuits preparing the second quantum state.
           values_1: Numerical parameters to be bound to the first circuit.
           values_2: Numerical parameters to be bound to the second circuit.

        Returns:
             List of lists of parameter values for fidelity circuit.
        """
        values_1 = self._preprocess_values(circuit_1, values_1)
        values_2 = self._preprocess_values(circuit_2, values_2)
        # now, values_1 and values_2 are explicitly made 2d arrays

        return np.hstack((values_1, values_2))

    @abstractmethod
    def _run(
        self,
        pubs_1: Iterable[StateFidelityPubLike],
        pubs_2: Iterable[StateFidelityPubLike],
        shots: int | None,
    ) -> AlgorithmJob[PrimitiveResult[StateFidelityResult]]:
        r"""
        Computes the state overlap (fidelity) calculation between two
        (parametrized) circuits (first and second) for a specific set of parameter
        values (first and second).

        Args:
            pubs_1: (Parametrized) quantum circuit and parameter values preparing :math:`|\psi\rangle`.
            pubs_2: (Parametrized) quantum circuit and parameter values preparing :math:`|\phi\rangle`. It must contain as many elements as `pubs_1`.
            shots: Number of shots to be used by the underlying sampler. If None is provided, the
                fidelity's default number of shots will be used for all circuits. If this number is
                also set to None, the underlying primitive's default number of shots will be used
                for all circuits.

        Returns:
            A newly constructed algorithm job instance to get the fidelity result.
        """
        raise NotImplementedError

    def run(
        self,
        pubs_1: Iterable[StateFidelityPubLike],
        pubs_2: Iterable[StateFidelityPubLike],
        shots: int | None = None,
    ) -> AlgorithmJob[PrimitiveResult[StateFidelityResult]]:
        r"""
        Runs asynchronously the state overlap (fidelity) calculation between two
        (parametrized) circuits (first and second) for a specific set of parameter
        values (first and second). This calculation depends on the particular
        fidelity method implementation.

        Args:
            pubs_1: (Parametrized) quantum circuit and parameter values preparing :math:`|\psi\rangle`.
            pubs_2: (Parametrized) quantum circuit and parameter values preparing :math:`|\phi\rangle`. It must contain as many elements as `pubs_1`.
            shots: Number of shots to be used by the underlying sampler. If None is provided, the
                fidelity's default number of shots will be used for all circuits. If this number is
                also set to None, the underlying primitive's default number of shots will be used
                for all circuits.

        Returns:
            Primitive job for the fidelity calculation.
            The job's result is an instance of :class:`.StateFidelityResult`.
        """
        job = self._run(pubs_1, pubs_2, shots)

        job._submit()
        return job

    @staticmethod
    def _truncate_fidelities(fidelities: np.ndarray) -> np.ndarray:
        """
        Ensures fidelity result in [0,1].

        Args:
           fidelities: Sequence of raw fidelity results.

        Returns:
             List of truncated fidelities.

        """
        return np.clip(fidelities, 0, 1)
