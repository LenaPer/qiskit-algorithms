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
Compute-uncompute fidelity interface using primitives
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2, PrimitiveResult, DataBin
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob

from .base_state_fidelity import BaseStateFidelity, StateFidelityPubLike
from .state_fidelity_result import StateFidelityResult
from ..algorithm_job import AlgorithmJob
from ..custom_types import Transpiler
from ..exceptions import AlgorithmError


class ComputeUncompute(BaseStateFidelity):
    r"""
    This class leverages the sampler primitive to calculate the state
    fidelity of two quantum circuits following the compute-uncompute
    method (see [1] for further reference).
    The fidelity can be defined as the state overlap.

    .. math::

            |\langle\psi(x)|\phi(y)\rangle|^2

    where :math:`x` and :math:`y` are optional parametrizations of the
    states :math:`\psi` and :math:`\phi` prepared by the circuits
    ``circuit_1`` and ``circuit_2``, respectively.

    **Reference:**
    [1] Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala,
    A., Chow, J. M., & Gambetta, J. M. (2019). Supervised learning
    with quantum-enhanced feature spaces. Nature, 567(7747), 209-212.
    `arXiv:1804.11326v2 [quant-ph] <https://arxiv.org/pdf/1804.11326.pdf>`_

    """

    def __init__(
        self,
        sampler: BaseSamplerV2,
        shots: int | None = None,
        local: bool = False,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ) -> None:
        r"""
        Args:
            sampler: Sampler primitive instance.
            shots: Number of shots to be used by the underlying sampler.
                The order of priority is: number of shots in ``run`` method > fidelity's
                number of shots > primitive's default number of shots.
                Higher priority setting overrides lower priority setting.
            local: If set to ``True``, the fidelity is averaged over
                single-qubit projectors

                .. math::

                    \hat{O} = \frac{1}{N}\sum_{i=1}^N|0_i\rangle\langle 0_i|,

                instead of the global projector :math:`|0\rangle\langle 0|^{\otimes n}`.
                This coincides with the standard (global) fidelity in the limit of
                the fidelity approaching 1. Might be used to increase the variance
                to improve trainability in algorithms such as :class:`~.time_evolvers.PVQD`.
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are produced within this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.

        Raises:
            ValueError: If the sampler is not an instance of ``BaseSamplerV2``.
        """
        if not isinstance(sampler, BaseSamplerV2):
            raise ValueError(
                f"The sampler should be an instance of BaseSamplerV2, " f"but got {type(sampler)}"
            )
        self._sampler: BaseSamplerV2 = sampler
        self._local = local
        self._shots = shots
        super().__init__(transpiler, transpiler_options)

    def create_fidelity_circuit(
        self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit
    ) -> QuantumCircuit:
        """
        Combines ``circuit_1`` and ``circuit_2`` to create the
        fidelity circuit following the compute-uncompute method.

        Args:
            circuit_1: (Parametrized) quantum circuit.
            circuit_2: (Parametrized) quantum circuit.

        Returns:
            The fidelity quantum circuit corresponding to circuit_1 and circuit_2.
        """
        if len(circuit_1.clbits) > 0:
            circuit_1.remove_final_measurements()
        if len(circuit_2.clbits) > 0:
            circuit_2.remove_final_measurements()

        circuit = circuit_1.compose(circuit_2.inverse())
        circuit.measure_all()
        return circuit

    def _run(
        self,
        pubs_1: Iterable[StateFidelityPubLike],
        pubs_2: Iterable[StateFidelityPubLike],
        shots: int | None,
    ) -> AlgorithmJob[PrimitiveResult[StateFidelityResult]]:
        r"""
        Computes the state overlap (fidelity) calculation between two
        (parametrized) circuits (first and second) for a specific set of parameter
        values (first and second) following the compute-uncompute method.

        Args:
            pubs_1: (Parametrized) quantum circuit and parameter values preparing :math:`|\psi\rangle`.
            pubs_2: (Parametrized) quantum circuit and parameter values preparing :math:`|\phi\rangle`. It must contain as many elements as `pubs_1`.
            shots: Number of shots to be used by the underlying sampler. If None is provided, the
                fidelity's default number of shots will be used for all circuits. If this number is
                also set to None, the underlying primitive's default number of shots will be used
                for all circuits.

        Returns:
            An AlgorithmJob for the fidelity calculation.

        Raises:
            AlgorithmError: If the sampler job is not completed successfully.
        """
        # The priority of number of shots options is as follows:
        # number in `run` method > fidelity's default number of shots >
        # primitive's default number of shots.
        if shots is None:
            shots = self.shots

        coerced_1: list[SamplerPub] = [SamplerPub.coerce(pub) for pub in pubs_1]
        coerced_2: list[SamplerPub] = [SamplerPub.coerce(pub) for pub in pubs_2]

        circuits = self._construct_circuits(
            [pub.circuit for pub in coerced_1],
            [pub.circuit for pub in coerced_2],
        )

        values = [
            self._construct_value_list(
                pub1.circuit,
                pub2.circuit,
                pub1.parameter_values,
                pub2.parameter_values
            ) for (pub1, pub2) in zip(coerced_1, coerced_2)]

        coerced_pubs = [
            SamplerPub.coerce(
                (circuit, value),
                shots
            ) for (circuit, value) in zip(circuits, values)
        ]

        job = self._sampler.run(coerced_pubs)

        return AlgorithmJob(ComputeUncompute._call, job, circuits, self._local)

    @staticmethod
    def _call(
        job: PrimitiveJob, circuits: Sequence[QuantumCircuit], local: bool
    ) -> PrimitiveResult[StateFidelityResult]:
        try:
            pubs_results = job.result()
        except Exception as exc:
            raise AlgorithmError("Sampler job failed!") from exc

        state_fidelity_results: list[StateFidelityResult] = []

        for result, circuit in zip(pubs_results, circuits):
            quasi_dists = [
                {
                    label: value / result.data.meas.num_shots
                    for label, value in result.data.meas.get_int_counts(i).items()
                }
                for i in range(result.data.meas.shape[0])
            ]

            if local:
                raw_fidelities = np.array([
                    ComputeUncompute._get_local_fidelity(prob_dist, circuit.num_qubits)
                    for prob_dist in quasi_dists
                ])
            else:
                raw_fidelities = np.array([
                    ComputeUncompute._get_global_fidelity(prob_dist) for prob_dist in quasi_dists
                ])

            fidelities = ComputeUncompute._truncate_fidelities(raw_fidelities)

            data = DataBin(
                fidelities=fidelities,
                raw_fidelities=raw_fidelities,
                shape=fidelities.shape
            )

            state_fidelity_results.append(
                StateFidelityResult(
                    data=data,
                    metadata=result.metadata,
                )
            )

        return PrimitiveResult(state_fidelity_results)

    @property
    def shots(self) -> int | None:
        """Return the number of shots used by the `run` method of the Sampler primitive. If None,
        the default number of shots of the primitive is used.

        Returns:
            The default number of shots.
        """
        return self._shots

    @shots.setter
    def shots(self, shots: int | None):
        """Update the fidelity's default number of shots setting.

        Args:
            shots: The new default number of shots.
        """

        self._shots = shots

    @staticmethod
    def _get_global_fidelity(probability_distribution: dict[int, float]) -> float:
        """Process the probability distribution of a measurement to determine the
        global fidelity.

        Args:
            probability_distribution: Obtained from the measurement result

        Returns:
            The global fidelity.
        """
        return probability_distribution.get(0, 0)

    @staticmethod
    def _get_local_fidelity(probability_distribution: dict[int, float], num_qubits: int) -> float:
        """Process the probability distribution of a measurement to determine the
        local fidelity by averaging over single-qubit projectors.

        Args:
            probability_distribution: Obtained from the measurement result

        Returns:
            The local fidelity.
        """
        fidelity = 0.0
        for qubit in range(num_qubits):
            for bitstring, prob in probability_distribution.items():
                # Check whether the bit representing the current qubit is 0
                if not bitstring >> qubit & 1:
                    fidelity += prob / num_qubits
        return fidelity
