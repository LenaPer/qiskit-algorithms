# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2017, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Faster Amplitude Estimation."""

from __future__ import annotations
from typing import cast, Tuple, Any
import warnings
import numpy as np

from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.primitives import BaseSamplerV2, StatevectorSampler
from qiskit_algorithms.exceptions import AlgorithmError

from .amplitude_estimator import AmplitudeEstimator, AmplitudeEstimatorResult
from .estimation_problem import EstimationProblem
from ..custom_types import Transpiler


class FasterAmplitudeEstimation(AmplitudeEstimator):
    """The Faster Amplitude Estimation algorithm.

    The Faster Amplitude Estimation (FAE) [1] algorithm is a variant of Quantum Amplitude
    Estimation (QAE), where the Quantum Phase Estimation (QPE) by an iterative Grover search,
    similar to [2].

    Due to the iterative version of the QPE, this algorithm does not require any additional
    qubits, as the originally proposed QAE [3] and thus the resulting circuits are less complex.

    References:

        [1]: K. Nakaji. Faster Amplitude Estimation, 2020;
            `arXiv:2002.02417 <https://arxiv.org/pdf/2003.02417.pdf>`_
        [2]: D. Grinko et al. Iterative Amplitude Estimation, 2019;
            `arXiv:1912.05559 <http://arxiv.org/abs/1912.05559>`_
        [3]: G. Brassard et al. Quantum Amplitude Amplification and Estimation, 2000;
            `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_

    """

    def __init__(
        self,
        delta: float,
        maxiter: int,
        rescale: bool = True,
        sampler: BaseSamplerV2 | None = None,
        *,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ) -> None:
        r"""
        Args:
            delta: The probability that the true value is outside of the final confidence interval.
            maxiter: The number of iterations, the maximal power of Q is `2 ** (maxiter - 1)`.
            rescale: Whether to rescale the problem passed to `estimate`.
            sampler: A sampler primitive to evaluate the circuits.
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are produced within this algorithm. If set to `None`, these won't be transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.

        """
        super().__init__()
        self._shots = (int(1944 * np.log(2 / delta)), int(972 * np.log(2 / delta)))
        self._rescale = rescale
        self._delta = delta
        self._maxiter = maxiter
        self._num_oracle_calls = 0
        self._sampler = sampler
        self._transpiler = transpiler
        self._transpiler_options = transpiler_options if transpiler_options is not None else {}

    @property
    def sampler(self) -> BaseSamplerV2 | None:
        """Get the sampler primitive.

        Returns:
            The sampler primitive to evaluate the circuits.
        """
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: BaseSamplerV2) -> None:
        """Set sampler primitive.

        Args:
            sampler: A sampler primitive to evaluate the circuits.
        """
        self._sampler = sampler

    def _cos_estimate(self, estimation_problem, k, shots):

        if self._sampler is None:
            warnings.warn(
                "No sampler provided, defaulting to StatevectorSampler from qiskit.primitives"
            )
            self._sampler = StatevectorSampler()

        circuit = self.construct_circuit(estimation_problem, k, measurement=True)

        try:
            pub = (circuit, None, shots)
            job = self._sampler.run([pub])
            result = job.result()[0]
        except Exception as exc:
            raise AlgorithmError("The job was not completed successfully. ") from exc

        circuit_results = getattr(result.data, next(iter(result.data.keys())))
        shots = result.metadata["shots"]

        self._num_oracle_calls += (2 * k + 1) * shots

        # sum over all probabilities where the objective qubits are 1
        prob = 0
        for bit, value in circuit_results.get_counts().items():
            # check if it is a good state
            if estimation_problem.is_good_state(bit):
                prob += value / shots

        cos_estimate = 1 - 2 * prob

        return cos_estimate

    def _chernoff(self, cos, shots) -> list[float]:
        width = np.sqrt(np.log(2 / self._delta) * 12 / shots)
        confint = [np.maximum(-1, cos - width), np.minimum(1, cos + width)]
        return confint

    def construct_circuit(
        self, estimation_problem: EstimationProblem, k: int, measurement: bool = False
    ) -> QuantumCircuit | tuple[QuantumCircuit, list[int]]:
        r"""Construct the circuit :math:`Q^k X |0\rangle>`.

        The A operator is the unitary specifying the QAE problem and Q the associated Grover
        operator.

        Args:
            estimation_problem: The estimation problem for which to construct the circuit.
            k: The power of the Q operator.
            measurement: Boolean flag to indicate if measurements should be included in the
                circuits.

        Returns:
            The circuit :math:`Q^k X |0\rangle`.
        """
        num_qubits = max(
            estimation_problem.state_preparation.num_qubits,
            estimation_problem.grover_operator.num_qubits,
        )
        circuit = QuantumCircuit(num_qubits, name="circuit")

        # add classical register if needed
        if measurement:
            c = ClassicalRegister(len(estimation_problem.objective_qubits))
            circuit.add_register(c)

        # add A operator
        circuit.compose(estimation_problem.state_preparation, inplace=True)

        # add Q^k
        if k != 0:
            circuit.compose(estimation_problem.grover_operator.power(k), inplace=True)

            # add optional measurement
        if measurement:
            # real hardware can currently not handle operations after measurements, which might
            # happen if the circuit gets transpiled, hence we're adding a safeguard-barrier
            circuit.barrier()
            circuit.measure(estimation_problem.objective_qubits, c[:])

        if self._transpiler is not None:
            circuit = self._transpiler.run(circuit, **self._transpiler_options)

        return circuit

    def estimate(self, estimation_problem: EstimationProblem) -> "FasterAmplitudeEstimationResult":
        """Run the amplitude estimation algorithm on provided estimation problem.

        Args:
            estimation_problem: The estimation problem.

        Returns:
            An amplitude estimation results object.

        Raises:
            AlgorithmError: Sampler run error.
        """
        if self._sampler is None:
            warnings.warn(
                "No sampler provided, defaulting to StatevectorSampler from qiskit.primitives"
            )
            self._sampler = StatevectorSampler()

        self._num_oracle_calls = 0

        if self._rescale:
            problem = estimation_problem.rescale(0.25)
        else:
            problem = estimation_problem

        theta_ci = [0, np.arcsin(0.25)]
        first_stage = True
        j_0 = self._maxiter

        theta_cis = [theta_ci]
        num_first_stage_steps = 0
        num_steps = 0

        def cos_estimate(power, shots):
            return self._cos_estimate(problem, power, shots)

        # v is first defined in an if below and referenced after in the else where static analysis
        # e.g. lint, may determine that v might not be defined before used. So this defines it here
        # to avoid lint error. Note the code cannot exit the first stage path until its defined so
        # this value here will never get used in practice.
        v = 0

        for j in range(1, self._maxiter + 1):
            num_steps += 1
            if first_stage:
                num_first_stage_steps += 1
                c = cos_estimate(2 ** (j - 1), self._shots[0])
                chernoff_ci = self._chernoff(c, self._shots[0])
                theta_ci = [np.arccos(x) / (2 ** (j + 1) + 2) for x in chernoff_ci[::-1]]

                if 2 ** (j + 1) * theta_ci[1] >= 3 * np.pi / 8 and j < self._maxiter:
                    j_0 = j
                    v = 2**j * np.sum(theta_ci)
                    first_stage = False
            else:
                cos = cos_estimate(2 ** (j - 1), self._shots[1])
                cos_2 = cos_estimate(2 ** (j - 1) + 2 ** (j_0 - 1), self._shots[1])
                sin = (cos * np.cos(v) - cos_2) / np.sin(v)
                rho = np.arctan2(sin, cos)
                n = int(((2 ** (j + 1) + 2) * theta_ci[1] - rho + np.pi / 3) / (2 * np.pi))

                theta_ci = [
                    (2 * np.pi * n + rho + sign * np.pi / 3) / (2 ** (j + 1) + 2)
                    for sign in [-1, 1]
                ]
            theta_cis.append(theta_ci)

        theta = np.mean(theta_ci)
        rescaling = 4 if self._rescale else 1
        value = (rescaling * np.sin(theta)) ** 2
        value_ci = ((rescaling * np.sin(theta_ci[0])) ** 2, (rescaling * np.sin(theta_ci[1])) ** 2)

        result = FasterAmplitudeEstimationResult()
        result.num_oracle_queries = self._num_oracle_calls
        result.num_steps = num_steps
        result.num_first_state_steps = num_first_stage_steps
        result.success_probability = 1 - (2 * self._maxiter - j_0) * self._delta

        result.estimation = value
        result.estimation_processed = problem.post_processing(value)  # type: ignore[assignment]
        result.confidence_interval = value_ci
        result.confidence_interval_processed = cast(
            Tuple[float, float], (problem.post_processing(x) for x in value_ci)
        )
        result.theta_intervals = theta_cis

        return result


class FasterAmplitudeEstimationResult(AmplitudeEstimatorResult):
    """The result object for the Faster Amplitude Estimation algorithm."""

    def __init__(self) -> None:
        super().__init__()
        self._success_probability: float | None = None
        self._num_steps: int | None = None
        self._num_first_state_steps: int | None = None
        self._theta_intervals: list[list[float]] | None = None

    @property
    def success_probability(self) -> float:
        """Return the success probability of the algorithm."""
        return self._success_probability

    @success_probability.setter
    def success_probability(self, probability: float) -> None:
        """Set the success probability of the algorithm."""
        self._success_probability = probability

    @property
    def num_steps(self) -> int:
        """Return the total number of steps taken in the algorithm."""
        return self._num_steps

    @num_steps.setter
    def num_steps(self, num_steps: int) -> None:
        """Set the total number of steps taken in the algorithm."""
        self._num_steps = num_steps

    @property
    def num_first_state_steps(self) -> int:
        """Return the number of steps taken in the first step of algorithm."""
        return self._num_first_state_steps

    @num_first_state_steps.setter
    def num_first_state_steps(self, num_steps: int) -> None:
        """Set the number of steps taken in the first step of algorithm."""
        self._num_first_state_steps = num_steps

    @property
    def theta_intervals(self) -> list[list[float]]:
        """Return the confidence intervals for the angles in each iteration."""
        return self._theta_intervals

    @theta_intervals.setter
    def theta_intervals(self, value: list[list[float]]) -> None:
        """Set the confidence intervals for the angles in each iteration."""
        self._theta_intervals = value
