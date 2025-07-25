# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Abstract base class of gradient for ``Estimator``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.transpiler.passes import TranslateParameterizedGates

from .estimator_gradient_result import EstimatorGradientResult
from ..utils import (
    DerivativeType,
    GradientCircuit,
    _assign_unique_parameters,
    _make_gradient_parameters,
    _make_gradient_parameter_values,
)
from ...algorithm_job import AlgorithmJob
from ...custom_types import Transpiler
from ...utils.circuit_key import _circuit_key


class BaseEstimatorGradient(ABC):
    """Base class for an ``EstimatorGradient`` to compute the gradients of the expectation value."""

    def __init__(
        self,
        estimator: BaseEstimatorV2,
        precision: float | None = None,
        derivative_type: DerivativeType = DerivativeType.REAL,
        *,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ):
        r"""
        Args:
            estimator: The estimator used to compute the gradients.
            precision: Precision to be used by the underlying Estimator. If provided, this number
                takes precedence over the default precision of the primitive. If None, the default
                precision of the primitive is used.
            derivative_type: The type of derivative. Can be either ``DerivativeType.REAL``
                ``DerivativeType.IMAG``, or ``DerivativeType.COMPLEX``.

                    - ``DerivativeType.REAL`` computes :math:`2 \mathrm{Re}[⟨ψ(ω)|O(θ)|dω ψ(ω)〉]`.
                    - ``DerivativeType.IMAG`` computes :math:`2 \mathrm{Im}[⟨ψ(ω)|O(θ)|dω ψ(ω)〉]`.
                    - ``DerivativeType.COMPLEX`` computes :math:`2 ⟨ψ(ω)|O(θ)|dω ψ(ω)〉`.

                Defaults to ``DerivativeType.REAL``, as this yields e.g. the commonly-used energy
                gradient and this type is the only supported type for function-level schemes like
                finite difference.
            transpiler: An optional object with a `run` method allowing to transpile the circuits
                that are run when using this algorithm. If set to `None`, these won't be
                transpiled.
            transpiler_options: A dictionary of options to be passed to the transpiler's `run`
                method as keyword arguments.
        """
        self._estimator: BaseEstimatorV2 = estimator
        self._precision = precision
        self._derivative_type = derivative_type

        self._transpiler = transpiler
        self._transpiler_options = transpiler_options if transpiler_options is not None else {}

        self._gradient_circuit_cache: dict[
            tuple,
            GradientCircuit,
        ] = {}

    @property
    def derivative_type(self) -> DerivativeType:
        """Return the derivative type (real, imaginary or complex).

        Returns:
            The derivative type.
        """
        return self._derivative_type

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        *,
        precision: float | Sequence[float] | None = None,
    ) -> AlgorithmJob:
        """Run the job of the estimator gradient on the given circuits.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            observables: The list of observables.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of
                the specified parameters. Each sequence of parameters corresponds to a circuit in
                ``circuits``. Defaults to None, which means that the gradients of all parameters in
                each circuit are calculated. None in the sequence means that the gradients of all
                parameters in the corresponding circuit are calculated.
            precision: Precision to be used by the underlying Estimator. If a single float is
                provided, this number will be used for all circuits. If a sequence of floats is
                provided, they will be used on a per-circuit basis. If not set, the gradient's default
                precision will be used for all circuits, and if that is None (not set) then the
                underlying primitive's (default) precision will be used for all circuits.

        Returns:
            The job object of the gradients of the expectation values. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``. The j-th
            element of the i-th result corresponds to the gradient of the i-th circuit with respect
            to the j-th parameter.

        Raises:
            ValueError: Invalid arguments are given.
        """
        if isinstance(circuits, QuantumCircuit):
            # Allow a single circuit to be passed in.
            circuits = (circuits,)
        if isinstance(observables, BaseOperator):
            # Allow a single observable to be passed in.
            observables = (observables,)

        if parameters is None:
            # If parameters is None, we calculate the gradients of all parameters in each circuit.
            parameters = [circuit.parameters for circuit in circuits]
        else:
            # If parameters is not None, we calculate the gradients of the specified parameters.
            # None in parameters means that the gradients of all parameters in the corresponding
            # circuit are calculated.
            parameters = [
                params if params is not None else circuits[i].parameters
                for i, params in enumerate(parameters)
            ]
        # Validate the arguments.
        self._validate_arguments(circuits, observables, parameter_values, parameters)

        if precision is None:
            precision = self.precision  # May still be None

        # Run the job.
        job = AlgorithmJob(
            self._run, circuits, observables, parameter_values, parameters, precision=precision
        )
        job._submit()
        return job

    @abstractmethod
    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        *,
        precision: float | Sequence[float] | None,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        raise NotImplementedError()

    def _preprocess(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        supported_gates: Sequence[str],
    ) -> tuple[Sequence[QuantumCircuit], Sequence[Sequence[float]], Sequence[Sequence[Parameter]]]:
        """Preprocess the gradient. This makes a gradient circuit for each circuit. The gradient
        circuit is a transpiled circuit by using the supported gates, and has unique parameters.
        ``parameter_values`` and ``parameters`` are also updated to match the gradient circuit.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of the specified
                parameters.
            supported_gates: The supported gates used to transpile the circuit.

        Returns:
            The list of gradient circuits, the list of parameter values, and the list of parameters.
            parameter_values and parameters are updated to match the gradient circuit.
        """
        translator = TranslateParameterizedGates(supported_gates)
        g_circuits: list[QuantumCircuit] = []
        g_parameter_values: list[Sequence[float]] = []
        g_parameters: list[Sequence[Parameter]] = []
        for circuit, parameter_value_, parameters_ in zip(circuits, parameter_values, parameters):
            circuit_key = _circuit_key(circuit)
            if circuit_key not in self._gradient_circuit_cache:
                unrolled = translator(circuit)
                self._gradient_circuit_cache[circuit_key] = _assign_unique_parameters(unrolled)
            gradient_circuit = self._gradient_circuit_cache[circuit_key]
            g_circuits.append(gradient_circuit.gradient_circuit)
            g_parameter_values.append(
                _make_gradient_parameter_values(  # type: ignore[arg-type]
                    circuit, gradient_circuit, parameter_value_
                )
            )
            g_parameters.append(_make_gradient_parameters(gradient_circuit, parameters_))
        return g_circuits, g_parameter_values, g_parameters

    def _postprocess(
        self,
        results: EstimatorGradientResult,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
    ) -> EstimatorGradientResult:
        """Postprocess the gradients. This method computes the gradient of the original circuits
        by applying the chain rule to the gradient of the circuits with unique parameters.

        Args:
            results: The computed gradients for the circuits with unique parameters.
            circuits: The list of original circuits submitted for gradient computation.
            parameter_values: The list of parameter values to be bound to the circuits.
            parameters: The sequence of parameters to calculate only the gradients of the specified
                parameters.

        Returns:
            The gradients of the original circuits.
        """
        gradients, metadata = [], []
        for idx, (circuit, parameter_values_, parameters_) in enumerate(
            zip(circuits, parameter_values, parameters)
        ):
            gradient = np.zeros(len(parameters_))
            if (
                "derivative_type" in results.metadata[idx]
                and results.metadata[idx]["derivative_type"] == DerivativeType.COMPLEX
            ):
                # If the derivative type is complex, cast the gradient to complex.
                gradient = gradient.astype("complex")
            gradient_circuit = self._gradient_circuit_cache[_circuit_key(circuit)]
            g_parameters = _make_gradient_parameters(gradient_circuit, parameters_)
            # Make a map from the gradient parameter to the respective index in the gradient.
            g_parameter_indices = {param: i for i, param in enumerate(g_parameters)}
            # Compute the original gradient from the gradient of the gradient circuit
            # by using the chain rule.
            for i, parameter in enumerate(parameters_):
                for g_parameter, coeff in gradient_circuit.parameter_map[parameter]:
                    # Compute the coefficient
                    if isinstance(coeff, ParameterExpression):
                        local_map = {
                            p: parameter_values_[circuit.parameters.data.index(p)]
                            for p in coeff.parameters
                        }
                        bound_coeff = coeff.bind(local_map)
                    else:
                        bound_coeff = coeff
                    # The original gradient is a sum of the gradients of the parameters in the
                    # gradient circuit multiplied by the coefficients.
                    gradient[i] += (
                        float(bound_coeff)
                        * results.gradients[idx][g_parameter_indices[g_parameter]]
                    )
            gradients.append(gradient)
            metadata.append({"parameters": parameters_})
        return EstimatorGradientResult(
            gradients=gradients, metadata=metadata, precision=results.precision
        )

    @staticmethod
    def _validate_arguments(
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
    ) -> None:
        """Validate the arguments of the ``run`` method.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            observables: The list of observables.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of the specified
                parameters.

        Raises:
            ValueError: Invalid arguments are given.
        """
        if len(circuits) != len(parameter_values):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )

        for i, (circuit, parameter_value) in enumerate(zip(circuits, parameter_values)):
            if not circuit.num_parameters:
                raise ValueError(f"The {i}-th circuit is not parameterised.")
            if len(parameter_value) != circuit.num_parameters:
                raise ValueError(
                    f"The number of values ({len(parameter_value)}) does not match "
                    f"the number of parameters ({circuit.num_parameters}) for the {i}-th circuit."
                )

        if len(circuits) != len(observables):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of observables ({len(observables)})."
            )

        for i, (circuit, observable) in enumerate(zip(circuits, observables)):
            if circuit.num_qubits != observable.num_qubits:
                raise ValueError(
                    f"The number of qubits of the {i}-th circuit ({circuit.num_qubits}) does "
                    f"not match the number of qubits of the {i}-th observable "
                    f"({observable.num_qubits})."
                )

        if len(circuits) != len(parameters):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of the list of specified parameters ({len(parameters)})."
            )

        for i, (circuit, parameters_) in enumerate(zip(circuits, parameters)):
            if not set(parameters_).issubset(circuit.parameters):
                raise ValueError(
                    f"The {i}-th parameters contains parameters not present in the "
                    f"{i}-th circuit."
                )

    @property
    def precision(self) -> float | None:
        """Return the precision used by the `run` method of the Estimator primitive. If None,
        the default precision of the primitive is used.

        Returns:
            The default precision.
        """
        return self._precision

    @precision.setter
    def precision(self, precision: float | None):
        """Update the gradient's default precision setting.

        Args:
            precision: The new default precision.
        """

        self._precision = precision
