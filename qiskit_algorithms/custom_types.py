# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Types used by the qiskit-algorithms package."""

from typing import Any, List, Protocol, Union

from qiskit import QuantumCircuit

_Circuits = Union[List[QuantumCircuit], QuantumCircuit]


class Transpiler(Protocol):
    """A Generic type to represent a transpiler."""

    def run(self, circuits: _Circuits, **options: Any) -> _Circuits:
        """Transpile a circuit or a list of quantum circuits."""
        pass
