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
Fidelity result class
"""

from __future__ import annotations

from collections.abc import Sequence, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StateFidelityResult:
    """This class stores the result of StateFidelity computations."""

    fidelities: Sequence[float]
    """List of truncated fidelity values for each pair of input circuits, ensured to be in [0,1]."""
    raw_fidelities: Sequence[float]
    """List of raw fidelity values for each pair of input circuits, which might not be in [0,1]
    depending on the error mitigation method used."""
    metadata: Sequence[Mapping[str, Any]]
    """Additional information about the fidelity calculation."""
    shots: int | Sequence[int]
    """Primitive number of shots options for the execution of the fidelity job."""
