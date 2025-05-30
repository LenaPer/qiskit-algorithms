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
Fidelity result class. It's now an alias for a PubResult. Its `data` field contains:
 - a fidelities attribute, which is an array of truncated fidelity values for each pair of input
   parameter values, ensured to be in [0,1].
 - a raw_fidelities attribute, which is an array of raw fidelity values for each pair of input
   parameter values, which might not be in [0,1] depending on the error mitigation method used.
"""

from qiskit.primitives import PubResult

StateFidelityResult = PubResult
