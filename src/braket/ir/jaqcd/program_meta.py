# Copyright 2019-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from typing import Dict, Tuple

from pydantic import validator, BaseModel

_valid_gates = {
    "CCNot",
    "CNot",
    "CPhaseShift",
    "CPhaseShift00",
    "CPhaseShift01",
    "CPhaseShift10",
    "CSwap",
    "CY",
    "CZ",
    "H",
    "I",
    "ISwap",
    "PhaseShift",
    "PSwap",
    "Rx",
    "Ry",
    "Rz",
    "S",
    "Swap",
    "Si",
    "T",
    "Ti",
    "Unitary",
    "V",
    "Vi",
    "X",
    "XX",
    "XY",
    "Y",
    "YY",
    "Z",
    "ZZ",
}

class GlobalOptions(BaseModel):
    """
    A data structure for representing the global options of programs.

    Attributes:
        gate_time: Dict[str, Tuple[int, str]]: Gate length dictionary

    Examples:
        >>> GlobalOptions(gate_time={"H":(50, "ns")})
    """

    gate_time: Dict[str, Tuple[int, str]]

    @validator('gate_time')
    def check_gate_time(cls, gt):
        for keys in gt:
            assert keys in _valid_gates
            assert gt[keys][1] in ["ns"]
        return gt

