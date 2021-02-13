from functools import singledispatch
from typing import List

from braket.ir.jaqcd import (
    CZ,
    XY,
    CCNot,
    CNot,
    CPhaseShift,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
    CSwap,
    H,
    I,
    ISwap,
    PhaseShift,
    PSwap,
    Rx,
    Ry,
    Rz,
    S,
    Si,
    Swap,
    T,
    Ti,
    X,
    Y,
    Z,
    PreservedRegionStart,
    PreservedRegionEnd
)


class QuilGate:
    """This a gate object for QuilGate instruction"""

    def __init__(self, name: str, target_qubits: List[int], params=None, modifiers=None):
        # Todo: Input validations
        if not isinstance(name, str):
            raise TypeError("Gate name must be a string")

        self.name = name
        self.target_qubits = target_qubits
        self.params = [] if params is None else params
        self.modifiers = [] if modifiers is None else modifiers

    def controlled(self, control_qubit):
        """Add the CONTROLLED modifier to the gate with the given control qubit."""
        self.modifiers.insert(0, "CONTROLLED")
        self.target_qubits.insert(0, control_qubit)

        return self

    def __str__(self):
        """Converts the QuilGate to Quil representation"""
        return "{}{}{} {}".format(
            " ".join(self.modifiers) + " " if self.modifiers else "",
            self.name,
            "(" + ",".join(str(param) for param in self.params) + ")" if self.params else "",
            " ".join(str(qubit) for qubit in self.target_qubits),
        )


@singledispatch
def translate(ir_instruction) -> QuilGate:
    """ Translates an IR instruction into a corresponding PyQuil Gate

    Args:
        ir_instruction: instruction representation in IR

    Returns:
         (Gate): A Gate object corresponding to ir_instruction
    """
    raise TypeError(f"Instruction not implemented: {ir_instruction}")


@translate.register
def i(ir_instruction: I) -> QuilGate:
    """ Produces the I identity gate.

    I = [[1, 0]
         [0, 1]]
    """
    return QuilGate("I", [ir_instruction.target])

@translate.register
def preserved_region_start(ir_instruction: PreservedRegionStart) -> QuilGate:
    return QuilGate("PreservedRegionStart", [])

@translate.register
def preserved_region_end(ir_instruction: PreservedRegionEnd) -> QuilGate:
    return QuilGate("PreservedRegionEnd", [])

@translate.register
def x(ir_instruction: X) -> QuilGate:
    """ Produces the X ("NOT") gate.

    X = [[0, 1],
         [1, 0]]
    """
    return QuilGate("X", [ir_instruction.target])


@translate.register
def y(ir_instruction: Y) -> QuilGate:
    """ Produces the Y gate.

    Y = [[0, 0 - 1j],
         [0 + 1j, 0]]
    """
    return QuilGate("Y", [ir_instruction.target])


@translate.register
def z(ir_instruction: Z) -> QuilGate:
    """ Produces the Z gate.

    Z = [[1,  0],
         [0, -1]]
    """
    return QuilGate("Z", [ir_instruction.target])


@translate.register
def h(ir_instruction: H) -> QuilGate:
    """ Produces the Hadamard gate.

    H = (1 / sqrt(2)) * [[1,  1],
                         [1, -1]]
    """
    return QuilGate("H", [ir_instruction.target])


@translate.register
def s(ir_instruction: S) -> QuilGate:
    """ Produces the S gate.

    S = [[1, 0],
         [0, 1j]]
    """
    return QuilGate("S", [ir_instruction.target])


@translate.register
def si(ir_instruction: Si) -> QuilGate:
    """ Produces the adjoint of the S gate (i.e. S^{\\dagger}).

    Si = [[1,  0],
          [0, -1j]]
    """
    return QuilGate("S", [ir_instruction.target], modifiers=["DAGGER"])


@translate.register
def t(ir_instruction: T) -> QuilGate:
    """ Produces the T gate.

    T = [[1, 0],
         [0, exp(1j * pi / 4)]]
    """
    return QuilGate("T", [ir_instruction.target])


@translate.register
def ti(ir_instruction: Ti) -> QuilGate:
    """ Produces the adjoint of the T gate (i.e. T^{\\dagger}).

    Ti = [[1, 0],
         [0, exp(-1j * pi / 4)]]
    """
    return QuilGate("T", [ir_instruction.target], modifiers=["DAGGER"])


@translate.register
def rx(ir_instruction: Rx) -> QuilGate:
    """ Produces the RX gate.

    RX(angle) = [[cos(angle / 2), -1j * sin(angle / 2)],
                 [-1j * sin(angle / 2), cos(angle / 2)]]
    """
    return QuilGate("RX", [ir_instruction.target], params=[ir_instruction.angle])


@translate.register
def ry(ir_instruction: Ry) -> QuilGate:
    """ Produces the RY gate.

    RY(angle) = [[cos(angle / 2), -sin(angle / 2)],
                 [sin(angle / 2),  cos(angle / 2)]]
    """
    return QuilGate("RY", [ir_instruction.target], params=[ir_instruction.angle])


@translate.register
def rz(ir_instruction: Rz) -> QuilGate:
    """ Produces the RZ gate.

    RZ(angle) = [[cos(angle / 2) - 1j * sin(angle / 2), 0]
                 [0, cos(angle / 2) + 1j * sin(angle / 2)]]
    """
    return QuilGate("RZ", [ir_instruction.target], params=[ir_instruction.angle])


@translate.register
def phase(ir_instruction: PhaseShift) -> QuilGate:
    """ Produces the PHASE gate.

    PHASE(angle) = [[1, 0],
                    [0, exp(1j * angle)]]
    """
    return QuilGate("PHASE", [ir_instruction.target], params=[ir_instruction.angle])


@translate.register
def cz(ir_instruction: CZ) -> QuilGate:
    """ Produces a controlled-Z gate.

    Z = [[1, 0, 0,  0],
         [0, 1, 0,  0],
         [0, 0, 1,  0],
         [0, 0, 0, -1]]
    """
    return QuilGate("CZ", [ir_instruction.control, ir_instruction.target])


@translate.register
def cnot(ir_instruction: CNot) -> QuilGate:
    """ Produces a controlled-NOT (controlled-X) gate.

    CNOT = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]]
    """
    return QuilGate("CNOT", [ir_instruction.control, ir_instruction.target])


@translate.register
def cnot(ir_instruction: CCNot) -> QuilGate:
    """ Produces a doubly-controlled NOT gate.

    CCNOT = [[1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 1, 0]]
    """
    return QuilGate("CCNOT", ir_instruction.controls + [ir_instruction.target])


@translate.register
def cphase00(ir_instruction: CPhaseShift00) -> QuilGate:
    """ Produces a controlled-phase gate that phases the |00> state.

    CPHASE00(angle) = diag([exp(1j * angle), 1, 1, 1])
    """
    return QuilGate(
        "CPHASE00", [ir_instruction.control, ir_instruction.target], params=[ir_instruction.angle]
    )


@translate.register
def cphase01(ir_instruction: CPhaseShift01) -> QuilGate:
    """ Produces a controlled-phase gate that phases the |01> state.

    CPHASE01(angle) = diag([1, exp(1j * angle), 1, 1])
    """
    return QuilGate(
        "CPHASE01", [ir_instruction.control, ir_instruction.target], params=[ir_instruction.angle]
    )


@translate.register
def cphase10(ir_instruction: CPhaseShift10) -> QuilGate:
    """ Produces a controlled-phase gate that phases the |01> state.

    CPHASE10(angle) = diag([1, 1, exp(1j * angle), 1])
    """
    return QuilGate(
        "CPHASE10", [ir_instruction.control, ir_instruction.target], params=[ir_instruction.angle]
    )


@translate.register
def cphase(ir_instruction: CPhaseShift) -> QuilGate:
    """ Produces a controlled-phase instruction.

    CPHASE(angle) = diag([1, 1, 1, exp(1j * angle)])
    """
    return QuilGate(
        "CPHASE", [ir_instruction.control, ir_instruction.target], params=[ir_instruction.angle]
    )


@translate.register
def swap(ir_instruction: Swap) -> QuilGate:
    """ Produces a SWAP gate.

    SWAP = [[1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]]
    """
    return QuilGate("SWAP", ir_instruction.targets)


@translate.register
def cswap(ir_instruction: CSwap) -> QuilGate:
    """ Produces a controlled-SWAP gate.

    CSWAP = [[1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1]]
    """
    return QuilGate("CSWAP", [ir_instruction.control] + ir_instruction.targets)


@translate.register
def iswap(ir_instruction: ISwap) -> QuilGate:
    """ Produces an ISWAP gate.

    ISWAP = [[1, 0,  0,  0],
             [0, 0,  1j, 0],
             [0, 1j, 0,  0],
             [0, 0,  0,  1]]
    """
    return QuilGate("ISWAP", ir_instruction.targets)


@translate.register
def xy(ir_instruction: XY) -> QuilGate:
    """ Produces an XY gate.

    XY(phi) = [[1,               0,               0, 0],
               [0,      cos(phi/2), 1j * sin(phi/2), 0],
               [0, 1j * sin(phi/2),      cos(phi/2), 0],
               [0,               0,               0, 1]
    """
    return QuilGate("XY", ir_instruction.targets, params=[ir_instruction.angle])


@translate.register
def pswap(ir_instruction: PSwap) -> QuilGate:
    """ Produces a parametrized SWAP gate.

    PSWAP(angle) = [[1, 0,               0,               0],
                    [0, 0,               exp(1j * angle), 0],
                    [0, exp(1j * angle), 0,               0],
                    [0, 0,               0,               1]]
    """
    return QuilGate("PSWAP", ir_instruction.targets, params=[ir_instruction.angle])
