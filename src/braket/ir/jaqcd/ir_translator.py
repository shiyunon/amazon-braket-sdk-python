import boto3
from braket.circuits import Circuit
from braket.device_schema import GateModelParameters

from enum import Enum
from typing import Any, Dict

from braket.ir.jaqcd import Program
from .utils import (
    add_qubits,
    connections_for_qubits,
    device_topology,
    get_device_supported_operations,
    get_device_supported_result_types,
    validate_distinct_qubits_for_instruction,
    validate_supported_operation,
    validate_supported_result_types,
    validate_topology,
)

from .quil_gate import translate

def ir_to_quil(
    ir_program: Program, device_arn: str, gate_model_parameters: Dict[str, Any], shots: int
) -> str:
    """ Converts Braket IR circuit to Quil.

    Spec: https://github.com/rigetti/quil/blob/master/spec/Quil.md

    Args:
        ir_program (Program): Program object of the IR
        device_arn (str): The ARN of the device the circuit will run on
        gate_model_parameters (Dict[str, Any]): Additional parameters needed to run the circuit
        shots (int): The number of shots to run this program

    Returns:
        (str): Quil representation of the given IR.

    Raises:
        ValidationException: If the instruction is not implemented.
    """
    qubit_count = gate_model_parameters["qubitCount"]
    gate_time_dict = gate_model_parameters['gateLengthParameter']
    supported_gates = get_device_supported_operations(device_arn)
    
    for gate in gate_time_dict:
        validate_supported_operation(device_arn, gate, supported_gates)

    gate_time_options = (["PRAGMA gate_time {} \"{}\"".format(gate_time,
                                                              gate_time_dict[gate_time])
                                                              for gate_time in gate_time_dict])
    gate_time_options_str = '\n'.join(gate_time_options)

    rewiring_strategy = (
        _RewiringStrategy.NAIVE
        if gate_model_parameters.get("disableQubitRewiring")
        else _RewiringStrategy.PARTIAL
    )

    ir_instructions = ir_program.instructions
    if ir_program.basis_rotation_instructions:
        ir_instructions.extend(ir_program.basis_rotation_instructions)


    translated_gate_list = []
    qubits = set()
    connections = set()
    in_region = False
    for instr in ir_instructions:
        if instr.type.value == "preserved_region_start":
            if in_region:
                raise ValueError
            in_region = True
            translated_gate_list.append("PRAGMA PRESERVE_BLOCK")

        elif instr.type.value == "preserved_region_end":
            if not in_region:
                raise ValueError
            in_region = False
            translated_gate_list.append("PRAGMA END_PRESERVE_BLOCK")

        else:
            validate_supported_operation(device_arn, instr.type.value, supported_gates)
            translated_gate_list.append(translate(instr))
            qubits_for_instruction = []
            add_qubits(instr, qubits_for_instruction)
            validate_distinct_qubits_for_instruction(qubits_for_instruction, instr.type.value)
            qubits.update(qubits_for_instruction)
            if rewiring_strategy is _RewiringStrategy.NAIVE:
                connections.update(connections_for_qubits(qubits_for_instruction))

    used_qubit_count = len(qubits)
    if used_qubit_count > qubit_count:
        raise ValueError(
            f"Insufficient qubits specified; expected: {qubit_count}, actual: {used_qubit_count}"
        )

    if ir_program.results:
        supported_result_types = get_device_supported_result_types(device_arn)
        validate_supported_result_types(
            device_arn, ir_program.dict()["results"], shots, qubits, supported_result_types
        )
    qubit_tuple = tuple(qubits)

    if rewiring_strategy is _RewiringStrategy.NAIVE:
        validate_topology(qubits, connections, device_topology(device_arn))
    rewiring = f'PRAGMA INITIAL_REWIRING "{rewiring_strategy}"'
    register = f"DECLARE ro BIT[{used_qubit_count}]"
    reset = f"RESET"
    instructions = "\n".join((str(gate) for gate in translated_gate_list))
    measurements = "\n".join((f"MEASURE {qubit_tuple[n]} ro[{n}]" for n in range(used_qubit_count)))
    return "\n".join((rewiring, gate_time_options_str, register, reset, instructions, measurements))


class _RewiringStrategy(str, Enum):
    NAIVE = "NAIVE"
    PARTIAL = "PARTIAL"
