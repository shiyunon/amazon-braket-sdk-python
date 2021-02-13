import json
import logging
import math
from functools import lru_cache, singledispatch
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from braket.ir.jaqcd import Unitary
from braket.ir.jaqcd.shared_models import (
    DoubleControl,
    DoubleTarget,
    MultiControl,
    MultiTarget,
    SingleControl,
    SingleTarget,
)
from networkx import Graph, from_edgelist

STRING_OBSERVABLES = frozenset(("i", "x", "y", "z", "h"))

log = logging.getLogger()
log.setLevel(logging.INFO)


def get_device_info(arn):
    
    with open("property.json", "r") as f:
        device_info = json.loads(f.read())
    return device_info


def get_unsupported_operation_message(
    backend_arn: str, operation: str, supported_operations: Set[str]
) -> str:
    """
    Get the error message for an unsupported operation.

    Args:
        backend_arn (str): The backend arn for the operation
        operation (str): The unsupported operation name
        supported_operations (Set[str]): Set of lowercase supported operations for the `backend_arn`

    Returns:
        str: Error message for the unsupported operation.
    """
    return (
        f"Backend ARN, {backend_arn}, does not support the operation {operation}. "
        f"Supported operations for this Backend ARN are {list(supported_operations)}."
    )


def get_unsupported_result_type_message(
    backend_arn: str, result_type: str, supported_result_type: Set[str]
) -> str:
    """
    Get the error message for an unsupported result type.

    Args:
        backend_arn (str): The backend arn for the result type
        result_type (str): The unsupported result type name
        supported_result_type (Set[str]): Set of lowercase supported result types for the
            `backend_arn`

    Returns:
        str: Error message for the unsupported result type.
    """
    return (
        f"Backend ARN, {backend_arn}, does not support the result type {result_type}. "
        f"Supported result types for this Backend ARN are {', '.join(supported_result_type)}."
    )


def validate_supported_operation(backend_arn: str, operation: str, supported_operations: Set[str]):
    """
    Validate if the operation is supported.

    Args:
        backend_arn (str): The backen_arn associated with the `supported_operations`
        operation (str): The operation to validate
        supported_operations (Set[str]): The set of lowercase operations to check the
            `operation` against

    Raises:
        ValidationException: If the operation is not supported
    """
    if operation.lower() not in supported_operations:
        error_message = get_unsupported_operation_message(
            backend_arn, operation, supported_operations
        )
        raise NotImplementedError


def validate_distinct_qubits_for_instruction(
    qubits_for_instruction: List[int], instruction_type: str
) -> None:
    """
    Validates whether distinct qubits have been specified for the instruction.

    Args:
        qubits_for_instruction (List[int]): qubit indices referenced by the instruction.
        instruction_type (str): instruction type name to be used for generating the exception
            message.

    Raises:
        ValidationException: If `qubits_for_instruction` contains duplicated qubit indices.
    """
    if len(set(qubits_for_instruction)) != len(qubits_for_instruction):
        raise NotImplementedError

def validate_qubit_count_for_qubits(qubits: Set[int], qubit_count: int) -> None:
    """
    Validates whether the `qubit_count` value is valid for the specified `qubits`.

    Args:
        qubits (Set[int]): Set of qubits referenced in the circuit.
        qubit_count (int): `qubit_count` value specified on task creation.

    Raises:
        ValidationException: If the `qubit_count` value is not greater than the maximum qubit index.
    """
    if qubit_count <= max(qubits):
        raise NotImplementedError


@singledispatch
def validate_instruction_components(instruction) -> None:
    pass


@validate_instruction_components.register
def _validate_unitary_instruction_components(instruction: Unitary) -> None:
    """
    Validates whether the specified unitary instruction is valid.

    Args:
        instruction (Unitary): Unitary instruction to be validated.

    Raises:
        ValidationException: If the matrix specified in the instruction is not unitary or, it is not
            compatible with the specified targets.
    """

    def _is_unitary(matrix: [List[List[List[float]]]]) -> bool:
        dim = len(matrix)
        for row in matrix:
            if len(row) != dim:
                return False

        # Check matrix * conjugate_transpose(matrix) == Identity
        for r in range(len(matrix)):
            for c in range(len(matrix)):
                real = 0
                imaginary = 0
                for i in range(len(matrix)):
                    real += matrix[r][i][0] * matrix[c][i][0] + matrix[r][i][1] * matrix[c][i][1]
                    imaginary += (
                        matrix[r][i][1] * matrix[c][i][0] - matrix[r][i][0] * matrix[c][i][1]
                    )
                # Check distance from identity using same tolerance values as numpy
                if (not math.isclose(imaginary, 0, rel_tol=1e-5, abs_tol=1e-8)) or (
                    not math.isclose(real, 1 if r == c else 0, rel_tol=1e-5, abs_tol=1e-8)
                ):
                    return False
        return True

    if len(instruction.matrix) != 2 ** len(instruction.targets):
        raise NotImplementedError

    if not _is_unitary(instruction.matrix):
        raise NotImplementedError


def validate_supported_result_types(
    backend_arn: str,
    result_types: List[Dict[str, Any]],
    shots: int,
    qubits: Set[int],
    supported_result_types: List[Dict[str, Any]],
) -> None:
    """
    Validate if the result types are supported.

    Args:
        backend_arn (str): The backend_arn associated with the `supported_result_types`
        result_types (List[Dict[str, Any]]): The result types to validate
        shots (int): The shots to run the circuit
        qubits (Set[int]): The qubits used in the circuit
        supported_result_types (List[Dict[str, Any]]): The set of supported result types to check
            the `result_types` against

    Raises:
        ValidationException: If the result type is not supported
    """
    supported_rt_names = [srt["name"].lower() for srt in supported_result_types]
    qubit_count = len(qubits)
    qubit_observable_mapping = {}
    qubit_target_mapping = {}
    for rt in result_types:
        rt_name = rt["type"]
        rt_observable = frozenset(
            map(lambda x: "hermitian" if isinstance(x, list) else x, rt.get("observable", []))
        )
        rt_targets = rt.get("targets")
        if rt_targets and frozenset(rt_targets) - qubits:
            raise NotImplementedError
        try:
            supported_rt_index = supported_rt_names.index(rt_name)
            supported_rt_dict = supported_result_types[supported_rt_index]
            minShots, maxShots = (
                get_with_default(supported_rt_dict, "minShots", float("-inf")),
                get_with_default(supported_rt_dict, "maxShots", float("inf")),
            )
            supported_rt_observables = frozenset(
                [obs.lower() for obs in get_with_default(supported_rt_dict, "observables", [])]
            )
            rt_observable_diff = rt_observable - supported_rt_observables
            if shots < minShots or shots > maxShots:
                raise NotImplementedError
            if rt_observable_diff:
                raise NotImplementedError
            if rt_name == "amplitude":
                if not all(len(state) == qubit_count for state in rt["states"]):
                    raise NotImplementedError
            _validate_observables(
                rt_name,
                rt.get("observable"),
                rt_targets,
                qubit_observable_mapping,
                qubit_target_mapping,
                qubits,
            )
        except ValueError:
            error_message = get_unsupported_result_type_message(
                backend_arn, rt_name, supported_rt_names
            )
            raise NotImplementedError


def _validate_observables(
    rt_name: str,
    rt_observable: Optional[List[Union[str, List]]],
    rt_targets: Optional[List[int]],
    qubit_observable_mapping: Dict[int, Any],
    qubit_target_mapping: Dict[int, List[int]],
    used_qubits: Set[int],
) -> None:
    """
    Ensure that no different observables act on overlapping targets and that observables are
    compatible with supplied target values.

    Args:
        rt_name (str): Result type name
        rt_observable (Optional[List[Union[str, List]]]): result type observable
        rt_targets (Optional[List[int]]): result type targets
        qubit_observable_mapping (Dict[int, Any]): qubit target to observable mapping
        used_qubits (Set[int]): set of used qubits in circuit

    Raises:
        ValidationException: if any different observables act on the same target or the supplied
            observable is incompatible with the size of its target
    """
    if rt_name == "probability":
        observable = ["z"]
    elif rt_observable:
        _validate_observable_targets(rt_observable, rt_targets)
        _validate_observable_components(rt_observable)
        observable = rt_observable
    else:
        return

    targets = list(rt_targets if rt_targets else used_qubits)
    index_factor_mapping = _index_factor_mapping(list(observable)) if len(observable) > 1 else None
    for i in range(len(targets)):
        target = targets[i]
        factor = [index_factor_mapping[i][0]] if index_factor_mapping else observable
        current_observable = qubit_observable_mapping.get(target)
        if current_observable and current_observable != factor:
            raise NotImplementedError
        qubit_observable_mapping[target] = factor
        if _qubit_count_for_observable(factor) > 1:
            current_target = qubit_target_mapping.get(target)
            new_targets = (
                targets[index_factor_mapping[i][1][0] : index_factor_mapping[i][1][1]]
                if index_factor_mapping
                else targets
            )
            if current_target and current_target != new_targets:
                raise NotImplementedError
            qubit_target_mapping[target] = new_targets


def _index_factor_mapping(
    factors,
) -> Dict[int, Tuple[List[Union[str, List[List[List[float]]]]], Tuple[int, int]]]:
    obj_dict = {}
    i = 0
    total = _qubit_count_for_single_observable(factors[0])
    while factors:
        if i >= total:
            factors.pop(0)
            if factors:
                total += _qubit_count_for_single_observable(factors[0])
        if factors:
            obj_dict[i] = (
                factors[0],
                (total - _qubit_count_for_single_observable(factors[0]), total),
            )
        i += 1
    return obj_dict


def _validate_observable_components(observable: List[Union[str, List[List[List[float]]]]]) -> None:
    """Validates the components of the observable individually"""
    if len(observable) == 1:
        single_observable = observable[0]
        # Only add additional validations for hermitian observable
        if isinstance(single_observable, list):
            _validate_hermitian_observable(single_observable)
    else:
        for factor in observable:
            _validate_observable_components([factor])


def _validate_hermitian_observable(observable: [List[List[List[float]]]]) -> None:
    """
    Validates whether the specified hermitian observable is valid.

    Args:
        observable ([List[List[List[float]]]]): Hermitian observable to be validated.

    Raises:
        ValidationException: If the matrix specified in the observable is not hermitian or, it
            is not compatible with the specified targets.
    """

    def _is_hermitian(matrix: [List[List[List[float]]]]) -> bool:
        dim = len(matrix)
        for row in matrix:
            if len(row) != dim:
                return False

        # Check matrix == conjugate_transpose(matrix)
        for r in range(len(matrix)):
            for c in range(len(matrix)):
                if not math.isclose(
                    matrix[r][c][0], matrix[c][r][0], rel_tol=1e-5, abs_tol=1e-8
                ) or not math.isclose(
                    matrix[r][c][1], -matrix[c][r][1], rel_tol=1e-5, abs_tol=1e-8
                ):
                    return False
        return True

    if not _is_hermitian(observable):
        raise NotImplementedError

def _validate_observable_targets(
    observable: List[Union[str, List[List[List[float]]]]], targets: List[int]
) -> None:
    """
    Validates whether the specified observables are compatible with the supplied target values.

    Args:
        observable (List[Union[str, List[List[List[float]]]]]): The observable to validate. Can be
            either a single observable (that is, either in the set {"i", "h", "x", "y", "z"} or a
            Hermitian matrix specified with a list of lists) or, for a tensor product observable,
            a list of single observables.
        targets (List[int]): List of indices that the supplied observables will be applied to.

    Raises:
        ValidationException: If targets is empty, and the observable applies to more than 1 qubit
            or, if len(targets) does not match the number of qubits that the observable applies to.
    """
    if not targets:
        if _qubit_count_for_observable(observable) != 1:
            raise NotImplementedError
    else:
        if _qubit_count_for_observable(observable) != len(set(targets)):
            raise NotImplementedError


def _qubit_count_for_single_observable(
    single_observable: Union[str, List[List[List[float]]]]
) -> int:
    return (
        1
        if isinstance(single_observable, str) and single_observable in STRING_OBSERVABLES
        else int(math.log2(len(single_observable[0])))
    )


def _qubit_count_for_observable(observables: List[Union[str, List[List[List[float]]]]]) -> int:
    if len(observables) == 1:
        return _qubit_count_for_single_observable(observables[0])
    return sum(_qubit_count_for_observable([factor]) for factor in observables)


def add_qubits(instruction, qubits: List[int]) -> None:
    """
    Add qubits from instruction to qubit list.

    Args:
        instruction: instruction in Program.instructions to process
        qubits (List[int]): the qubit list to add the instruction's qubits
    """
    _add_control_qubits(instruction, qubits)
    _add_target_qubits(instruction, qubits)


@singledispatch
def _add_target_qubits(instruction, qubits: List[int]) -> None:
    raise NotImplementedError


@_add_target_qubits.register
def _(instruction: SingleTarget, qubits: List[int]) -> None:
    qubits.append(instruction.target)


@_add_target_qubits.register
def _(instruction: DoubleTarget, qubits: List[int]) -> None:
    qubits.extend(instruction.targets)


@_add_target_qubits.register
def _(instruction: MultiTarget, qubits: List[int]) -> None:
    qubits.extend(instruction.targets)


@singledispatch
def _add_control_qubits(instruction, qubits: List[int]) -> None:
    # Do nothing if there are no control qubits
    pass


@_add_control_qubits.register
def _(instruction: SingleControl, qubits: List[int]) -> None:
    qubits.append(instruction.control)


@_add_control_qubits.register
def _(instruction: DoubleControl, qubits: List[int]) -> None:
    qubits.extend(instruction.controls)


@_add_control_qubits.register
def _(instruction: MultiControl, qubits: List[int]) -> None:
    qubits.extend(instruction.controls)


def validate_contiguous_qubits(qubits: Set[int]) -> None:
    if max(qubits) >= len(qubits):
        raise NotImplementedError


def connections_for_qubits(qubits: List[int]) -> FrozenSet[Tuple[int, int]]:
    """Gets the qubit pairs used by a list of qubits.

    Pairings are made in the same order as the qubit list; the nth qubit is paired with the n+1th,
    the n+1th with the n+2th, and so on. Note that this means that the order of the list matters.

    Each of the returned pairs is a sorted tuple with the smaller qubit index first.

    Examples:
        >>> connections_for_qubits([3, 4, 1])
        frozenset({(3, 4), (1, 4)})

    Args:
        qubits (List[int]): A list of qubits

    Returns:
        FrozenSet[Tuple[int, int]]: The pairs of qubits in the given qubit list.
    """
    return frozenset(tuple(sorted((qubits[i], qubits[i + 1]))) for i in range(len(qubits) - 1))


@lru_cache()
def get_device_supported_operations(
    device_arn: str, action_type="braket.ir.jaqcd.program"
) -> FrozenSet[str]:
    """
    Get the supported operations for the device.
    The result is cached in-memory for the lifetime of the process.

    Args:
        device_arn (str): The backend arn of the simulator call to call Braket API with.
        action_type (str): type of the action to decide the supported actions for the device

    Returns:
        Set[str]: Set of operators the gate based QPU supports in all lowercase.
    """

    device_info = get_device_info(device_arn)

    supported_operations = device_info["action"][action_type].get(
        "supportedOperations", []
    )
    return frozenset(supported_operations)


@lru_cache()
def get_device_supported_result_types(
    device_arn: str, action_type="braket.ir.jaqcd.program"
) -> List[Dict[str, Any]]:
    """
    Get the supported result_types for the device.
    The result is cached in-memory for the lifetime of the process.

    Args:
        device_arn (str): The backend arn of the device
        action_type (bool): type of action you need to get supported result types for

    Returns:
        List[Dict[str, Any]]: list of supported result type dicts
    """
    device_info = get_device_info(device_arn)
    return device_info["action"][action_type].get(
        "supportedResultTypes", []
    )


def get_with_default(obj: dict, key: str, default: Any) -> Any:
    """
    Get the value from a dict. If the key does not exist or the value is None then return
    the `default`.

    Args:
        obj (dict): Dictionary to get the value from
        key (str): Key of the value to get
        default (Any): Value to return if the `key` does not exist in `obj` or the value is None.

    Returns:
        Any: value of the `key` in the `obj` or `default` if the key does not exist in obj
            or the value is None.
    """
    value = obj.get(key)
    if value is None:
        return default
    else:
        return value


@lru_cache()
def device_topology(arn: str) -> Graph:
    """Returns the topology of the device with the given ARN.

    The result is cached in-memory for the lifetime of the process.

    Args:
        arn (str): The ARN of the device

    Returns:
        Graph: The topology of the device as a NetworkX Graph object
    """
    device_info = get_device_info(arn)
    raw_graph = device_info["paradigm"]["connectivity"][
        "connectivityGraph"
    ]
    edges = []
    for item in raw_graph.items():
        i = item[0]
        edges.extend([(int(i), int(j)) for j in item[1]])
    return from_edgelist(edges)


def validate_topology(qubits: Set[int], connections: Set[Tuple[int, int]], device_topology: Graph):
    """Validates that the given qubits and connections are in the given device topology.

    Args:
        qubits (List[int]): The qubits to validate
        connections (Set[Tuple[int, int]]): The qubit connections to validate
        device_topology (Graph): The device topology to validate against

    Raises:
        ValidationException: If any of the given qubits or connections
            are not found in the given topology.
    """
    device_qubits = frozenset(device_topology.nodes)
    if not device_qubits.issuperset(qubits):
        missing_qubits = qubits - device_qubits
        raise NotImplementedError

    connections_ordered = frozenset(tuple(sorted(connection)) for connection in connections)
    device_connections = frozenset(tuple(sorted(edge)) for edge in device_topology.edges)
    if not device_connections.issuperset(connections_ordered):
        missing_connections = connections_ordered - device_connections
        raise NotImplementedError


def validate_supports_disabled_qubit_rewiring(
    device_arn: str, disable_qubit_rewiring: bool
) -> None:
    """Ensure that disabled qubit rewiring is supported by the device if it is requested.

    Args:
        device_arn (str): The ARN of the device
        disable_qubit_rewiring (bool): Whether disabled qubit rewiring is requested

    Raises:
        ValidationException: If disabled qubit rewiring is requested but unsupported by the device.
    """
    if disable_qubit_rewiring:

        supported = get_device_info(device_arn)["action"][
            "braket.ir.jaqcd.program"
        ].get("disabledQubitRewiringSupported")
        # If disabled qubit rewiring isn't explicitly unsupported,
        # the `disableQubitRewiring` can just be ignored, e.g. SV1
        if supported is not None and not supported:
            raise NotImplementedError
