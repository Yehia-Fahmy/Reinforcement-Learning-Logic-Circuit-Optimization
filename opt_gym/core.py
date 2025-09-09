"""
Core functionalities for the optimization environment

TODO: I should have really implemented this in C++

"""

from dataclasses import dataclass
from bitarray import bitarray
from typing import NewType

LUT_ID = NewType("LUT_ID", int)
LUT_IDS = tuple[LUT_ID, ...]


@dataclass(kw_only=True)
class LUTNetlist:
    """
    A LUTNetlist represents a netlist of LUTs (Look-Up Tables)
    LUT_IDs are used as indexes.
    inputs are given ids: [-num_inputs, 0)
    outputs are given ids: [0, num_outputs)
    intermediates are given ids: [num_outputs, num_outputs + num_intermediates)
    """

    num_inputs: int
    num_intermediates: int
    num_outputs: int

    # Note: Inputs are bit index 0, 1, 2, ..., num_inputs - 1
    # The truth tables are indexed with an integer that represents the input combination
    # For example, index (1010)_2 = 10 corresponds to lut_inputs = [low, high, low, high]
    lut_inputs: list[LUT_IDS]
    lut_truths: list[bitarray]

    def __post_init__(self):
        # Make sure all truths are little-endian
        for t in self.lut_truths:
            if not t.endian == "little":
                raise ValueError("Truth tables must be little-endian")

        # Check the number of LUTs matches num_intermediates + num_outputs
        expected_num_luts = self.max_id()
        if len(self.lut_truths) != expected_num_luts:
            raise ValueError(
                f"Expected {expected_num_luts} LUTs (intermediates + outputs), got {len(self.lut_truths)}"
            )
        if len(self.lut_inputs) != expected_num_luts:
            raise ValueError(
                f"Expected {expected_num_luts} LUT input lists, got {len(self.lut_inputs)}"
            )

        # Check each truth table has the correct length (2^num_inputs)
        for idx, (inputs, truth) in enumerate(zip(self.lut_inputs, self.lut_truths)):
            if len(inputs) == 0:
                assert len(truth) == 1 or len(truth) == 0
                continue  # Empty LUTs are either eliminated or constant.
            expected_len = 1 << len(inputs)
            if len(truth) != expected_len:
                raise ValueError(
                    f"LUT {idx}: Truth table length {len(truth)} does not match expected {expected_len} for {len(inputs)} inputs"
                )

    def max_id(self) -> int:
        """
        Returns the total number of LUTs in the netlist at the start.
        This is the sum of intermediate and output LUTs.
        Measured at the start of the netlist! Including anything constant or eliminated.
        """
        return self.num_intermediates + self.num_outputs

    def is_valid_ID(self, lut_id: LUT_ID) -> bool:
        """
        Check if the given LUT_ID is valid, it discludes inputs.
        Valid IDs are in the range [0, num_outputs + num_intermediates).
        """
        return 0 <= lut_id < self.max_id()

    def is_output_ID(self, lut_id: LUT_ID) -> bool:
        """
        Check if the given LUT_ID is an output ID.
        Output IDs are in the range [0, num_outputs).
        """
        return 0 <= lut_id < self.num_outputs

    def is_intermediate_ID(self, lut_id: LUT_ID) -> bool:
        """
        Check if the given LUT_ID is an intermediate ID.
        Intermediate IDs are in the range [num_outputs, num_outputs + num_intermediates).
        """
        return self.num_outputs <= lut_id < self.num_outputs + self.num_intermediates
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, LUTNetlist):
            return False
        
        # Janky canonicalization: sort inputs within each LUT
        # This ignores truth tables which in principle need to be permuted and compared :)
        self_sorted_lut_inputs = [tuple(sorted(inputs)) for inputs in self.lut_inputs]
        value_sorted_lut_inputs = [tuple(sorted(inputs)) for inputs in value.lut_inputs]

        return (
            self.num_inputs == value.num_inputs
            and self.num_intermediates == value.num_intermediates
            and self.num_outputs == value.num_outputs
            and self_sorted_lut_inputs == value_sorted_lut_inputs
        )
    
    def __hash__(self) -> int:
        # Create a hash based on the netlist structure and a canonical representation of inputs
        hash_components = [
            self.num_inputs,
            self.num_intermediates,
            self.num_outputs,
        ]
        
        # Janky canonicalization for hashing: sort inputs within each LUT
        sorted_lut_inputs = tuple(tuple(sorted(inputs)) for inputs in self.lut_inputs)
        hash_components.append(sorted_lut_inputs)
        
        return hash(tuple(hash_components))


def dont_cares(truth: bitarray, inputs: LUT_IDS) -> LUT_IDS:
    """
    Returns a list of input IDs that are don't-cares in the given truth table.
    """

    dont_care_ids: list[LUT_ID] = []
    for i, input_id in enumerate(inputs):
        bit_mask = 1 << i
        # Check if flipping the bit at position i changes the output
        for j in range(len(truth)):
            if truth[j] != truth[j ^ bit_mask]:
                break
        else:
            # If we didn't break, it means this input is a don't-care
            dont_care_ids.append(input_id)
    return tuple(dont_care_ids)


def fold_into(
    truth_parent: bitarray,
    truth_parent_inputs: LUT_IDS,
    truth_child: bitarray,
    truth_child_inputs: LUT_IDS,
    truth_child_id: LUT_ID,
) -> tuple[bitarray, LUT_IDS]:
    """
    Folds the child LUT into the parent LUT.
    Returns a new parent truth table and a new list of inputs.
    Does not do any simplification, so the returned truth table may
    contain don't-cares or be constant.
    """

    # Combine inputs from parent and child, removing duplicates and the child_id
    new_inputs: list[LUT_ID] = []
    for inp in truth_parent_inputs:
        if inp != truth_child_id:
            new_inputs.append(inp)
    for inp in truth_child_inputs:
        if inp not in new_inputs:
            new_inputs.append(inp)
    num_new_inputs = len(new_inputs)

    # Initialize the new truth table
    new_truth_size = 1 << num_new_inputs
    new_truth = bitarray(new_truth_size, endian="little")

    # Map child and parent inputs to their positions in the new combined input list
    child_input_pos_map = [new_inputs.index(inp) for inp in truth_child_inputs]
    parent_input_pos_map: list[int] = []
    for inp in truth_parent_inputs:
        if inp != truth_child_id:
            parent_input_pos_map.append(new_inputs.index(inp))
        else:
            parent_input_pos_map.append(-1)

    # Populate the new truth table
    for i in range(new_truth_size):
        # Determine the output of the child LUT for the current input combination `i`
        child_index = 0
        for bit_idx, pos in enumerate(child_input_pos_map):
            if (i >> pos) & 1:
                child_index |= 1 << bit_idx
        child_output = truth_child[child_index]

        # Determine the index for the parent LUT
        parent_index = 0
        for bit_idx, inp_id in enumerate(truth_parent_inputs):
            if inp_id == truth_child_id:
                # Use the calculated child output
                if child_output:
                    parent_index |= 1 << bit_idx
            else:
                # Use the bit from the main input combination `i`
                pos = parent_input_pos_map[bit_idx]
                if (i >> pos) & 1:
                    parent_index |= 1 << bit_idx

        # Set the corresponding bit in the new truth table
        new_truth[i] = truth_parent[parent_index]

    return new_truth, tuple(new_inputs)


def eliminate_input(
    truth: bitarray,
    inputs: LUT_IDS,
    input_id: LUT_ID,
    assume_value: int,
) -> tuple[bitarray, LUT_IDS]:
    """
    Eliminates the specified input from the truth table by fixing its value.
    Useful for constant propagation, and don't-care elimination.
    Returns a new truth table and a new list of inputs.
    """
    if input_id not in inputs:
        raise ValueError("Input to eliminate is not in the input list.")
    if assume_value not in (0, 1):
        raise ValueError("Assume value must be 0 or 1.")

    elim_idx = inputs.index(input_id)
    new_inputs = [inp for inp in inputs if inp != input_id]
    num_new_inputs = len(new_inputs)
    new_truth_size = 1 << num_new_inputs
    new_truth = bitarray(new_truth_size, endian="little")

    # Create a mask for bits below the eliminated input's position
    lower_mask = (1 << elim_idx) - 1
    # The value of the bit to be inserted for the eliminated input
    elim_bit_val = assume_value << elim_idx

    for i in range(new_truth_size):
        # Split the new index `i` into lower and upper parts around the eliminated position
        lower_bits = i & lower_mask
        upper_bits = (i & ~lower_mask) << 1
        # Combine them with the assumed value for the eliminated input to get the old index
        old_idx = lower_bits | upper_bits | elim_bit_val
        new_truth[i] = truth[old_idx]

    return new_truth, tuple(new_inputs)


__all__ = [
    "LUTNetlist",
    "LUT_ID",
    "dont_cares",
    "fold_into",
    "eliminate_input",
]
