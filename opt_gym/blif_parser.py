import re
from .core import LUTNetlist, LUT_ID, LUT_IDS
from bitarray import bitarray

_EMPTY = bitarray(endian="little")
_CONST1 = bitarray("1", endian="little")
_CONST0 = bitarray("0", endian="little")


def _convert_sop_to_bitarray(num_inputs: int, sop_lines: list[str]) -> bitarray:
    """
    Converts a Sum-of-Products (SOP) representation to a bitarray truth table.

    The truth table index is formed by the integer value of the inputs (input_0, input_1, ...),
    matching the order in the .names line.

    Args:
        num_inputs: The number of inputs to the LUT (k).
        sop_lines: A list of strings, where each string is a product term from the BLIF file's ON-set.

    Returns:
        A bitarray representing the full truth table.
    """
    if num_inputs == 0:
        # A constant LUT
        # Example:
        # .names outport[19]
        #  0
        # .names outport[20]
        #  0
        if not sop_lines:
            return _CONST0
        elif sop_lines[0] == "1":
            return _CONST1
        else:
            return _CONST0

    num_entries = 1 << num_inputs
    # Initialize truth table to all zeros (the OFF-set)
    truth_table = bitarray(num_entries, endian="little")
    truth_table.setall(0)

    # Process each product term which defines part of the ON-set
    for line in sop_lines:
        parts = line.split()
        if not parts:
            continue
        # Expects format like "1-0 1" or just "1-0". We only care about the input pattern.
        in_pattern = parts[0]

        # Find all minterms (truth table indices) covered by this product term
        for i in range(num_entries):
            # Check if the input combination for this index matches the pattern
            is_match = True
            for j in range(num_inputs):
                input_bit = (i >> j) & 1  # Corresponds to j-th input
                pattern_char = in_pattern[j]
                if (pattern_char == "0" and input_bit == 1) or (
                    pattern_char == "1" and input_bit == 0
                ):
                    is_match = False
                    break
            if is_match:
                truth_table[i] = 1

    return truth_table

comment_pattern = re.compile(r"#[^\n]*")  # Regex to match comments

def parse_blif_to_netlist(file_content: str) -> LUTNetlist:
    """
    Parses BLIF file content and converts it into a LUTNetlist.

    Args:
        file_content: The content of the .blif file as a string.

    Returns:
        The constructed LUTNetlist.
    """
    # Pre-process content: remove comments, join continued lines
    content = file_content.replace("\r", "")  # Remove carriage returns
    content = comment_pattern.sub("", content)  # Remove comments
    content = content.replace("\\\n", " ")  # Join continued lines
    content_lines = content.split("\n")
    content_lines = [line.strip() for line in content_lines]
    content_lines = [line for line in content_lines if line]  # Remove empty lines

    pi_names: list[str] = []
    po_names: list[str] = []

    # Simple parsing for single-line commands
    for line in content_lines:
        parts = line.split()
        if not parts:
            continue
        command = parts[0]
        if command == ".inputs":
            pi_names.extend(parts[1:])
        elif command == ".outputs":
            po_names.extend(parts[1:])

    raw_luts: list[tuple[list[str], str, list[str]]] = []
    i = 0
    while i < len(content_lines):
        line = content_lines[i]
        if line.startswith(".names"):
            parts = line.split()
            blif_lut_inputs = parts[1:-1]
            blif_lut_output = parts[-1]
            sop_lines: list[str] = []
            i += 1
            # collect SOP lines until next command
            while i < len(content_lines) and not content_lines[i].startswith("."):
                term = content_lines[i]
                if term:
                    sop_lines.append(term)
                i += 1
            raw_luts.append((blif_lut_inputs, blif_lut_output, sop_lines))
        else:
            i += 1

    # Identify all signals and categorize them
    primary_inputs = set(pi_names)
    primary_outputs = set(po_names)
    lut_outputs = {lut[1] for lut in raw_luts}

    # An intermediate signal is an output of a LUT that is not a primary output
    intermediates = lut_outputs - primary_outputs

    # Assign unique IDs to all signals for the netlist graph
    num_pis = len(primary_inputs)
    num_pos = len(primary_outputs)
    num_intermediates = len(intermediates)
    num_luts = num_pos + num_intermediates

    sorted_pi_names = sorted(list(primary_inputs))
    sorted_po_names = sorted(list(primary_outputs))
    sorted_intermediate_names = sorted(list(intermediates))

    signal_to_id: dict[str, LUT_ID] = {}
    # Primary inputs get negative IDs
    for i, name in enumerate(sorted_pi_names, start=-num_pis):
        signal_to_id[name] = LUT_ID(i)
    # LUTs driving primary outputs get IDs [0, num_pos - 1]
    for i, name in enumerate(sorted_po_names):
        signal_to_id[name] = LUT_ID(i)
    # LUTs driving intermediate signals get the remaining IDs
    for i, name in enumerate(sorted_intermediate_names, start=num_pos):
        signal_to_id[name] = LUT_ID(i)

    # Build the final LUT data structures
    lut_inputs: list[LUT_IDS] = [() for _ in range(num_luts)]
    lut_truths: list[bitarray] = [_EMPTY for _ in range(num_luts)]

    for lut_in_names, lut_out_name, sop_lines in raw_luts:
        if lut_out_name not in signal_to_id:
            # This could be an unused LUT or a constant driver not otherwise listed.
            # For a valid, flattened netlist, every LUT output should be a PO or an intermediate.
            continue

        lut_id = signal_to_id[lut_out_name]

        # Map input signal names to their corresponding IDs
        input_ids = [signal_to_id[name] for name in lut_in_names]
        lut_inputs[lut_id] = tuple(input_ids)

        # Generate the truth table from the SOP representation
        lut_truths[lut_id] = _convert_sop_to_bitarray(len(input_ids), sop_lines)

    return LUTNetlist(
        num_inputs=num_pis,
        num_intermediates=num_intermediates,
        num_outputs=num_pos,
        lut_inputs=lut_inputs,
        lut_truths=lut_truths,
    )
