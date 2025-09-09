from .core import LUTNetlist, LUT_ID, LUT_IDS, dont_cares, fold_into, eliminate_input
from dataclasses import dataclass, field
from bitarray import bitarray
from collections import deque

_EMPTY = bitarray(endian="little")  # Empty truth table for eliminated LUTs
_ = dont_cares
_ = eliminate_input


@dataclass(kw_only=True)
class MoveDelta:
    """
    Represents the effect of a move (elimination) in the LUT environment.
    """

    eliminated: LUT_ID

    # Changes are changes in truths, inputs, outputs
    new_lut_inputs: dict[LUT_ID, LUT_IDS] = field(default_factory=dict[LUT_ID, LUT_IDS])
    new_lut_truths: dict[LUT_ID, bitarray] = field(
        default_factory=dict[LUT_ID, bitarray]
    )
    new_lut_outputs: dict[LUT_ID, LUT_IDS] = field(
        default_factory=dict[LUT_ID, LUT_IDS]
    )

    def largest_new_k(self) -> int:
        """
        Returns the largest number of inputs (k) in the new LUTs.
        """
        return max(len(inputs) for inputs in self.new_lut_inputs.values())


@dataclass(kw_only=True)
class NetlistInfo:
    """
    Represents information about the netlist.
    Good for computing final rewards
    """

    # Number of primary inputs
    num_inputs: int
    # Number of intermediates at the start of the netlist
    num_intermediates: int
    # Number of outputs
    num_outputs: int
    # Total number of actual LUTs (not counting constant or eliminated LUTs)
    num_luts: int
    # Maximum depth of the netlist (largest number of LUTs from input to output)
    max_depth: int
    lut_counts: dict[int, int]  # k -> count of LUTs with k inputs
    fanout_counts: dict[int, int]
    depth_counts: dict[int, int]  # depth -> count of LUTs at that depth
    depth_list: list[int]  # List of depths for each LUT ID


class LUTEnv:
    """
    A general LUT environment for reinforcement learning.
    """

    def __init__(self, netlist: LUTNetlist):
        self.netlist = netlist

        # Construct outputs for all intermediate and output nodes
        total_nodes = self.netlist.max_id()
        lut_outputs_list: list[list[LUT_ID]] = [[] for _ in range(total_nodes)]
        for lut_id, inputs in enumerate(self.netlist.lut_inputs):
            for input_id in inputs:
                if input_id >= 0:  # Skip primary inputs (negative IDs)
                    lut_outputs_list[input_id].append(LUT_ID(lut_id))

        # Convert to tuples after construction
        self.lut_outputs: list[LUT_IDS] = [
            tuple(outputs) for outputs in lut_outputs_list
        ]

        topo_luts: list[LUT_ID] = []

        # Construct a topological ordering of the LUTs.
        # This is used for depth calculations.
        waiting_count: list[int] = [
            sum(1 for id in inputs if id >= 0) for inputs in self.netlist.lut_inputs
        ]
        ready: deque[LUT_ID] = deque()
        for i in range(total_nodes):
            if waiting_count[i] == 0:
                ready.append(LUT_ID(i))
                topo_luts.append(LUT_ID(i))

        while ready:
            lut_id = ready.popleft()
            for output_id in self.lut_outputs[lut_id]:
                waiting_count[output_id] -= 1
                if waiting_count[output_id] == 0:
                    ready.append(output_id)
                    topo_luts.append(output_id)

        self.topo_luts = tuple(topo_luts)

    def is_move_legal(self, eliminated: LUT_ID) -> bool:
        """
        Check if a move (elimination) is legal.
        Legal is a very simplistic check, it just checks if the number
        is valid and the lut isn't already eliminated.
        It could be that folding will produce an excessively large LUT!
        """
        # Invalid ID:
        if not self.netlist.is_valid_ID(eliminated):
            return False

        if not self.netlist.lut_inputs[eliminated]:
            # If the LUT has no inputs, it is already eliminated.
            return False

        if self.netlist.is_output_ID(eliminated):
            # Outputs can only be folded if they actually output into another LUT.
            # It doesn't save any resources, but it might reduce the depth of the circuit.
            # Otherwise there is nothing to do with them.
            return len(self.lut_outputs[eliminated]) != 0

        # Otherwise its fine.
        return True

    def get_moves(self) -> LUT_IDS:
        """
        Get a list of legal moves (eliminations) for the current state.
        Linear in complexity, so its smarter to call this only once.
        Otherwise you will immediately create an n^2 algorithm.
        Move legality is monotonic, so a move can only ever become illegal after
        some actions.
        """
        return tuple(
            LUT_ID(i)
            for i in range(self.netlist.max_id())
            if self.is_move_legal(LUT_ID(i))
        )

    def observe_move(self, eliminated: LUT_ID) -> MoveDelta:
        """
        Observe a move (elimination) and return the delta
        """
        delta = MoveDelta(eliminated=eliminated)

        if not self.is_move_legal(eliminated):
            raise ValueError(f"Illegal move: {eliminated}")

        # If the eliminated LUT is an output, then we keep it,
        # Otherwise an intermediate that goes nowhere is not useful.
        if not self.netlist.is_output_ID(eliminated):
            delta.new_lut_inputs[eliminated] = tuple()
            delta.new_lut_truths[eliminated] = _EMPTY

        # After folding it will not have any outputs.
        delta.new_lut_outputs[eliminated] = tuple()  # Goes nowhere.
        eliminated_original_truth = self.netlist.lut_truths[eliminated]
        eliminated_original_inputs = self.netlist.lut_inputs[eliminated]
        eliminated_original_outputs = self.lut_outputs[eliminated]

        # For luts with new TTs, we need to check legality and do optimizations, these are managed in this
        # deque.
        changed_luts: deque[LUT_ID] = deque()

        # Point the outputs of all inputs of the eliminated LUT to its outputs
        for input_id in eliminated_original_inputs:
            if input_id >= 0:
                out_list = list(self.lut_outputs[input_id])
                # Remove the eliminated LUT from the outputs of this input
                out_list.remove(eliminated)
                # Add all parents if not already present
                # N^2 but we assume fan out is very low.
                for parent_id in eliminated_original_outputs:
                    if parent_id not in out_list:
                        out_list.append(parent_id)
                delta.new_lut_outputs[input_id] = tuple(out_list)

        # Fold the eliminated LUT's truth into its parents and update their inputs
        for parent_id in eliminated_original_outputs:
            new_parent_truth, new_parent_inputs = fold_into(
                truth_parent=self.netlist.lut_truths[parent_id],
                truth_parent_inputs=self.netlist.lut_inputs[parent_id],
                truth_child=eliminated_original_truth,
                truth_child_inputs=eliminated_original_inputs,
                truth_child_id=eliminated,
            )
            delta.new_lut_inputs[parent_id] = new_parent_inputs
            delta.new_lut_truths[parent_id] = new_parent_truth
            changed_luts.append(parent_id)

        # TODO: Don't care elimination?
        # TODO: Constant folding?

        return delta

    def commit_move(self, delta: MoveDelta):
        # Apply the changes from the delta to the netlist
        for lut_id, new_inputs in delta.new_lut_inputs.items():
            self.netlist.lut_inputs[lut_id] = new_inputs
        for lut_id, new_truth in delta.new_lut_truths.items():
            self.netlist.lut_truths[lut_id] = new_truth
        for lut_id, new_outputs in delta.new_lut_outputs.items():
            self.lut_outputs[lut_id] = new_outputs

    def get_info(self) -> NetlistInfo:
        """
        Get information about the current state of the netlist.
        Returns a NetlistInfo object with various statistics.
        """
        # Get depths for all LUTs
        depth_list = self.get_depths()
        max_depth = max(depth_list) if depth_list else 0

        # Initialize counters
        lut_counts: dict[int, int] = {}
        fanout_counts: dict[int, int] = {}
        depth_counts: dict[int, int] = {}

        # Count LUTs by number of inputs (k) and fanout
        for i in range(self.netlist.max_id()):
            lut_id = LUT_ID(i)
            # Only count active LUTs (those with inputs)
            if self.netlist.lut_inputs[lut_id]:
                # Count by number of inputs
                k = len(self.netlist.lut_inputs[lut_id])
                lut_counts[k] = lut_counts.get(k, 0) + 1

                # Count by fanout
                fanout = len(self.lut_outputs[lut_id])
                if self.netlist.is_output_ID(lut_id):
                    fanout += 1  # Output LUTs have implicit fanout to circuit output
                fanout_counts[fanout] = fanout_counts.get(fanout, 0) + 1

                # Count by depth
                depth = depth_list[lut_id]
                depth_counts[depth] = depth_counts.get(depth, 0) + 1

        return NetlistInfo(
            num_inputs=self.netlist.num_inputs,
            num_intermediates=self.netlist.num_intermediates,
            num_outputs=self.netlist.num_outputs,
            num_luts=sum(lut_counts.values()),
            max_depth=max_depth,
            lut_counts=lut_counts,
            fanout_counts=fanout_counts,
            depth_counts=depth_counts,
            depth_list=depth_list,
        )

    def get_depths(self) -> list[int]:
        """
        Get the depths of all LUTs in the netlist.
        """
        depths: list[int] = [0] * self.netlist.max_id()
        # Topological sort means we can just run through the luts in order :)
        for lut_id in self.topo_luts:
            inputs = self.netlist.lut_inputs[lut_id]
            depths_not_primary = [
                depths[input_id] for input_id in inputs if input_id >= 0
            ]
            depths[lut_id] = max(depths_not_primary, default=0) + 1

        return depths

    def copy(self) -> "LUTEnv":
        """
        Create a deep copy of the current environment.
        This is useful for resetting the environment to a previous state.
        """
        # Create new netlist without validation
        new_netlist = LUTNetlist.__new__(LUTNetlist)
        new_netlist.num_inputs = self.netlist.num_inputs
        new_netlist.num_intermediates = self.netlist.num_intermediates
        new_netlist.num_outputs = self.netlist.num_outputs
        new_netlist.lut_inputs = self.netlist.lut_inputs.copy()
        new_netlist.lut_truths = self.netlist.lut_truths.copy()

        # Create new environment without recalculating everything
        new_env = LUTEnv.__new__(LUTEnv)
        new_env.netlist = new_netlist
        new_env.lut_outputs = self.lut_outputs.copy()
        new_env.topo_luts = self.topo_luts

        return new_env

    def emit_dotfile(self, filename: str):
        """
        Emit the current state of the netlist as a DOT file.
        """
        with open(filename, "w") as f:
            f.write("digraph G {\n")
            f.write("    rankdir=LR;\n")
            f.write("    ranksep=2;\n")
            f.write("    splines=line;\n\n")

            # RANKS will cause layout to crash, the algorithm is not efficient enough.
            # Emit all the input nodes.
            f.write("    // Input nodes\n")
            f.write("    subgraph {\n")
            f.write("        // rank = source;\n")
            f.write("        node [shape=cds];\n")
            for i in range(1, self.netlist.num_inputs + 1):
                if (i - 1) % 10 == 0:
                    f.write("        ")
                f.write(f"i{i};")
                if i % 10 == 0 or i == self.netlist.num_inputs:
                    f.write("\n")
                else:
                    f.write(" ")
            f.write("    }\n\n")

            # Emit all output nodes
            f.write("    // Output nodes\n")
            f.write("    subgraph {\n")
            f.write("        // rank = sink;\n")
            f.write("        node [shape=cds];\n")
            for i in range(self.netlist.num_outputs):
                if i % 10 == 0:
                    f.write("        ")
                f.write(f"o{i};")
                if (i + 1) % 10 == 0 or i == self.netlist.num_outputs - 1:
                    f.write("\n")
                else:
                    f.write(" ")
            f.write("    }\n\n")

            # Group LUTs by their number of inputs (k)
            luts_by_k: dict[int, list[int]] = {}
            total_luts = self.netlist.num_intermediates + self.netlist.num_outputs
            for i in range(total_luts):
                # Consider a LUT active if it's an output LUT or has outputs
                is_output_lut = i >= self.netlist.num_intermediates
                if self.lut_outputs[i] or is_output_lut:
                    k = len(self.netlist.lut_inputs[i])
                    if k > 0:  # Don't process LUTs with no inputs
                        if k not in luts_by_k:
                            luts_by_k[k] = []
                        luts_by_k[k].append(i)

            # Emit LUT nodes, grouped by k
            f.write("    // LUT nodes\n")
            for k in sorted(luts_by_k.keys()):
                f.write("    subgraph {\n")
                f.write(_generate_lut_node(k))
                f.write("\n    ")
                lut_list = luts_by_k[k]
                for idx, lut_id in enumerate(lut_list):
                    if idx % 10 == 0 and idx > 0:
                        f.write("\n    ")
                    f.write(f"l{lut_id};")
                    if idx < len(lut_list) - 1 and (idx + 1) % 10 != 0:
                        f.write(" ")
                f.write("\n\n")
                f.write("    }\n\n")

            # Special handling for constants that are used.
            f.write("    // Constant nodes\n")
            f.write("    subgraph {\n")
            f.write("        node [shape=none];\n")

            const_nodes: list[str] = []
            for i in range(self.netlist.num_intermediates, total_luts):
                # Check if this is an output LUT with no inputs (constant)
                if len(self.netlist.lut_inputs[i]) == 0:
                    # Determine constant value from truth table
                    truth = self.netlist.lut_truths[i]
                    if len(truth) > 0 and truth[0]:
                        const_value = "1'b1"
                    else:
                        const_value = "1'b0"
                    const_nodes.append(f'l{i} [label="{const_value}"]')

            # Emit constant nodes in blocks of 5
            for idx, node in enumerate(const_nodes):
                if idx % 5 == 0:
                    f.write("        ")
                f.write(node + ";")
                if (idx + 1) % 5 == 0 or idx == len(const_nodes) - 1:
                    f.write("\n")
                else:
                    f.write(" ")

            f.write("    }\n\n")

            # Connect the nodes
            f.write("    // Edges\n")
            f.write('    edge [arrowhead="dot"];\n')
            edge_count = 0
            for lut_id in range(total_luts):
                is_output_lut = lut_id >= self.netlist.num_intermediates
                if not (self.lut_outputs[lut_id] or is_output_lut):
                    continue  # Skip disconnected intermediate LUTs

                # Connections from inputs to this LUT
                for port_idx, input_id in enumerate(self.netlist.lut_inputs[lut_id]):
                    if input_id < 0:  # Primary input
                        source_node = f"i{abs(input_id)}"
                        source_port = "e"
                    else:  # Another LUT
                        source_node = f"l{input_id}"
                        source_port = "o:e"

                    dest_node = f"l{lut_id}"
                    dest_port = f"i{port_idx}:w"

                    if edge_count % 4 == 0:
                        f.write("    ")
                    f.write(f"{source_node}:{source_port} -> {dest_node}:{dest_port};")
                    edge_count += 1
                    if edge_count % 4 == 0:
                        f.write("\n")
                    else:
                        f.write(" ")

                # Connections from this LUT to its fanout
                if is_output_lut:
                    # This is a primary output of the circuit
                    output_idx = lut_id - self.netlist.num_intermediates
                    dest_node = f"o{output_idx}"
                    dest_port = "w"

                    if edge_count % 4 == 0:
                        f.write("    ")
                    f.write(f"l{lut_id}:o:e -> {dest_node}:{dest_port};")
                    edge_count += 1
                    if edge_count % 4 == 0:
                        f.write("\n")
                    else:
                        f.write(" ")

            # Add final newline if needed
            if edge_count % 4 != 0:
                f.write("\n")

            f.write("}\n")


def _generate_lut_node(
    k: int, border: int = 2, height: int = 20, width: int = 40
) -> str:
    """
    Generates a Graphviz node definition for a k-input Look-Up Table (LUT).

    This function creates an HTML-like table label for use in Graphviz,
    representing a LUT with a specified number of inputs and configurable
    dimensions and borders.

    Args:
        k: The number of inputs for the LUT (e.g., 6 for a LUT6). Must be a positive integer.
        border: The thickness of the outer border of the table.
        height: The height of each input port cell in pixels.
        width: The width of the central LUT label cell in pixels.

    Returns:
        A string containing the complete, formatted Graphviz node definition.

    Raises:
        ValueError: If k is not a positive integer.
    """
    if k < 1:
        raise ValueError("k (number of inputs) must be a positive integer.")

    # Generate the vertically separated text for the LUT label, e.g., "L<BR/>U<BR/>T<BR/>6"
    lut_text = "L<BR/>U<BR/>T<BR/>" + str(k)

    # Define indentation levels for clean formatting
    indent = "    "  # 4 spaces

    # Start building the node definition string
    # The f-strings and multi-line strings help format it just like the example.
    lines = [
        f"{indent}// LUT{k} Node Definition",
        f"{indent}node [",
        f"{indent*2}shape=plaintext",
        f"{indent*2}label=<",
        f'{indent*3}<TABLE BORDER="{border}" CELLBORDER="0" CELLSPACING="0">',
    ]

    # Handle the special case for a 1-input LUT
    if k == 1:
        row = (
            f"{indent*4}<TR>\n"
            f'{indent*5}<TD PORT="i0" WIDTH="0" HEIGHT="{height}"></TD>\n'
            f'{indent*5}<TD WIDTH="{width}" PORT="o">{lut_text}</TD>\n'
            f"{indent*5}<TD>\\N </TD>\n"
            f"{indent*4}</TR>"
        )
        lines.append(row)
    else:  # Handle LUTs with 2 or more inputs
        # Loop through all k inputs to create the table rows
        for i in range(k):
            if i == 0:
                # The first row is special: it contains the first input port and
                # the main LUT body, which spans most of the rows.
                row = (
                    f"{indent*4}<TR>\n"
                    f'{indent*5}<TD PORT="i0" WIDTH="0" HEIGHT="{height}"></TD>\n'
                    f'{indent*5}<TD ROWSPAN="{k - 1}" WIDTH="{width}" PORT="o">{lut_text}</TD>\n'
                    f"{indent*4}</TR>"
                )
                lines.append(row)
            elif i < k - 1:
                # Middle rows just contain a single input port cell.
                # These are kept on a single line for compactness, matching the example.
                row = f'{indent*4}<TR><TD HEIGHT="{height}" PORT="i{i}"></TD></TR>'
                lines.append(row)
            else:  # This is the last row (i == k - 1)
                # The last row contains the final input port and a special cell
                # to display the node's name ('\N').
                row = (
                    f"{indent*4}<TR>\n"
                    f'{indent*5}<TD HEIGHT="{height}" PORT="i{i}"></TD>\n'
                    f"{indent*5}<TD>\\N </TD>\n"
                    f"{indent*4}</TR>"
                )
                lines.append(row)

    # Close all the open tags
    lines.append(f"{indent*3}</TABLE>")
    lines.append(f"{indent*2}>")
    lines.append(f"{indent}];")

    return "\n".join(lines)
