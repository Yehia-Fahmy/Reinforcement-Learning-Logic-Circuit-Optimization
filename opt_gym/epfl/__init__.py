import os
from ..blif_parser import parse_blif_to_netlist
from ..envs import LUTEnv

# Define benchmark paths
EPFL_DIR = os.path.dirname(__file__) # Directory of this file
BENCHMARKS_DIR = os.path.join(EPFL_DIR, "benchmarks-2023.1")
ARITHMETIC = os.path.join(BENCHMARKS_DIR, "arithmetic")
RANDOM_CONTROL = os.path.join(BENCHMARKS_DIR, "random_control")


def list_benches() -> list[tuple[str, str]]:
    """
    List all available EPFL benchmark.
    Example output: [("arithmetic", "adder"), ...]
    """
    benches: list[tuple[str, str]] = []
    for folder in [ARITHMETIC, RANDOM_CONTROL]:
        for _, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".blif"):
                    bench_name = os.path.splitext(file)[0]
                    benches.append((os.path.basename(folder), bench_name))
    return benches


def get_env(bench_base: str, bench_name: str) -> LUTEnv:
    """
    Get the environment for a specific EPFL benchmark.
    """
    if bench_base in ("arithmetic", "random_control"):
        f = os.path.join(BENCHMARKS_DIR, bench_base, f"{bench_name}.blif")
        with open(f, "r") as file:
            file_content = file.read()
        lut_graph = parse_blif_to_netlist(file_content)
    else:
        raise ValueError(f"Unknown benchmark base: {bench_base}")

    return LUTEnv(netlist=lut_graph)
