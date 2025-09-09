import opt_gym.epfl as epfl
import opt_gym.envs as envs

_ = envs

with open("epfl_benchmarks.log", "w") as f:
    print("All benchmarks in EPFL:")
    for base, name in epfl.list_benches():
        print(f"base: {base}, name: {name}")
        f.write(f"\nBenchmark: {base, name}\n")
        env = epfl.get_env(base, name)
        info = env.get_info()
        f.write(f"num_inputs: {info.num_inputs}\n")
        f.write(f"num_intermediates: {info.num_intermediates}\n")
        f.write(f"num_outputs: {info.num_outputs}\n")
        f.write(f"num_luts: {info.num_luts}\n")
        f.write(f"max_depth: {info.max_depth}\n")
        f.write(f"lut_counts: {info.lut_counts}\n")
        f.write(f"fanout_counts: {info.fanout_counts}\n")
        f.write(f"depth_counts: {info.depth_counts}\n")
        f.write(f"depth_list: {info.depth_list}\n")
    print("All benchmarks in EPFL written to epfl_benchmarks.log")
