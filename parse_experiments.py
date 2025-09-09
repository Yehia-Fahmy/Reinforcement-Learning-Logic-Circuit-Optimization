#!/usr/bin/env python3

import re
import sys

def parse_random_agent_experiments(log_file_path):
    """Parse the random_agent.log file to extract experiment data."""
    
    experiments = []
    current_experiment = None
    experiment_num = 0
    
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Check for start of new experiment (Benchmark line)
                if line.startswith("Benchmark:"):
                    if current_experiment:
                        experiments.append(current_experiment)
                    
                    experiment_num += 1
                    current_experiment = {
                        'experiment_num': experiment_num,
                        'benchmark': line,
                        'initial_num_luts': None,
                        'initial_max_depth': None,
                        'final_num_luts': None,
                        'final_max_depth': None
                    }
                
                # Check for initial values
                elif "Initial Netlist Info:" in line:
                    # Look for num_luts and max_depth in the next few lines
                    pass
                
                elif "num_luts:" in line and current_experiment and current_experiment['initial_num_luts'] is None:
                    match = re.search(r'num_luts: (\d+)', line)
                    if match:
                        current_experiment['initial_num_luts'] = int(match.group(1))
                
                elif "max_depth:" in line and current_experiment and current_experiment['initial_max_depth'] is None:
                    match = re.search(r'max_depth: (\d+)', line)
                    if match:
                        current_experiment['initial_max_depth'] = int(match.group(1))
                
                # Check for final values
                elif "Final Netlist Info:" in line:
                    # Look for num_luts and max_depth in the next few lines
                    pass
                
                elif "num_luts:" in line and current_experiment and current_experiment['initial_num_luts'] is not None and current_experiment['final_num_luts'] is None:
                    match = re.search(r'num_luts: (\d+)', line)
                    if match:
                        current_experiment['final_num_luts'] = int(match.group(1))
                
                elif "max_depth:" in line and current_experiment and current_experiment['initial_max_depth'] is not None and current_experiment['final_max_depth'] is None:
                    match = re.search(r'max_depth: (\d+)', line)
                    if match:
                        current_experiment['final_max_depth'] = int(match.group(1))
        
        # Add the last experiment if exists
        if current_experiment:
            experiments.append(current_experiment)
            
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return experiments

def parse_reinforce_experiments(log_file_path):
    """Parse the reinforce.log file to extract experiment data."""
    
    experiments = []
    
    try:
        with open(log_file_path, 'r') as file:
            content = file.read()
            
            # Split by experiments (each experiment is separated by blank lines)
            experiment_blocks = content.strip().split('\n\n')
            
            for i, block in enumerate(experiment_blocks):
                if not block.strip():
                    continue
                    
                lines = block.strip().split('\n')
                if len(lines) < 3:
                    continue
                
                # Parse benchmark name
                benchmark_line = lines[0]
                if not benchmark_line.startswith("Benchmark:"):
                    continue
                
                # Parse initial values
                initial_line = lines[1]
                initial_match = re.search(r'Initial: (\d+) LUTs, (\d+) depth', initial_line)
                if not initial_match:
                    continue
                
                # Parse final values
                final_line = lines[2]
                final_match = re.search(r'Final: (\d+) LUTs, (\d+) depth', final_line)
                if not final_match:
                    continue
                
                experiment = {
                    'experiment_num': i + 1,
                    'benchmark': benchmark_line,
                    'initial_num_luts': int(initial_match.group(1)),
                    'initial_max_depth': int(initial_match.group(2)),
                    'final_num_luts': int(final_match.group(1)),
                    'final_max_depth': int(final_match.group(2))
                }
                
                # Parse additional info if available
                if len(lines) > 3:
                    for line in lines[3:]:
                        if "Total episodes:" in line:
                            episodes_match = re.search(r'Total episodes: (\d+)', line)
                            if episodes_match:
                                experiment['total_episodes'] = int(episodes_match.group(1))
                        elif "Total time:" in line:
                            time_match = re.search(r'Total time: ([\d.]+)s', line)
                            if time_match:
                                experiment['total_time'] = float(time_match.group(1))
                
                experiments.append(experiment)
            
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return experiments

def normalize_benchmark_name(benchmark_line):
    """Normalize benchmark names for comparison."""
    # Extract the benchmark name from the line
    if "Benchmark:" in benchmark_line:
        benchmark = benchmark_line.split("Benchmark:")[1].strip()
        # Remove seed information and normalize
        benchmark = re.sub(r', Seed: \d+', '', benchmark)
        # Remove extra commas and spaces
        benchmark = re.sub(r'\s*,\s*', ' ', benchmark)
        benchmark = benchmark.lower().strip()
        return benchmark
    return benchmark_line.lower().strip()

def find_matching_experiments(random_experiments, reinforce_experiments):
    """Find matching experiments between the two log files."""
    matches = []
    used_reinforce_indices = set()
    
    for random_exp in random_experiments:
        random_benchmark = normalize_benchmark_name(random_exp['benchmark'])
        
        for i, reinforce_exp in enumerate(reinforce_experiments):
            if i in used_reinforce_indices:
                continue
                
            reinforce_benchmark = normalize_benchmark_name(reinforce_exp['benchmark'])
            
            if random_benchmark == reinforce_benchmark:
                matches.append({
                    'random': random_exp,
                    'reinforce': reinforce_exp,
                    'benchmark': random_benchmark
                })
                used_reinforce_indices.add(i)
                break
    
    return matches

def write_comparison_log(random_experiments, reinforce_experiments, matches, output_file):
    """Write the comparison results to the output file."""
    
    try:
        with open(output_file, 'w') as file:
            file.write("Experiment Comparison Results\n")
            file.write("=" * 50 + "\n\n")
            
            # Summary counts
            file.write(f"Total Random Agent Experiments: {len(random_experiments)}\n")
            file.write(f"Total Reinforce Experiments: {len(reinforce_experiments)}\n")
            file.write(f"Matching Experiments: {len(matches)}\n\n")
            
            # Write matching experiments comparison
            if matches:
                file.write("MATCHING EXPERIMENTS COMPARISON:\n")
                file.write("-" * 35 + "\n\n")
                for match in matches:
                    random_exp = match['random']
                    reinforce_exp = match['reinforce']
                    benchmark = match['benchmark']
                    
                    # Initial values
                    initial_luts = random_exp['initial_num_luts']
                    initial_depth = random_exp['initial_max_depth']
                    
                    # Final values for both methods
                    random_final_luts = random_exp['final_num_luts']
                    random_final_depth = random_exp['final_max_depth']
                    reinforce_final_luts = reinforce_exp['final_num_luts']
                    reinforce_final_depth = reinforce_exp['final_max_depth']
                    
                    # Calculate products
                    random_product = random_final_luts * random_final_depth
                    reinforce_product = reinforce_final_luts * reinforce_final_depth
                    
                    # Calculate improvements (negative values mean better performance)
                    random_lut_improvement = random_final_luts - initial_luts
                    reinforce_lut_improvement = reinforce_final_luts - initial_luts
                    random_depth_improvement = random_final_depth - initial_depth
                    reinforce_depth_improvement = reinforce_final_depth - initial_depth
                    
                    # Calculate how much better Reinforce is than Random
                    lut_difference = reinforce_lut_improvement - random_lut_improvement
                    depth_difference = reinforce_depth_improvement - random_depth_improvement
                    product_difference = reinforce_product - random_product
                    
                    file.write(f"Benchmark: {benchmark}\n")
                    file.write(f"  Initial: {initial_luts} LUTs, {initial_depth} depth\n\n")
                    
                    file.write("  Random Agent:\n")
                    file.write(f"    Final: {random_final_luts} LUTs, {random_final_depth} depth\n")
                    file.write(f"    Product (LUT × Depth): {random_product}\n\n")
                    
                    file.write("  Reinforce Agent:\n")
                    file.write(f"    Final: {reinforce_final_luts} LUTs, {reinforce_final_depth} depth\n")
                    file.write(f"    Product (LUT × Depth): {reinforce_product}\n\n")
                    
                    file.write("  Performance Comparison (Reinforce vs Random):\n")
                    # For LUT and depth, negative values are better (reduction), so we invert the difference
                    lut_improvement = random_lut_improvement - reinforce_lut_improvement
                    depth_improvement = random_depth_improvement - reinforce_depth_improvement
                    # For product, negative values are better (lower product), so we invert the difference
                    product_improvement = random_product - reinforce_product
                    
                    file.write(f"    LUT improvement: {lut_improvement:+d} (positive = Reinforce better)\n")
                    file.write(f"    Depth improvement: {depth_improvement:+d} (positive = Reinforce better)\n")
                    file.write(f"    Product improvement: {product_improvement:+d} (positive = Reinforce better)\n")
                    file.write("\n" + "="*50 + "\n\n")
            else:
                file.write("No matching experiments found between the two log files.\n\n")
            
            # Add summary section categorizing performance
            if matches:
                file.write("PERFORMANCE SUMMARY:\n")
                file.write("-" * 20 + "\n\n")
                
                reinforce_better = []
                random_better = []
                ties = []
                
                for match in matches:
                    benchmark = match['benchmark']
                    random_exp = match['random']
                    reinforce_exp = match['reinforce']
                    
                    # Calculate improvements
                    random_lut_improvement = random_exp['final_num_luts'] - random_exp['initial_num_luts']
                    reinforce_lut_improvement = reinforce_exp['final_num_luts'] - reinforce_exp['initial_num_luts']
                    random_depth_improvement = random_exp['final_max_depth'] - random_exp['initial_max_depth']
                    reinforce_depth_improvement = reinforce_exp['final_max_depth'] - reinforce_exp['initial_max_depth']
                    
                    random_product = random_exp['final_num_luts'] * random_exp['final_max_depth']
                    reinforce_product = reinforce_exp['final_num_luts'] * reinforce_exp['final_max_depth']
                    
                    # Determine overall winner based on product (LUT × Depth)
                    if reinforce_product < random_product:
                        reinforce_better.append(benchmark)
                    elif random_product < reinforce_product:
                        random_better.append(benchmark)
                    else:
                        ties.append(benchmark)
                
                file.write("Benchmarks where Reinforce outperformed Random:\n")
                if reinforce_better:
                    for benchmark in reinforce_better:
                        file.write(f"  • {benchmark}\n")
                else:
                    file.write("  None\n")
                file.write("\n")
                
                file.write("Benchmarks where Random outperformed Reinforce:\n")
                if random_better:
                    for benchmark in random_better:
                        file.write(f"  • {benchmark}\n")
                else:
                    file.write("  None\n")
                file.write("\n")
                
                if ties:
                    file.write("Benchmarks with tied performance:\n")
                    for benchmark in ties:
                        file.write(f"  • {benchmark}\n")
                    file.write("\n")
                
                file.write(f"Summary: Reinforce better in {len(reinforce_better)} benchmarks, ")
                file.write(f"Random better in {len(random_better)} benchmarks")
                if ties:
                    file.write(f", {len(ties)} ties")
                file.write("\n")
        
        print(f"Results written to '{output_file}'")
        
    except Exception as e:
        print(f"Error writing to file: {e}")

def main():
    random_log_file = "random_agent.log"
    reinforce_log_file = "reinforce.log"
    output_file = "comparison.log"
    
    print(f"Parsing random agent experiments from '{random_log_file}'...")
    random_experiments = parse_random_agent_experiments(random_log_file)
    
    print(f"Parsing reinforce experiments from '{reinforce_log_file}'...")
    reinforce_experiments = parse_reinforce_experiments(reinforce_log_file)
    
    print("Finding matching experiments...")
    matches = find_matching_experiments(random_experiments, reinforce_experiments)
    
    if random_experiments:
        print(f"Found {len(random_experiments)} random agent experiments")
    else:
        print("No random agent experiments found.")
    
    if reinforce_experiments:
        print(f"Found {len(reinforce_experiments)} reinforce experiments")
    else:
        print("No reinforce experiments found.")
    
    print(f"Found {len(matches)} matching experiments")
    
    write_comparison_log(random_experiments, reinforce_experiments, matches, output_file)

if __name__ == "__main__":
    main() 