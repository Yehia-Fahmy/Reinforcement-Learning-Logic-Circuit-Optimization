# Reinforcement Learning for Logic Circuit Optimization

> **AI-Powered VLSI Circuit Design Optimization**

A machine learning system that uses **Reinforcement Learning (RL)** to automatically optimize digital logic circuits by minimizing hardware resources while maintaining performance. This project demonstrates advanced AI techniques applied to real-world semiconductor design challenges.

## 🎯 What This Project Does

This project implements a **Reinforcement Learning agent** that optimizes Look-Up Table (LUT) netlists - the fundamental building blocks of modern FPGAs and ASICs. The AI agent learns to strategically merge LUTs to:

- **Minimize hardware cost**: Reduce the total number of LUTs required
- **Optimize performance**: Minimize circuit depth for faster operation
- **Maintain feasibility**: Respect input constraints (≤6 inputs per LUT)

## 🚀 Key Technical Achievements

- **Custom RL Environment**: Built `opt_gym` library for circuit optimization
- **REINFORCE Implementation**: Advanced policy gradient algorithm using JAX/Flax
- **Multi-objective Optimization**: Balances hardware cost vs. performance trade-offs
- **Industry Benchmarks**: Tested on EPFL benchmark suite (real circuit designs)
- **Transfer Learning**: Models generalize across different circuit types
- **Production-Ready**: Compatible with standard FPGA synthesis tools

## 📊 Performance Highlights

- **Automated Optimization**: Reduces manual circuit design time from hours to minutes
- **Scalable**: Handles circuits with 1000+ LUTs efficiently
- **Intelligent**: Learns optimal merging strategies through trial and error
- **Generalizable**: Works across arithmetic, control, and random logic circuits

## 🛠️ Quick Start

### Prerequisites
```bash
# Create virtual environment
python3 -m venv ~/ml-playground
source ~/ml-playground/bin/activate
pip install -r requirements.txt
```

### Basic Usage
```bash
# Run REINFORCE agent on adder circuit
python reinforce.py --base arithmetic --name adder

# Test random baseline for comparison
python random_agent.py --base arithmetic --name adder

# Explore available benchmarks
python epfl_benchmarks.py
```

### Key Files
- **`reinforce.py`**: Main RL implementation using REINFORCE algorithm
- **`opt_gym/`**: Custom RL environment for circuit optimization
- **`problem_*.py`**: Progressive implementations (exhaustive search → heuristics → RL)
- **`epfl_benchmarks.py`**: Benchmark circuit loader

## 🎯 Technical Deep Dive

### RL Environment Design
- **State Space**: LUT netlist representation (connectivity, truth tables, depth)
- **Action Space**: LUT merge operations (which LUTs to combine)
- **Reward Function**: Multi-objective (LUT count reduction + depth optimization)
- **Termination**: Episode ends when no valid merges remain

### Algorithm Implementation
- **Policy Network**: Neural network that learns optimal merge strategies
- **Feature Engineering**: Custom state representations for circuit topology
- **Training**: JAX-based implementation for high-performance computing
- **Evaluation**: Comprehensive benchmarking across circuit types

## 📈 Business Impact

This solution addresses critical challenges in:
- **Semiconductor Industry**: Automated FPGA/ASIC optimization
- **EDA Tools**: Next-generation circuit synthesis algorithms
- **Hardware Design**: Reduced time-to-market for new chips
- **Cost Optimization**: Lower hardware requirements = lower production costs

## 🔧 Technical Stack

- **Deep Learning**: JAX/Flax for high-performance neural networks
- **Reinforcement Learning**: REINFORCE policy gradient algorithm
- **Circuit Design**: Custom LUT netlist data structures
- **Benchmarking**: EPFL industry-standard test circuits
- **Optimization**: Multi-objective reward functions

## 📁 Project Structure

```
├── reinforce.py              # Main RL implementation
├── opt_gym/                  # Custom RL environment
│   ├── core.py              # LUT netlist data structures
│   ├── envs.py              # RL environment wrapper
│   └── epfl/                # Benchmark circuits
├── problem_*.py             # Progressive implementations
├── epfl_benchmarks.py       # Benchmark loader
└── requirements.txt         # Dependencies
```

## 💼 For Recruiters

**This project demonstrates:**
- **Advanced ML Engineering**: Custom RL environments and algorithms
- **Domain Expertise**: Deep understanding of VLSI circuit design
- **Production Focus**: Industry-standard benchmarks and evaluation
- **Technical Depth**: From theoretical RL to practical implementation
- **Problem-Solving**: Multi-objective optimization in constrained environments

**Key Skills Showcased:**
- Reinforcement Learning (REINFORCE, policy gradients)
- Deep Learning (JAX/Flax, neural networks)
- VLSI Design (LUT optimization, circuit synthesis)
- Software Engineering (modular design, benchmarking)
- Research & Development (algorithm experimentation)

## 🚀 Getting Started

1. **Explore the environment**:
   ```bash
   python epfl_benchmarks.py
   python lutenv_explore.py
   ```

2. **Run the RL agent**:
   ```bash
   python reinforce.py --base arithmetic --name adder
   ```

3. **Compare with baselines**:
   ```bash
   python random_agent.py --base arithmetic --name adder
   ```

4. **Study the implementations**:
   - `problem_0.py`: Exhaustive search (dynamic programming)
   - `problem_1.py`: Heuristic-based optimization
   - `problem_2.py`: Advanced RL techniques
   - `problem_3.py`: Gymnasium integration (advanced)

This project showcases the intersection of **AI/ML** and **hardware design**, demonstrating how modern machine learning techniques can solve complex engineering optimization problems in the semiconductor industry.
