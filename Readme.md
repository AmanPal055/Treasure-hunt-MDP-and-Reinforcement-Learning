# Markov Decision Processes & Reinforcement Learning

**Introduction to AI — Assignment 2** | Two tabular decision-making agents demonstrating **model-based planning** and **model-free reinforcement learning**.

## Overview

| | Part A — TreasureHunt | Part B — Highway Driving |
|---|---|---|
| **Environment** | A ship navigates uncertain winds, land, and moving pirates | A car travels through traffic on a four-lane highway |
| **Method** | **Value Iteration** for a known Markov Decision Process | **Tabular Q-Learning** from simulator experience |
| **What the agent knows** | Transition probabilities and reward parameters | Receives states and rewards through interaction |
| **Goal** | Collect treasures and reach the fort while avoiding pirates | Drive efficiently without collisions |

This repository includes the two submitted `agent.py` implementations together with the course-provided simulators, test inputs, visual assets, and run scripts. **The agent files are unchanged from the submission.**

## Project structure

```text
MDP-and-Reinforcement-Learning/
├── part_a/
│   ├── agent.py         # Submitted Value Iteration agent
│   ├── env.py           # Course-provided TreasureHunt environment
│   ├── run.py           # Part A runner
│   ├── images/          # Visualization assets
│   └── tests/           # Example layouts and probability files (1, 2, 3)
├── part_b/
│   ├── agent.py         # Submitted tabular Q-Learning agent
│   ├── env.py           # Course-provided highway simulator
│   └── run.py           # Part B runner
├── requirements.txt    # Dependencies from the supplied starter ZIP
├── .gitignore
└── README.md
```

## Part A: TreasureHunt — Value Iteration

The ship tries to move **up, down, left, or right**. Wind can change the executed direction, and two pirates patrol separate regions. The state captures the ship's location, both pirate locations, and the uncollected treasures.

The agent in [`part_a/agent.py`](part_a/agent.py):

1. Loads the layout, movement probabilities, rewards, and discount factor.
2. Uses **breadth-first search (BFS)** to identify the pirates' patrol regions.
3. Builds compact state indices, including a joint index for both pirates, and four treasure-collection stages.
4. Precomputes possible ship movements, pirate movements, and collisions.
5. Applies **Value Iteration**, using Bellman updates to compare the expected values of all four actions; the implementation allows up to 150 iterations per treasure stage and checks convergence.
6. Caches the policy for fast action selection using `get_action()`.

Value Iteration maximizes the expected return:

$$
V(s) \leftarrow \max_a \sum_{s'} P(s'\mid s,a)\left[R(s,a,s') + \gamma V(s')\right]
$$

Actions: `0 = UP`, `1 = DOWN`, `2 = LEFT`, `3 = RIGHT`.

## Part B: Highway Driving — Q-Learning

The agent observes **speed, lane, and the nearest-car distance in each of four lanes**. It must learn good driving actions by interacting with the simulator without using a known transition model.

The agent in [`part_b/agent.py`](part_b/agent.py):

1. Represents each observation as a six-element tuple: `(speed, lane, d1, d2, d3, d4)`.
2. Stores Q-values in a **lazy tabular dictionary**, initialized to `6.0` for unseen states.
3. Selects actions using **epsilon-greedy exploration**, gradually reducing epsilon from `1.0` to `0.01` during training.
4. Uses a visit-count-based **adaptive learning rate** (bounded between `0.02` and `0.25`).
5. Updates action values using the **Q-Learning temporal-difference update**, with a terminal-state case.
6. Uses the highest-valued action for evaluation.

Q-Learning update:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\right]
$$

For a terminal transition, the target is the immediate reward only.

Actions: `0 = increase speed`, `1 = decrease speed`, `2 = increase lane`, `3 = decrease lane`, `4 = no-op`.

## Getting started

### 1. Prerequisites

- Python **3.10** (the version specified by the assignment)
- The packages listed in [`requirements.txt`](requirements.txt)
- A Unix-like system is recommended because the supplied runners use `signal.SIGALRM` for training timeouts.

The course setup uses Conda:

```bash
conda create -n ai_a2 python=3.10
conda activate ai_a2
pip install -r requirements.txt
```

A `venv` with Python 3.10 is also suitable for local experimentation.

### 2. Run Part A (TreasureHunt)

From the repository root:

```bash
cd part_a
python run.py --layout_file tests/1/layout.txt --prob_file tests/1/prob.txt \
  --num_runs 10 --T 120 --output_dir outputs/test1
```

The runner outputs an average reward score and creates episode GIFs under `part_a/outputs/test1/`. You can use `tests/2/` or `tests/3/` instead of `tests/1/` to try the other supplied configurations.

### 3. Run Part B (Highway Driving)

From the repository root (or go back from `part_a/` using `cd ..`):

```bash
cd part_b
python run.py --num_runs 10 --T 120 --output_dir outputs/highway --df 0.99
```

The runner outputs an average reward score and creates episode GIFs under `part_b/outputs/highway/`.

**CLI arguments:** `--T` is the training-time limit in seconds; `--num_runs` is the number of evaluation episodes; `--output_dir` is where GIFs are saved; `--df` is Part B's discount factor. Part A instead loads its discount factor from `prob.txt`.

## Notes about the starter code

The uploaded starter archive needed three small **packaging/runner** adjustments for this repository:

- Renamed `requirments.txt` to **`requirements.txt`** so the documented `pip install -r requirements.txt` command works.
- Fixed Part B's `run.py` import from `ta_agents` to **`agent`**, matching the submitted file.
- Added `os.makedirs(output_dir, exist_ok=True)` to both run scripts before writing GIFs, so their output folders are created automatically.

**No changes were made to either submitted `agent.py` or to the environment implementations** (`env.py`). These runner fixes are for using the GitHub project and are not a replacement for the official Gradescope submission format.

## Learning outcomes

This assignment contrasts two ways to solve sequential decision-making problems:

- **Model-based MDP planning:** Compute a policy from known transition and reward models.
- **Model-free RL:** Learn an action-value policy through rewards and environment interactions.

It also demonstrates compact state representations, transition precomputation, dynamic programming, exploration/exploitation, and tabular temporal-difference learning.

## Attribution and sharing

This project is based on **Introduction to AI — Assignment 2**. The `env.py`, `run.py`, test inputs, and visualization assets originate from course-provided starter code; the two `agent.py` files are the submitted solutions. **Check the course's public-sharing policy and any collaborator permissions before publishing** starter materials or solution code. No permission or open-source license for the starter materials was supplied with the archive.

No performance claims are made without retained evaluation logs.
