## [Powder: Real Life is Uncertain. Consensus Should Be Too!](https://dl.acm.org/doi/10.1145/3713082.3730374)

This project aims to dyanmically determine two constants in a heteorogenous cloud cluster: 
  - Total number of servers in this cluster (n), and 
  - Maximum number of failures ie., crash and byzantine, that this cluster can tolerate (f).

First, it attempts to map the probability of failures to a server's age
  and its probability of experiencing hardware (besides power) failures.
Next, it formalizes the notion of a majority quorum in a heterogenous cluster with
  a distribution of servers, each with unique age and probabilities of byzantine failures.
Overall, it aims to provide a sustainable solution that extends the lifecycle
  of a server in agreement protocols while dynamically moderating the number of
  nodes required to tolerate failures.
The requirements are all to match a certain 9s of durability and availability SLOs.

### Main contributions

- Analyzing the degradation (or aging) of servers in the cloud
- Tieing a server's age to its probability of byzantine behavior
- Formalizing the notion of majority quorum within a heterogenous cluster
- Showing the advantages of this approach
    - Lower resource utilization
    - Extending server lifecycle to lower carbon emissions
    - Latency and throughput overheads
    - CPU, memory, and network bandwidth consumptions

## Setup

### Python environment

1. Create and activate a virtual environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

### Wolfram kernel configuration

The notebooks use Wolfram via `wolframclient`. Configure the kernel path in
`notebooks/config.yaml` (or set `WOLFRAM_KERNEL_PATH` in your environment).

Example config snippet:

```
wolfram:
  kernel_paths:
    darwin: "/Applications/Wolfram.app/Contents/MacOS/WolframKernel"
    linux: "/usr/local/Wolfram/Desktop/11.3/Executables/WolframKernel"
    windows: "C:\\Program Files\\Wolfram Research\\Wolfram Desktop\\11.3\\WolframKernel.exe"
```

If you are new to the Wolfram Python client, see:
https://reference.wolfram.com/language/WolframClientForPython/docpages/basic_usages.html

### Running notebooks

From the repo root:

```
jupyter lab
```

Open notebooks in `notebooks/` (e.g., `markov-calculator.ipynb`).

### Notebook output hygiene (optional but recommended)

To keep diffs clean, consider `nbstripout`:

```
pip install nbstripout
nbstripout --install
```

This strips notebook outputs on commit while preserving the code cells.


