## Powder: Fading Into Larger Quorums for Consensus with an Aging Cluster

This project aims to dyanmically determine two constants in a heteorogenous cloud cluster: 
  - Total number of servers in this cluster (n), and 
  - Maximum number of failures ie., crash and byzantine, that this cluster can tolerate (f).

First, it attempts to map the probability of byzantine failures to a server's age
  and its probability of experiencing hardware (besides power) failures.
Next, it formalizes the notion of a majority quorum in a heterogenous cluster with
  a distribution of servers, each with unique age and probabilities of byzantine failures.
Overall, it aims to provide a sustainable solution that extends the lifecycle
  of a server in agreement protocols while dynamically moderating the number of
  nodes required to tolerate failures.

### Main contributions

- Analyzing the degradation (or aging) of servers in the cloud
- Tieing a server's age to its probability of byzantine behavior
- Formalizing the notion of majority quorum within a heterogenous cluster
- Showing the advantages of this approach
    - Lower resource utilization
    - Extending server lifecycle to lower carbon emissions
    - Latency and throughput overheads
    - CPU, memory, and network bandwidth consumptions
