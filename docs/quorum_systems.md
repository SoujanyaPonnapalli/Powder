
### List of papers

1. [Probabilistic Quorum Systems](https://github.com/SoujanyaPonnapalli/Powder/blob/main/docs/quorum_systems.md#probabilistic-quorum-systems)
2. [Probabilistic Quorum Systems (Extended)](https://github.com/SoujanyaPonnapalli/Powder/edit/main/docs/quorum_systems.md#probabilistic-quorum-systems-extended)
3. [Probabilistic Quorums for Dynamic Systems](https://github.com/SoujanyaPonnapalli/Powder/edit/main/docs/quorum_systems.md#probabilistic-quorums-for-dynamic-systems)
4. [Probabilistically Bounded Staleness for Practical Partial Quorums](https://arxiv.org/pdf/1204.6082)
5. [Probabilistic Consistency Guarantee in Partial Quorum-Based Data Store](https://ieeexplore.ieee.org/document/8998160)
6. [Fault-Tolerant Storage and Quorum Systems for Dynamic Environments](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=110fba38e01f1397d772c30b29cf23faa99ae521)

### [Probabilistic Quorum Systems](https://reitermk.github.io/papers/2001/IC.pdf)

#### Highlights:

#### Summary:

#### Details:

#### References:

### [Probabilistic Quorum Systems (Extended)](dc)

#### Highlights:

This paper extends Probabilistic Quorum Systems (PQS) to cope with scalability and high dynamism in the following ways.
- First, each participant has only a partial knowledge of the full system to avoid maintaining any 
  global information of the system size and its constituents. To this end, each individual member
  selection probability is non-uniform.
- Second, PQSs are extended to address evolving quorums as the system grows/shrinks 
  in order for them to remain viable.
- Using a PQS, each participant can disseminate new updates to shared data by contacting 
  a subset (a probabilistic quorum) of `k√n` processes chosen uniformly at random, 
  where n is the size of the system and k is a reliability parameter. 

#### Summary:

#### Details:

#### References:

### [Probabilistic Quorums For Dynamic Systems](https://www.cs.huji.ac.il/w~ittaia/papers/dpq-TR.pdf)

#### Highlights:

- For a dynamic model where nodes constantly join and leave the system, a quorum chosen at time s must evolve and
  transform as the system grows/shrinks in order to remain viable; this paper introduces dynamic ε-intersecting
  quorum systems

- This paper 

#### Summary:

#### Details:

- A quorum system is a set of sets such that every two sets in the quorum system intersect.
- An ε-intersecting quorum system is a distribution on sets such that every two sets
  from the distribution intersect with probability 1 − ε.
- This relaxation of consistency results in a dramatic improvement of the load balancing and resiliency
of quorum systems, making the approach especially attractive for scalable and dynamic settings

Definition 2.1 (-intersecting quorum system[15]) Let Q be a set system, let ac be an access strategy for Q, and let 0 <  < 1 be given. The tuple hQ, aci is an -intersecting quorum system if
Pr[Q ∩ Q0 6= ∅] ≥ 1 − , where the probability is taken with respect to the strategy ac.


#### References:


