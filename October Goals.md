High Level goals:

1. TiDB powder integration
   1. Make the ILP aware of machine constraints (storage/partition size)
   2. Make the ILP integrate into TiDB
      1. ILP can initialize the TiDB deployment
      2. ILP can recover the system from failures
   3. ILP has an optimization cost for GB movement (or constraint?)
   4. Make TiDB Placement Driver call out to the ILP (where the ILP/sim will pick which nodes stick around in a merge and the ILP/sim will pick which nodes will take a split partition)
2. Simulator Calibration
   1. Measure time it takes to recover a node (data download, snapshot freq, snapshot size, etc)
   2. Figure out way to make simulator and deployment agree on node selection
3. Correlated Failures Goal
   1. Need to add correlated failures into the 1node model
   2. Make the ILP region aware
   3. Review formal proof of availability between RSMs has strictly non-negative covariance
4. Experiment pre-requisites
   1. Verify that a 24hr deployment would be enough time to see downtime with a cluster
   2. Verify that the machine price/availability curves are correlated enough to show a benefit
   3. Verify that 24hr re-partitioning is a good use of time
5. Software Update scheduling
   1. Figure this out just generally. Maybe random sample different schedules so that I can update 1/7 of the nodes in 24hrs and pick the one that works out best including cost to reconfigure X GB
6. Deployment setup
   1. Figure out how to get data from the Azure dataset for hardware failures
   2. Make TiDB Deployment script
   3. Find different workloads that we can run TiDB on for 24hrs
      1. Uniform
      2. Diurnal
      3. Hot Partition
      4. Shifting hotspot
7. Offline setup
   1. Make reconfiguration splitter planner
   2. Benchmark large monte carlo run performance
   3. Make long running 1yr simulated experiment runner

Execution order:

1. Get canonical HW failure, price, and SW failure information from Azure dataset
   1. Unless included in the dataset we can say that there will be some planned downtime each week for a node. The “week” duration between planned downtime can be sampled from a normal distribution, but each of the nodes will “start their week” at a uniformly random sampled interval to not all shut down at the same time
   2. Get a number for how frequently a region will be down and how long a region might be down for from researching past outages
2. Validate the number of machines we should use for an experiment and how long the experiment should be.
   1. How many failures would we expect in a 24hr deployment. Would we expect the ILP to have a decent win against our baseline heuristics?
3. Add correlated failures into the monte carlo simulator
4. Make download speed for monte carlo simulator dependent on which regions the nodes are in
5. Make the ILP pipeline reject placements which put all nodes into the same region without running any markov calculations
6. Add storage information into the ILP. A partition has X GB of data in it. The data is loaded into k machines. Adding a new machine into the partition will “cost” k GB of data transfer. We want to keep track of how many GB must be transferred inter region and intra region. Then lets add a constraint to the optimizer so that we require < Z gb of movement for a candidate placement to be valid. Finally we can change the ILP to search for minimum Z that meets our availability requirements. Then we can minimize cost with a maximum data transfer of Z * 1.3. Finally we can also add in constraints where each machine will have a maximum amount of storage and a placement is only valid if the data of all partitions each machine is included in sum to be less than the machine’s maximum available storage
7. Add TiDB to the repo and run a test deployment on AWS
8. Write failure emulator which will kill and restart tidb in accordance to the machine’s transient failure rate or will kill and not restart tidb in accordance with the machine’s permanent failure rate
9. Make TiDB placement driver variant which queries the ILP (with pre-computed dataset) to handle reactive machine failures
10. Measure download speed between regions, leader election timing, other misc setup timings with a benchmark to get a probability distribution for each phase