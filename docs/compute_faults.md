
## [List of Papers]()
1. [Cores that don't count | Google](https://github.com/SoujanyaPonnapalli/Powder/new/main/docs/reading#cores-that-dont-count)
2. [Data corruptions at Scale | Facebook](https://github.com/SoujanyaPonnapalli/Powder/new/main/docs/reading#silent-data-corruptions-at-scale)

---

## [1. Cores that don't count](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s01-hochschild.pdf)
*Peter H. Hochschild, Paul Turner, Jeffrey C. Mogul, Rama Govindaraju, Parthasarathy Ranganathan, David E. Culler, and Amin Vahdat. 2021.
In Workshop on Hot Topics in Operating Systems (HotOS ’21)*

### Highlights:

- This paper provides evidence for silent computational errors in practice
  which often go undetected during manufacturing testing of the hardware
  and presents a call-to-action for new systems software
- They emphasize the need for software that detects, isolates, and tolerates
  silent data corruption due to computational hardware failure aka
  `silent corrupt execution errors` (CEEs)
- These errors are besides the incremental increases in
  the background rate of hardware errors;
  CEEs can manifest long after initial installation; 
  and that they typically afflict specific cores aka `mercurial cores` on multi-core CPUs, 
  rather than the entire chip.
- CEEs can appear to be sudden and unpredictable for several reasons including *minor software changes*; Software updates that result in heavy-use of otherwise rarely employed instructions

Some quoted examples of interest: 
  - A subset of servers that are repeatedly responsible for producing erroneous results
  - Common assumptions: storage devices and networks can corrupt data at rest or in transit, we are accustomed
to thinking of processors as fail-stop or fail-noisy; but silent computational errors are common; we observe on the order of a `few mercurial cores per several thousand machines`

### Summary:

*Why are we learning about these errors only now?*
- Many plausible reasons: larger server fleets; 
increased attention to overall reliability; 
improvements in software development that reduce the rate of software bugs. 
More fundamentally, ever-smaller feature sizes that push closer to the limits of CMOS scaling, 
coupled with ever-increasing complexity in architectural design. 
Together, these create new challenges for the verification methods to detect diverse manufacturing defects – especially those defects that manifest in corner cases, or only after post-deployment aging.

*What is the nature of such errors?* 
- Errors in computation due to mercurial cores can compound to significant increase in the blast radius of the failures they can cause
- Often manifest as software bugs or application-level bugs and it often takes a lot of engineering time to figure out their root causes

*How common are silent computational faults?*
- storage devices and networks can corrupt data at rest or in transit, we are accustomed
to thinking of processors as fail-stop or fail-noisy; but silent computational errors are common; we observe on the order of a `few mercurial cores per several thousand machines`

*How easy are they to be fixed outside software?*
- Storage and memory faults causes data corruption at rest; network faults cause data corruption errors 
  while data is being transmitted. Nevertheless, it is easy to find the correct value of the data and to
  detect mismatches unlike computational faults. However, the cost of detecting and correcting storage and network errors can be amortized 
  over larger data chunks which seems harder to do for computational instructions

*Why are these errors of interest to us?*
- The trade-off between performance and hardware reliability is becoming more difficult;
We are entering an era in which unreliable hardware increasingly fails silently rather than
failstop, which changes some of our fundamental assumptions.
We can not rely on chip vendors to test for diverse manufacturing defects.
Moreover, there is already a vast installed base of vulnerable chips, and we need to find
scalable ways to keep using these systems without suffering from frequent errors,
rather than replacing them (at enormous expense) or waiting several years for new,
more resilient hardware.

*Can we design software that can tolerate CEEs without excessive overheads?*
- They suggest checking at applications following the E2E argument and discuss
systems that support checkpointing and restarting work on a different core from the
latest checkpoint
- The authors already hint at consensus:
One well-known approach is triple modular redundancy[1],
where the same computation is done three times, and (under
the assumption that at most one fails) majority-voting yields
a reliable result
- Byzantine fault tolerance [2] has been
proposed as a means for providing resilience against arbitrary
non-fail-stop errors [3]; BFT might be applicable to CEEs in
some cases.

### Details:

- Most crash-fault tolerant consensus (CFT) protocols build on this assumption that server failures are fail-stop in nature
- However, this paper discusses `ephemeral computational errors` which go undetected in the manufacturing tests which are often silent in nature where the only observable symptom is an erroneous computation
- The authors coin the term `mercurial cores` to refer to cores that silently produce erroneous results
- Micro-code updates?
- Silent data corruption is a symptom that is caused by mercurial cores. Different types of observed symptoms 
    - Wrong answers that are detected nearly immediately,
through self-checking, exceptions, or segmentation faults,
which might allow automated retries.
    - Machine checks, which are more disruptive.
    -  Wrong answers that are detected, but only after it is too
late to retry the computation.
    - Wrong answers that are never detected

- Wrong answers that are not immediately
detected have potential real-world consequences: these can
propagate through other (correct) computations to amplify
their effects – for example, bad metadata can cause the loss of
an entire file system, and a corrupted encryption key can render large amounts of data permanently inaccessible.

- Failures mostly appear non-deterministically at variable rate. Faulty cores typically
fail repeatedly and intermittently, and often get worse with
time; we have some evidence that aging is a factor. In a multicore processor, typically just one core fails, often consistently.



## References
[1] The Use of Triple-Modular Redundancy to Improve Computer Reliability. IBM Journal of Research and
Development, 6(2):200–209, 1962
[2] Miguel Castro and Barbara Liskov. Practical Byzantine Fault Tolerance. In Proc. OSDI, 1999.
[3] Upright Cluster Services. 

## [Silent Data Corruptions at Scale](https://arxiv.org/pdf/2102.11245)
*Harish Dattatraya Dixit, Sneha Pendharkar, Matt Beadon, Chris Mason, Tejasvi Chakravarthy, Bharath Muthiah, Sriram Sankar, Facebook Inc*
