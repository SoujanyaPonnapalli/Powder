
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
Development, 6(2):200–209, 1962 \
[2] Miguel Castro and Barbara Liskov. Practical Byzantine Fault Tolerance. In Proc. OSDI, 1999. \
[3] Upright Cluster Services.

---

## [Silent Data Corruptions at Scale](https://arxiv.org/pdf/2102.11245)
*Harish Dattatraya Dixit, Sneha Pendharkar, Matt Beadon, Chris Mason, Tejasvi Chakravarthy, Bharath Muthiah, Sriram Sankar, Facebook Inc*

### Highlights and Summary

- This paper shows that while prior work within this domain focused on soft errors due to radiation or synthetic fault injection, silent data corruptions are not limited to soft errors due to radiation or environmental effects with probabilistic models but that silent data corruptions can occur due to device characteristics and are repeatable at scale

- It is also possible for the devices to get weaker with usage. A computational block used frequently can show wear
  and tear, and degrade faster than the other parts of the CPU; Degradation based failures can have negative impact as
  the aging is not uniform across different chips that fall under this failure category; error correcting codes (ECC)
  are incorporate at devices to protect against such degradation

- It is statistically more likely to encounter silent data corruption with increasing CPU population. It is our observation that increased density and wider datapaths increase the probability of silent errors. This is not limited to CPUs and is applicable to special function accelerators and other devices with wide datapaths.

- They use distributed wordcount and (de)-compression applications to show how silent data corruption errors show up as application-level software bugs and discuss how they can be traced and detected across a large fleet of servers at Facebook

- A better way to prevent application-level failures is to implement software level redundancy and
  periodically verify that the data being computed is accurate at multiple checkpoints.
  It is important to consider the cost of accurate computation while adopting these approaches
  to large-scale data center infrastructure. The cost of redundancy has a direct effect on
  resources, more redundant the architecture, the larger the duplicate resource pool requirements.
  However, this only provides probabilistic fault tolerance to the application.

### Details:

- Example-1: In one such computation, when the file size was being computed, a file with a valid file size
  was provided as input to the decompression algorithm, within the decompression pipeline.
  The algorithm invoked the Scala power function which returned a 0 size value for a file
  which was known to have a non-zero decompressed file size. Since file size computation is now 0,
  the file was not written into the decompressed output database and application reported missing files
  after decompression.

- After a few iterations of debugging, it became obvious that the computation of 1.1 raised to the power of 53
  as an input to the math.pow function in Scala would always produce a result of 0 on Core 59 of the CPU. But on
  other cores, the result produced was correct.

- To debug a silent error, we cannot proceed forward without understanding which machine level instructions are executed.

- This paper also lists a few best practices to minimize computational errors (a total of 10)
    - Avoid absolute address references
    - Avoid unintended branches
    - External library references
    - Compiler optimizations
    - Stub and redundant instructions, etc

Hardware approaches to counter SDCs
    - protected datapaths?
    - specialized screenings?
    - understanding behavior at scale
    - architectural priority
All of these are future work at this time!

Software fault tolerance approaches
    - Redundancy
    - Fault-tolerant libraries

### References

[1] R. C. Baumann. 2005. Radiation-induced soft errors in advanced semiconductor technologies.
IEEE Transactions on Device and Materials Reliability 5, 3 (2005), 305–316.
https://doi.org/10.1109/TDMR.2005.853449 \
[2] G. C. Cardarilli, F. Kaddour, A. Leandri, M. Ottavi, S. Pontarelli, and R. Velazco.
2002. Bit flip injection in processor-based architectures: a case study.
In Proceedings of the Eighth IEEE International On-Line Testing Workshop (IOLTW 2002). 117–127.
https://doi.org/10.1109/OLT.2002.1030194 \
[3] James Elliott, Frank Mueller, Frank Stoyanov, and Clayton Webster. 2013.
Quantifying the impact of single bit flips on floating point arithmetic. Technical Report.
North Carolina State University. Dept. of Computer Science. \
[4] D. Fiala, F. Mueller, C. Engelmann, R. Riesen, K. Ferreira, and R. Brightwell.
2012. Detection and correction of silent data corruption for large-scale highperformance computing. In SC ’12:
Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis. 1–12.
https://doi.org/10.1109/SC.2012.49 \
[5] Paul M. Frank. 1990.
Fault diagnosis in dynamic systems using analytical and knowledge-based redundancy: A survey and some new results.
Automatica 26, 3 (1990), 459 – 474. https://doi.org/10.1016/0005-1098(90)90018-D \
[6] S. S. Mukherjee, J. Emer, and S. K. Reinhardt. 2005.
The soft error problem: an architectural perspective.
In 11th International Symposium on High-Performance Computer Architecture. 243–247.
https://doi.org/10.1109/HPCA.2005.37 \
[7] N. Oh, P. P. Shirvani, and E. J. McCluskey. 2002.
Error detection by duplicated instructions in super-scalar processors.
IEEE Transactions on Reliability 51, 1 (2002), 63–75. https://doi.org/10.1109/24.994913

---