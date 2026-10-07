//! Job-level worker pool.
//!
//! A single Monte Carlo experiment runs entirely on one thread; parallelism
//! is applied *across* independent jobs.  Workers are spawned once and live
//! for the whole run -- never one thread per job -- and they claim work in
//! **batches**, so a worker synchronises with the queue once per batch
//! rather than once per job.
//!
//! ```text
//! reader thread      bounded channel             persistent workers
//! stdin NDJSON  -->  Vec<(idx, line)> batches -->  loop {
//!                        (backpressure)                claim one batch
//!                                                      run every job in it
//!                                                      send Vec<(idx, result)>
//!                                                  }
//!                                                         |
//! collector  <-- reorder by idx --> stdout NDJSON <-------+
//! ```
//!
//! Jobs are independent and individually seeded, so the output is identical
//! for any worker count and any batch size.  Only the ordering guarantee
//! differs: [`Ordering::InputOrder`] buffers to restore input order, while
//! [`Ordering::AsCompleted`] emits as soon as a batch lands.

use std::io::{BufRead, Write};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::job::{run_job_json, strip_runs, JobResult};

/// How results are written out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    /// Emit in the order jobs were read.  Costs a reorder buffer.
    InputOrder,
    /// Emit as soon as each batch completes.  Lower latency, arbitrary order.
    AsCompleted,
}

/// Pool configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolConfig {
    /// Number of persistent worker threads.
    pub workers: usize,
    /// Jobs handed to a worker per claim.
    pub batch_size: usize,
    /// How results are written out.
    pub ordering: Ordering,
    /// Whether to drop per-run detail and emit only aggregates.
    pub summary_only: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            workers: default_workers(),
            batch_size: 16,
            ordering: Ordering::InputOrder,
            summary_only: false,
        }
    }
}

/// Available parallelism, or 1 if it cannot be determined.
pub fn default_workers() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

type Batch = Vec<(usize, String)>;

/// A finished job: its input position and its already-encoded NDJSON line.
///
/// Workers encode their own results.  Doing it in the collector would put
/// all the JSON serialisation of a wide run on one thread, and would make
/// the in-order reorder buffer hold whole `JobResult` structs -- including
/// every per-run record -- instead of the bytes they turn into.
type ResultBatch = Vec<(usize, Vec<u8>)>;

/// Size of the `n`-th batch under the start-up ramp.
///
/// The first `workers` batches hold a single job, so every worker has
/// something to do as soon as the reader has read that many lines.  After
/// each full sweep of the pool the size doubles, up to `batch_size`, which
/// is where the per-batch queue synchronisation stops mattering.
fn ramped_batch_size(emitted: usize, workers: usize, batch_size: usize) -> usize {
    let sweeps = emitted / workers.max(1);
    let size = 1usize.checked_shl(sweeps.min(31) as u32).unwrap_or(usize::MAX);
    size.clamp(1, batch_size)
}

/// Encode one result as an NDJSON line, trailing newline included.
fn encode(result: &JobResult) -> Vec<u8> {
    let mut line = serde_json::to_vec(result).unwrap_or_else(|e| {
        format!(r#"{{"error":"failed to encode result: {e}"}}"#).into_bytes()
    });
    line.push(b'\n');
    line
}

/// Read NDJSON jobs from `input`, run them across the pool, and write NDJSON
/// results to `output`.
///
/// Returns the number of jobs processed.
pub fn run_stream<R: BufRead + Send, W: Write>(
    input: R,
    output: &mut W,
    config: PoolConfig,
) -> std::io::Result<usize> {
    let workers = config.workers.max(1);
    let batch_size = config.batch_size.max(1);

    // Bounded so a slow consumer applies backpressure to the reader and
    // memory stays flat however long the stream is.
    let (job_tx, job_rx) = sync_channel::<Batch>(workers * 2);
    let (result_tx, result_rx) = sync_channel::<ResultBatch>(workers * 2);

    let shared_rx = Arc::new(Mutex::new(job_rx));

    thread::scope(|scope| -> std::io::Result<usize> {
        for _ in 0..workers {
            let shared_rx = Arc::clone(&shared_rx);
            let result_tx: SyncSender<ResultBatch> = result_tx.clone();
            let summary_only = config.summary_only;

            scope.spawn(move || {
                loop {
                    // Hold the lock only long enough to take a batch; the
                    // work itself happens outside it.
                    let batch = {
                        let rx = shared_rx.lock().expect("job queue mutex poisoned");
                        rx.recv()
                    };
                    let Ok(batch) = batch else {
                        break; // Reader finished and the channel closed.
                    };

                    let mut done: ResultBatch = Vec::with_capacity(batch.len());
                    for (index, line) in batch {
                        let result = run_job_json(&line);
                        let result = if summary_only {
                            strip_runs(result)
                        } else {
                            result
                        };
                        done.push((index, encode(&result)));
                    }

                    if result_tx.send(done).is_err() {
                        break; // Collector went away.
                    }
                }
            });
        }
        // Drop the template sender so the collector sees a close once every
        // worker has exited.
        drop(result_tx);

        let reader = scope.spawn(move || -> std::io::Result<usize> {
            let mut index = 0usize;
            let mut emitted_batches = 0usize;
            // Ramp the batch size: the first `workers` batches hold one job
            // each so every worker starts immediately, then the size
            // doubles each time around until it reaches the configured
            // value.  A flat batch size starves a wide run -- with 200
            // workers and batches of 16, the reader has to get 3,200 jobs
            // in before the last worker sees anything.
            let mut current = ramped_batch_size(emitted_batches, workers, batch_size);
            let mut batch: Batch = Vec::with_capacity(current);

            for line in input.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                batch.push((index, line));
                index += 1;
                if batch.len() >= current {
                    if job_tx.send(std::mem::take(&mut batch)).is_err() {
                        return Ok(index);
                    }
                    emitted_batches += 1;
                    current = ramped_batch_size(emitted_batches, workers, batch_size);
                    batch = Vec::with_capacity(current);
                }
            }

            if !batch.is_empty() {
                let _ = job_tx.send(batch);
            }
            // Closing the channel is what tells the workers to stop.
            drop(job_tx);
            Ok(index)
        });

        let mut written = 0usize;
        match config.ordering {
            Ordering::AsCompleted => {
                for done in result_rx {
                    for (_, line) in done {
                        output.write_all(&line)?;
                        written += 1;
                    }
                    output.flush()?;
                }
            }
            Ordering::InputOrder => {
                // Hold finished lines until their turn comes.  The job
                // channel is bounded, so the number in flight -- and hence
                // the size of this buffer -- is bounded too.
                let mut pending: std::collections::HashMap<usize, Vec<u8>> =
                    std::collections::HashMap::new();
                let mut next = 0usize;

                for done in result_rx {
                    for (index, line) in done {
                        pending.insert(index, line);
                    }
                    let mut wrote_any = false;
                    while let Some(line) = pending.remove(&next) {
                        output.write_all(&line)?;
                        written += 1;
                        next += 1;
                        wrote_any = true;
                    }
                    if wrote_any {
                        output.flush()?;
                    }
                }

                // Any stragglers, in case indices arrived out of order at
                // the very end.
                let mut leftover: Vec<usize> = pending.keys().copied().collect();
                leftover.sort_unstable();
                for index in leftover {
                    if let Some(line) = pending.remove(&index) {
                        output.write_all(&line)?;
                        written += 1;
                    }
                }
                output.flush()?;
            }
        }

        output.flush()?;
        reader.join().expect("reader thread panicked")?;
        Ok(written)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn job_line(job_id: &str, seed: u64, sims: usize) -> String {
        format!(
            r#"{{"job_id":"{job_id}","mode":"monte_carlo",
                 "run":{{"max_time":86400.0,"stop_on_data_loss":false,"num_simulations":{sims},"base_seed":{seed}}},
                 "node_configs":{{"c":{{"region":"us-east","cost_per_hour":1.0,
                   "failure_dist":{{"type":"exponential","rate":0.0001}},
                   "recovery_dist":{{"type":"constant","value":600.0}},
                   "data_loss_dist":{{"type":"constant","value":863999999.0}},
                   "log_replay_rate_dist":{{"type":"constant","value":100.0}},
                   "snapshot_download_time_dist":{{"type":"constant","value":0.0}},
                   "spawn_dist":{{"type":"constant","value":0.0}}}}}},
                 "cluster":{{"target_cluster_size":3,"nodes":[
                   {{"node_id":"node0","config":"c"}},
                   {{"node_id":"node1","config":"c"}},
                   {{"node_id":"node2","config":"c"}}]}},
                 "protocol":{{"type":"leaderless"}},
                 "strategy":{{"type":"noop"}}}}"#
        )
        .replace('\n', "")
    }

    fn stream(jobs: &[String], config: PoolConfig) -> Vec<JobResult> {
        let input = jobs.join("\n");
        let mut output = Vec::new();
        let count = run_stream(Cursor::new(input), &mut output, config).unwrap();
        let text = String::from_utf8(output).unwrap();
        let results: Vec<JobResult> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(results.len(), count);
        results
    }

    #[test]
    fn processes_every_job_in_input_order() {
        let jobs: Vec<String> = (0..20)
            .map(|i| job_line(&format!("job{i}"), i as u64, 2))
            .collect();
        let results = stream(&jobs, PoolConfig::default());

        assert_eq!(results.len(), 20);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.job_id.as_deref(), Some(format!("job{i}").as_str()));
            assert!(result.error.is_none());
        }
    }

    #[test]
    fn output_is_identical_across_worker_and_batch_settings() {
        let jobs: Vec<String> = (0..25)
            .map(|i| job_line(&format!("job{i}"), i as u64, 3))
            .collect();

        let baseline = stream(
            &jobs,
            PoolConfig {
                workers: 1,
                batch_size: 1,
                ..Default::default()
            },
        );

        for (workers, batch_size) in [(1, 16), (2, 1), (4, 3), (8, 16), (3, 100)] {
            let other = stream(
                &jobs,
                PoolConfig {
                    workers,
                    batch_size,
                    ..Default::default()
                },
            );
            assert_eq!(other.len(), baseline.len());
            for (a, b) in baseline.iter().zip(other.iter()) {
                assert_eq!(a.job_id, b.job_id);
                assert_eq!(a.summary, b.summary, "workers={workers} batch={batch_size}");
                assert_eq!(a.runs, b.runs, "workers={workers} batch={batch_size}");
            }
        }
    }

    #[test]
    fn as_completed_returns_the_same_set_in_any_order() {
        let jobs: Vec<String> = (0..12)
            .map(|i| job_line(&format!("job{i}"), i as u64, 2))
            .collect();

        let ordered = stream(&jobs, PoolConfig::default());
        let unordered = stream(
            &jobs,
            PoolConfig {
                workers: 4,
                ordering: Ordering::AsCompleted,
                ..Default::default()
            },
        );

        assert_eq!(unordered.len(), ordered.len());
        let mut ids: Vec<String> = unordered
            .iter()
            .map(|r| r.job_id.clone().unwrap())
            .collect();
        ids.sort();
        let mut expected: Vec<String> =
            ordered.iter().map(|r| r.job_id.clone().unwrap()).collect();
        expected.sort();
        assert_eq!(ids, expected);
    }

    #[test]
    fn blank_lines_are_skipped() {
        let input = format!("{}\n\n   \n{}\n", job_line("a", 1, 1), job_line("b", 2, 1));
        let mut output = Vec::new();
        let count = run_stream(Cursor::new(input), &mut output, PoolConfig::default()).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn a_bad_job_does_not_stop_the_stream() {
        let input = format!(
            "{}\n{{not json\n{}",
            job_line("good1", 1, 1),
            job_line("good2", 2, 1)
        );
        let mut output = Vec::new();
        run_stream(Cursor::new(input), &mut output, PoolConfig::default()).unwrap();

        let text = String::from_utf8(output).unwrap();
        let results: Vec<JobResult> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        assert_eq!(results.len(), 3);
        assert!(results[0].error.is_none());
        assert!(results[1].error.is_some());
        assert!(results[2].error.is_none());
        // Order is preserved even though the middle job failed fast.
        assert_eq!(results[0].job_id.as_deref(), Some("good1"));
        assert_eq!(results[2].job_id.as_deref(), Some("good2"));
    }

    #[test]
    fn empty_input_produces_no_output() {
        let mut output = Vec::new();
        let count = run_stream(Cursor::new(""), &mut output, PoolConfig::default()).unwrap();
        assert_eq!(count, 0);
        assert!(output.is_empty());
    }

    #[test]
    fn summary_only_drops_per_run_detail() {
        let jobs = vec![job_line("a", 1, 4)];
        let results = stream(
            &jobs,
            PoolConfig {
                summary_only: true,
                ..Default::default()
            },
        );
        assert!(results[0].runs.is_none());
        assert_eq!(results[0].summary.as_ref().unwrap().num_runs, 4);
    }

    #[test]
    fn the_batch_ramp_fills_the_pool_before_it_grows() {
        let (workers, batch_size) = (8, 16);
        // Every worker gets a single-job batch first.
        for n in 0..workers {
            assert_eq!(ramped_batch_size(n, workers, batch_size), 1, "batch {n}");
        }
        // Then the size doubles once per sweep of the pool.
        assert_eq!(ramped_batch_size(workers, workers, batch_size), 2);
        assert_eq!(ramped_batch_size(workers * 2, workers, batch_size), 4);
        assert_eq!(ramped_batch_size(workers * 3, workers, batch_size), 8);
        assert_eq!(ramped_batch_size(workers * 4, workers, batch_size), 16);
        // And never exceeds the configured size.
        assert_eq!(ramped_batch_size(workers * 99, workers, batch_size), 16);
    }

    #[test]
    fn the_batch_ramp_handles_degenerate_settings() {
        assert_eq!(ramped_batch_size(0, 1, 1), 1);
        assert_eq!(ramped_batch_size(1_000_000, 1, 4), 4);
        // A zero worker count must not divide by zero.
        assert_eq!(ramped_batch_size(5, 0, 8), 8);
    }

    #[test]
    fn default_worker_count_is_at_least_one() {
        assert!(default_workers() >= 1);
    }
}
