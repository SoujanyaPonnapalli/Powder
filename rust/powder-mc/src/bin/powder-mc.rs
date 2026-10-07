//! Command line entry point.
//!
//! Jobs arrive as JSON from a file, stdin, or the environment, so changing
//! an input never requires a rebuild.
//!
//! ```text
//! powder-mc --config scenario.json
//! powder-mc --config -                     # read one job from stdin
//! POWDER_MC_CONFIG=scenario.json powder-mc # path or inline JSON
//! powder-mc --stream [-j N] [--batch-size K] [--unordered]
//! ```
//!
//! `--stream` reads one JSON job per stdin line and writes one JSON result
//! per stdout line.  It amortises process startup across a sweep and is
//! where the persistent worker pool applies: each job runs single-threaded
//! on one worker, and workers claim jobs in batches.

// Built with `--features mimalloc`, the binary uses mimalloc's
// per-thread heaps instead of the system allocator.  The engine itself
// allocates about three times per simulation, so this changes nothing for
// most workloads; it is here for the case where many worker threads are
// each churning through short jobs and the system allocator's slow paths
// start to show.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::io::{self, BufReader, Read, Write};
use std::process::ExitCode;

use powder_mc::job::run_job_json;
use powder_mc::pool::{default_workers, run_stream, Ordering, PoolConfig};

const USAGE: &str = "\
powder-mc - Monte Carlo simulator for replicated state machine deployments

USAGE:
    powder-mc --config <PATH|->        Run one JSON job and print the result
    powder-mc --stream [OPTIONS]       Run NDJSON jobs from stdin
    powder-mc                          Same as --config, reading POWDER_MC_CONFIG

OPTIONS:
    -c, --config <PATH|->   Job file, or '-' for stdin
        --stream            Read one JSON job per stdin line, write one
                            JSON result per stdout line
    -j, --jobs <N>          Worker threads in stream mode
                            [default: available parallelism]
        --batch-size <K>    Jobs each worker claims at a time [default: 16]
        --unordered         Emit results as they complete rather than in
                            input order
        --summary-only      Omit per-run detail and emit only aggregates
    -h, --help              Print this message

ENVIRONMENT:
    POWDER_MC_CONFIG    Path to a job file, or inline JSON, used when
                        --config is not given

A single Monte Carlo experiment always runs on one thread.  Parallelism
applies across jobs in stream mode.
";

/// Parsed command line.
struct Args {
    config: Option<String>,
    stream: bool,
    workers: usize,
    batch_size: usize,
    ordering: Ordering,
    summary_only: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            config: None,
            stream: false,
            workers: default_workers(),
            batch_size: 16,
            ordering: Ordering::InputOrder,
            summary_only: false,
        }
    }
}

fn parse_args(argv: &[String]) -> Result<Option<Args>, String> {
    let mut args = Args::default();
    let mut i = 0;

    while i < argv.len() {
        let arg = argv[i].as_str();
        let mut next_value = |name: &str| -> Result<String, String> {
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };

        match arg {
            "-h" | "--help" => return Ok(None),
            "-c" | "--config" => args.config = Some(next_value("--config")?),
            "--stream" => args.stream = true,
            "-j" | "--jobs" => {
                let raw = next_value("--jobs")?;
                args.workers = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--jobs expects a number, got '{raw}'"))?;
                if args.workers == 0 {
                    return Err("--jobs must be at least 1".to_string());
                }
            }
            "--batch-size" => {
                let raw = next_value("--batch-size")?;
                args.batch_size = raw
                    .parse::<usize>()
                    .map_err(|_| format!("--batch-size expects a number, got '{raw}'"))?;
                if args.batch_size == 0 {
                    return Err("--batch-size must be at least 1".to_string());
                }
            }
            "--unordered" => args.ordering = Ordering::AsCompleted,
            "--summary-only" => args.summary_only = true,
            other => return Err(format!("unrecognised argument '{other}'")),
        }
        i += 1;
    }

    Ok(Some(args))
}

/// Resolve the single-job input: explicit flag, then the environment.
///
/// A `POWDER_MC_CONFIG` value is treated as inline JSON when it starts with
/// `{`, and as a path otherwise.
fn read_single_job(config: Option<&str>) -> Result<String, String> {
    let source = match config {
        Some(value) => value.to_string(),
        None => match std::env::var("POWDER_MC_CONFIG") {
            Ok(value) => value,
            Err(_) => {
                return Err(
                    "no job given: pass --config <PATH|->, or set POWDER_MC_CONFIG".to_string(),
                )
            }
        },
    };

    if source == "-" {
        let mut text = String::new();
        io::stdin()
            .read_to_string(&mut text)
            .map_err(|e| format!("failed to read stdin: {e}"))?;
        return Ok(text);
    }

    if source.trim_start().starts_with('{') {
        return Ok(source);
    }

    std::fs::read_to_string(&source).map_err(|e| format!("failed to read {source}: {e}"))
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();

    let args = match parse_args(&argv) {
        Ok(Some(args)) => args,
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("powder-mc: {message}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    if args.stream {
        // The reader runs on its own thread, so it takes the `Stdin` handle
        // rather than a `StdinLock` -- the latter is not `Send`.  Buffering
        // is large enough that the per-read internal lock costs nothing.
        let reader = BufReader::with_capacity(1 << 20, io::stdin());
        let stdout = io::stdout();
        let mut writer = io::BufWriter::with_capacity(1 << 20, stdout.lock());

        let config = PoolConfig {
            workers: args.workers,
            batch_size: args.batch_size,
            ordering: args.ordering,
            summary_only: args.summary_only,
        };

        return match run_stream(reader, &mut writer, config) {
            Ok(_) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("powder-mc: stream failed: {e}");
                ExitCode::FAILURE
            }
        };
    }

    let text = match read_single_job(args.config.as_deref()) {
        Ok(text) => text,
        Err(message) => {
            eprintln!("powder-mc: {message}");
            return ExitCode::from(2);
        }
    };

    let mut result = run_job_json(&text);
    if args.summary_only {
        result = powder_mc::job::strip_runs(result);
    }
    let failed = result.error.is_some();

    let encoded = match serde_json::to_string(&result) {
        Ok(encoded) => encoded,
        Err(e) => {
            eprintln!("powder-mc: failed to encode result: {e}");
            return ExitCode::FAILURE;
        }
    };

    let stdout = io::stdout();
    let mut handle = stdout.lock();
    if writeln!(handle, "{encoded}").is_err() || handle.flush().is_err() {
        return ExitCode::FAILURE;
    }

    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
