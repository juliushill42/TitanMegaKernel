use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use titanmk::ir::Schedule;
use titanmk::validator::validate;

#[derive(Parser)]
#[command(
    name = "titanmk",
    version,
    about = "Titan Hardware-Agnostic Megakernel Schedule Validator"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate a schedule JSON file: deadlock, race, SM-order, and
    /// recursive sub-schedule checks. Exits non-zero on failure.
    Validate {
        /// Path to schedule JSON file.
        path: PathBuf,
        /// Print the full certificate (interface + fingerprint) on success.
        #[arg(long)]
        verbose: bool,
    },
    /// Print the SHA-256 fingerprint of a schedule without full validation.
    Fingerprint { path: PathBuf },
    /// Run the built-in good/bad demo schedules and report results.
    Demo,
    /// Validate a schedule, then generate CUDA source (.cu) for it.
    /// Fails (and writes nothing) if validation fails -- only
    /// certified schedules are ever code-generated.
    GenCuda {
        /// Path to schedule JSON file.
        path: PathBuf,
        /// Output path for generated .cu source.
        #[arg(short, long, default_value = "generated_megakernel.cu")]
        out: PathBuf,
        /// Uniform per-tensor element count for the demo TensorSet.
        #[arg(long, default_value_t = 8)]
        tensor_len: u64,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Validate { path, verbose } => cmd_validate(&path, verbose),
        Commands::Fingerprint { path } => cmd_fingerprint(&path),
        Commands::Demo => cmd_demo(),
        Commands::GenCuda {
            path,
            out,
            tensor_len,
        } => cmd_gen_cuda(&path, &out, tensor_len),
    }
}

fn load_schedule(path: &PathBuf) -> Result<Schedule, String> {
    let data = fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;
    serde_json::from_str(&data).map_err(|e| format!("parse error: {e}"))
}

fn cmd_validate(path: &PathBuf, verbose: bool) -> ExitCode {
    let schedule = match load_schedule(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[titanmk] ERROR loading '{}': {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    match validate(&schedule) {
        Ok(cert) => {
            println!("[titanmk] CERTIFIED OK: '{}'", cert.schedule_name);
            println!("  nodes (recursive):  {}", cert.node_count);
            println!("  fingerprint:        {}", cert.fingerprint);
            if verbose {
                println!("  interface:");
                let mut keys: Vec<_> = cert.interface.keys().collect();
                keys.sort();
                for k in keys {
                    let acc = &cert.interface[k];
                    println!(
                        "    {:<24} read={:<5} written={:<5} space={:?}",
                        k, acc.read, acc.written, acc.space
                    );
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[titanmk] VALIDATION FAILED: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_fingerprint(path: &PathBuf) -> ExitCode {
    match load_schedule(path) {
        Ok(schedule) => {
            println!("{}", titanmk::validator::fingerprint_schedule(&schedule));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[titanmk] ERROR loading '{}': {e}", path.display());
            ExitCode::FAILURE
        }
    }
}

fn cmd_demo() -> ExitCode {
    use titanmk::adversarial::*;

    let cases: Vec<(&str, Schedule, bool)> = vec![
        ("good_linear", good_linear(), true),
        (
            "good_nested_repeated_block",
            good_nested_repeated_block(),
            true,
        ),
        ("bad_cycle", bad_cycle(), false),
        ("bad_dangling_wait", bad_dangling_wait(), false),
        ("bad_sm_conflict", bad_sm_conflict(), false),
        ("bad_race_diff_sm", bad_race_diff_sm(), false),
        ("bad_nested_cycle", bad_nested_cycle(), false),
    ];

    let mut all_ok = true;
    for (name, schedule, expect_ok) in cases {
        let result = validate(&schedule);
        let actual_ok = result.is_ok();
        let status = if actual_ok == expect_ok {
            "PASS"
        } else {
            "FAIL"
        };
        if actual_ok != expect_ok {
            all_ok = false;
        }
        match result {
            Ok(cert) => println!(
                "[{status}] {name:<28} -> CERTIFIED ({} nodes, fp={}...)",
                cert.node_count,
                &cert.fingerprint[..12]
            ),
            Err(e) => println!("[{status}] {name:<28} -> REJECTED ({e})"),
        }
    }

    if all_ok {
        println!("\n[titanmk] all demo cases behaved as expected.");
        ExitCode::SUCCESS
    } else {
        println!("\n[titanmk] one or more demo cases did NOT behave as expected.");
        ExitCode::FAILURE
    }
}

fn cmd_gen_cuda(path: &PathBuf, out: &PathBuf, tensor_len: u64) -> ExitCode {
    let schedule = match load_schedule(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[titanmk] ERROR loading '{}': {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    let cert = match validate(&schedule) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[titanmk] VALIDATION FAILED, refusing to generate CUDA: {e}");
            return ExitCode::FAILURE;
        }
    };

    let (src, fields) = match titanmk::cuda_codegen::generate_with_fields(&schedule, tensor_len) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[titanmk] CUDA codegen failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = fs::write(out, src) {
        eprintln!("[titanmk] failed to write '{}': {e}", out.display());
        return ExitCode::FAILURE;
    }

    let fields_path = out.with_file_name("tensor_fields.inc");
    if let Err(e) = fs::write(&fields_path, fields) {
        eprintln!("[titanmk] failed to write '{}': {e}", fields_path.display());
        return ExitCode::FAILURE;
    }

    println!(
        "[titanmk] CERTIFIED '{}' (fp={}) -> {}",
        cert.schedule_name,
        &cert.fingerprint[..12],
        out.display()
    );
    ExitCode::SUCCESS
}
