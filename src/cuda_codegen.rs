//! CUDA codegen: walks a certified `Schedule` (recursive, validator
//! already proved deadlock/race freedom) and emits a single `.cu`
//! source file containing `run_megakernel(...)`, which dispatches the
//! ops in `titan_kernels.cuh` in topological order on one CUDA stream.
//!
//! Recursion mirrors zig codegen: SubSchedule nodes are inlined at the
//! call site by recursively emitting their topo-ordered body, so
//! repeated blocks (transformer layers) become sequential inlined
//! regions of the same launch sequence.
//!
//! HONEST SCOPE NOTE: this emits sequential kernel launches on one
//! stream (no host sync between them), removing host round-trip
//! overhead per AMK's stated bottleneck. It does not yet fuse
//! everything into one cooperative-groups megakernel grid -- that is
//! a follow-on once this sequential version is validated end-to-end
//! on real GPU hardware.

use crate::ir::{Node, NodeKind, OpKind, Schedule};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaCodegenError {
    #[error("cycle detected during topological sort of schedule '{0}'")]
    Cycle(String),
    #[error("op node {0} has unexpected arity for {1:?}")]
    BadArity(u32, OpKind),
}

/// Topologically sort `schedule.nodes` by `waits_on` (Kahn's algorithm),
/// deterministic (ties broken by ascending node id).
fn topo_sort(schedule: &Schedule) -> Result<Vec<&Node>, CudaCodegenError> {
    let index = schedule.index();
    let mut in_degree: HashMap<u32, usize> = index.keys().map(|&id| (id, 0)).collect();
    for node in &schedule.nodes {
        *in_degree.get_mut(&node.id).unwrap() = node.waits_on.len();
    }

    let mut dependents: HashMap<u32, Vec<u32>> = HashMap::new();
    for node in &schedule.nodes {
        for &dep in &node.waits_on {
            dependents.entry(dep).or_default().push(node.id);
        }
    }
    for v in dependents.values_mut() {
        v.sort_unstable();
    }

    let mut queue: Vec<u32> = in_degree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(&id, _)| id)
        .collect();
    queue.sort_unstable();

    let mut out = Vec::with_capacity(schedule.nodes.len());
    let mut head = 0;
    while head < queue.len() {
        let id = queue[head];
        head += 1;
        out.push(index[&id]);
        if let Some(deps) = dependents.get(&id) {
            for &dep_id in deps {
                let d = in_degree.get_mut(&dep_id).unwrap();
                *d -= 1;
                if *d == 0 {
                    let pos = queue[head..].binary_search(&dep_id).unwrap_or_else(|e| e) + head;
                    queue.insert(pos, dep_id);
                }
            }
        }
    }

    if out.len() != schedule.nodes.len() {
        return Err(CudaCodegenError::Cycle(schedule.name.clone()));
    }
    Ok(out)
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn elementwise_op_code(op: &str) -> i32 {
    match op {
        "id" => 0,
        "relu" => 1,
        "gelu" => 2,
        "silu" => 3,
        _ => 0,
    }
}

fn emit_op(
    out: &mut String,
    node: &Node,
    op: &OpKind,
    indent: &str,
) -> Result<(), CudaCodegenError> {
    match op {
        OpKind::MatMul { m, n, k } => {
            if node.outputs.is_empty() || node.inputs.len() < 2 {
                return Err(CudaCodegenError::BadArity(node.id, op.clone()));
            }
            let o = sanitize(&node.outputs[0].name);
            let a = sanitize(&node.inputs[0].name);
            let b = sanitize(&node.inputs[1].name);
            let _ = writeln!(
                out,
                "{indent}titan_matmul({o}, {a}, {b}, {m}, {n}, {k}, stream);"
            );
        }
        OpKind::RmsNorm { dim } => {
            if node.outputs.is_empty() || node.inputs.is_empty() {
                return Err(CudaCodegenError::BadArity(node.id, op.clone()));
            }
            let o = sanitize(&node.outputs[0].name);
            let i = sanitize(&node.inputs[0].name);
            let _ = writeln!(
                out,
                "{indent}titan_rmsnorm({o}, {i}, /*rows=*/1, {dim}, stream);"
            );
        }
        OpKind::Softmax { dim } => {
            if node.outputs.is_empty() || node.inputs.is_empty() {
                return Err(CudaCodegenError::BadArity(node.id, op.clone()));
            }
            let o = sanitize(&node.outputs[0].name);
            let i = sanitize(&node.inputs[0].name);
            let _ = writeln!(
                out,
                "{indent}titan_softmax({o}, {i}, /*rows=*/1, {dim}, stream);"
            );
        }
        OpKind::Attention {
            heads,
            head_dim,
            seq_len,
        } => {
            if node.outputs.is_empty() || node.inputs.len() < 3 {
                return Err(CudaCodegenError::BadArity(node.id, op.clone()));
            }
            let o = sanitize(&node.outputs[0].name);
            let q = sanitize(&node.inputs[0].name);
            let k = sanitize(&node.inputs[1].name);
            let v = sanitize(&node.inputs[2].name);
            let _ = writeln!(
                out,
                "{indent}titan_attention({o}, {q}, {k}, {v}, {heads}, {head_dim}, {seq_len}, stream);"
            );
        }
        OpKind::Elementwise { op: name } => {
            if node.outputs.is_empty() {
                return Err(CudaCodegenError::BadArity(node.id, op.clone()));
            }
            let o = sanitize(&node.outputs[0].name);
            if node.inputs.is_empty() {
                let _ = writeln!(
                    out,
                    "{indent}// node {}: source op '{}' on '{o}' -- assumed pre-populated by host, no-op here",
                    node.id, name
                );
            } else {
                let i = sanitize(&node.inputs[0].name);
                let code = elementwise_op_code(name);
                let _ = writeln!(
                    out,
                    "{indent}titan_elementwise({o}, {i}, /*len=*/TITAN_TENSOR_LEN, /*op=*/{code}, stream); // {name}"
                );
            }
        }
        OpKind::Barrier => {
            let _ = writeln!(out, "{indent}titan_barrier(stream);");
        }
    }
    Ok(())
}

fn emit_schedule(
    out: &mut String,
    schedule: &Schedule,
    depth: usize,
) -> Result<(), CudaCodegenError> {
    let indent = "    ".repeat(depth + 1);
    let ordered = topo_sort(schedule)?;

    if depth > 0 {
        let _ = writeln!(
            out,
            "{}{{ // sub-schedule: {}",
            "    ".repeat(depth),
            schedule.name
        );
    }

    for node in ordered {
        match &node.kind {
            NodeKind::Op(op) => {
                let _ = writeln!(out, "{indent}// node {} (sm {})", node.id, node.sm);
                emit_op(out, node, op, &indent)?;
            }
            NodeKind::SubSchedule(sub) => {
                let _ = writeln!(
                    out,
                    "{indent}// node {} -> inlined sub-schedule '{}'",
                    node.id, sub.name
                );
                emit_schedule(out, sub, depth + 1)?;
            }
        }
    }

    if depth > 0 {
        let _ = writeln!(out, "{}}}", "    ".repeat(depth));
    }
    Ok(())
}

fn collect_tensors(schedule: &Schedule, names: &mut HashSet<String>) {
    for node in &schedule.nodes {
        match &node.kind {
            NodeKind::Op(_) => {
                for t in node.inputs.iter().chain(node.outputs.iter()) {
                    names.insert(sanitize(&t.name));
                }
            }
            NodeKind::SubSchedule(sub) => collect_tensors(sub, names),
        }
    }
}

/// Generate both the `.cu` source and a header listing TensorSet
/// field pointers (`{&t.x0, &t.x1, ...}`), for the host harness to
/// allocate/iterate without hand-listing tensor names.
pub fn generate_with_fields(
    schedule: &Schedule,
    tensor_len: u64,
) -> Result<(String, String), CudaCodegenError> {
    let cu = generate(schedule, tensor_len)?;

    let mut names: HashSet<String> = HashSet::new();
    collect_tensors(schedule, &mut names);
    let mut sorted_names: Vec<&String> = names.iter().collect();
    sorted_names.sort();

    let mut fields = String::new();
    let _ = writeln!(
        fields,
        "// AUTO-GENERATED: TensorSet field pointers, in declaration order."
    );
    for name in &sorted_names {
        let _ = writeln!(fields, "&t.{name},");
    }
    Ok((cu, fields))
}

/// Generate the full `.cu` source for a certified schedule.
///
/// `tensor_len` is a uniform per-tensor element count for the demo
/// harness. Production use carries per-tensor shapes from `Tensor.shape`
/// into per-call sizes; documented as the next iteration's scope.
pub fn generate(schedule: &Schedule, tensor_len: u64) -> Result<String, CudaCodegenError> {
    let mut names: HashSet<String> = HashSet::new();
    collect_tensors(schedule, &mut names);
    let mut sorted_names: Vec<&String> = names.iter().collect();
    sorted_names.sort();

    let mut out = String::new();
    let _ = writeln!(
        out,
        "// AUTO-GENERATED by titanmk gen-cuda. DO NOT EDIT BY HAND."
    );
    let _ = writeln!(out, "// Source schedule: {}", schedule.name);
    let _ = writeln!(out, "#include \"titan_kernels.cuh\"");
    let _ = writeln!(out, "#include <cstdio>");
    let _ = writeln!(out);
    let _ = writeln!(out, "#define TITAN_TENSOR_LEN {tensor_len}ULL");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "inline void titan_zero_init(float* ptr, uint64_t len, cudaStream_t stream) {{"
    );
    let _ = writeln!(
        out,
        "    cudaMemsetAsync(ptr, 0, len * sizeof(float), stream);"
    );
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "// TensorSet: device pointers for every tensor in the schedule's"
    );
    let _ = writeln!(
        out,
        "// aggregate interface. Caller (host) allocates with cudaMalloc."
    );
    let _ = writeln!(out, "struct TensorSet {{");
    for name in &sorted_names {
        let _ = writeln!(out, "    float* {name};");
    }
    let _ = writeln!(out, "}};");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "void run_megakernel(const TensorSet& t, cudaStream_t stream) {{"
    );
    for name in &sorted_names {
        let _ = writeln!(out, "    float* {name} = t.{name};");
    }
    let _ = writeln!(out);

    emit_schedule(&mut out, schedule, 0)?;

    let _ = writeln!(out, "}}");
    Ok(out)
}
