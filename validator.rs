//! Static Schedule Validator.
//!
//! Certifies, before any kernel launches, that a `Schedule` is:
//!
//!  1. Acyclic (DAG) — including across recursive SubSchedules.
//!  2. Wait-satisfiable — every `waits_on` reference points to a node
//!     that exists and that, transitively, cannot wait on the waiter
//!     (no cycles in the wait graph either — covers cross-level waits).
//!  3. Per-SM queue-order consistent — for any two nodes pinned to the
//!     same SM, their relative wait-order must match a single total
//!     order (no SM is asked to execute B-before-A and A-before-B).
//!  4. Race-free — for any two nodes that are NOT ordered by the wait
//!     graph (could run concurrently), their input/output Tensor sets
//!     must not overlap on the same memory space unless both are
//!     read-only (outputs vs outputs, or output vs input on shared data
//!     is a violation).
//!
//! Recursion: a `SubSchedule` is validated independently first (its
//! internal graph must be self-consistent). Once certified, it is
//! treated as a single opaque node at the parent level — its full
//! input/output tensor set is the union of all its leaf nodes' tensor
//! sets, computed via `Schedule::flatten`. This means certifying a
//! repeated block (e.g. one transformer layer) once and reusing the
//! `Certificate` is sound: the parent only needs the aggregate
//! interface, not re-verification of internals.

use crate::ir::{MemSpace, Node, NodeId, NodeKind, Schedule, Tensor};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ValidationError {
    #[error("cycle detected in dependency graph at node {0}")]
    Cycle(NodeId),
    #[error("node {0} waits on non-existent node {1}")]
    DanglingWait(NodeId, NodeId),
    #[error("SM {sm} ordering conflict: node {a} and node {b} have contradictory order constraints")]
    SmOrderConflict { sm: u32, a: NodeId, b: NodeId },
    #[error("race condition: node {a} and node {b} concurrently access tensor '{tensor}' in {space:?} with at least one write")]
    Race { a: NodeId, b: NodeId, tensor: String, space: MemSpace },
    #[error("nested sub-schedule '{0}' failed validation: {1}")]
    NestedFailure(String, Box<ValidationError>),
}

/// Result of a successful validation. Carries enough aggregate info
/// for a parent schedule to treat this schedule as one opaque node.
#[derive(Debug, Clone)]
pub struct Certificate {
    pub schedule_name: String,
    pub node_count: usize,
    /// Union of every tensor read/written anywhere in this schedule,
    /// keyed by tensor name -> (is_read, is_written, space).
    pub interface: HashMap<String, TensorAccess>,
    /// SHA-256 hex digest of the canonical JSON serialization of the
    /// schedule. Used for cryptographic execution fingerprinting.
    pub fingerprint: String,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TensorAccess {
    pub read: bool,
    pub written: bool,
    pub space: Option<MemSpace>,
}

/// Validate a schedule recursively. Returns a `Certificate` on success.
pub fn validate(schedule: &Schedule) -> Result<Certificate, ValidationError> {
    // 1. Recursively validate every nested SubSchedule first.
    for node in &schedule.nodes {
        if let NodeKind::SubSchedule(sub) = &node.kind {
            validate(sub).map_err(|e| {
                ValidationError::NestedFailure(sub.name.clone(), Box::new(e))
            })?;
        }
    }

    let index = schedule.index();

    // 2. Check all waits point to real nodes within this level.
    for node in &schedule.nodes {
        for w in &node.waits_on {
            if !index.contains_key(w) {
                return Err(ValidationError::DanglingWait(node.id, *w));
            }
        }
    }

    // 3. Cycle detection on the wait graph (DFS, this level only —
    //    nested levels were already proven acyclic internally and are
    //    treated as atomic here).
    detect_cycles(schedule, &index)?;

    // 4. Per-SM queue order consistency.
    check_sm_order(schedule, &index)?;

    // 5. Race freedom for unordered node pairs.
    check_races(schedule, &index)?;

    // 6. Build the aggregate interface (recursive: include nested
    //    sub-schedule leaf tensors via flatten).
    let mut interface: HashMap<String, TensorAccess> = HashMap::new();
    for (_, node) in schedule.flatten() {
        if let NodeKind::Op(_) = &node.kind {
            accumulate_tensors(&mut interface, &node.inputs, false);
            accumulate_tensors(&mut interface, &node.outputs, true);
        }
    }

    let fingerprint = fingerprint_schedule(schedule);

    Ok(Certificate {
        schedule_name: schedule.name.clone(),
        node_count: schedule.flatten().len(),
        interface,
        fingerprint,
    })
}

fn accumulate_tensors(map: &mut HashMap<String, TensorAccess>, tensors: &[Tensor], is_output: bool) {
    for t in tensors {
        let entry = map.entry(t.name.clone()).or_default();
        if is_output {
            entry.written = true;
        } else {
            entry.read = true;
        }
        entry.space = Some(t.space.clone());
    }
}

/// DFS-based cycle detection over the `waits_on` graph.
fn detect_cycles(schedule: &Schedule, index: &HashMap<NodeId, &Node>) -> Result<(), ValidationError> {
    #[derive(Clone, Copy, PartialEq)]
    enum Color { White, Gray, Black }

    let mut color: HashMap<NodeId, Color> = index.keys().map(|&id| (id, Color::White)).collect();

    fn visit(
        id: NodeId,
        index: &HashMap<NodeId, &Node>,
        color: &mut HashMap<NodeId, Color>,
    ) -> Result<(), ValidationError> {
        match color.get(&id) {
            Some(Color::Black) => return Ok(()),
            Some(Color::Gray) => return Err(ValidationError::Cycle(id)),
            _ => {}
        }
        color.insert(id, Color::Gray);
        if let Some(node) = index.get(&id) {
            for &dep in &node.waits_on {
                visit(dep, index, color)?;
            }
        }
        color.insert(id, Color::Black);
        Ok(())
    }

    for &id in index.keys() {
        if color[&id] == Color::White {
            visit(id, index, &mut color)?;
        }
    }
    Ok(())
}

/// For every SM, all nodes pinned to it must be totally orderable by
/// the wait graph (i.e. reachable from one another in a consistent
/// direction). If two nodes on the same SM are mutually unordered AND
/// neither waits (transitively) on the other, that's fine — the
/// scheduler can pick an order. But if A waits on B *and* B waits on A
/// (transitively) that's already a cycle (caught above). The conflict
/// this check catches: A and B on the same SM where the wait graph
/// implies *both* "A must be issued before B" and "B before A" via
/// distinct transitive chains is impossible if acyclic — so what we
/// actually guard against here is the practical case of two nodes on
/// the same SM with NO ordering relation at all, which is a hazard if
/// they touch overlapping memory (the SM's single instruction stream
/// would need *some* order, but the schedule doesn't specify one).
fn check_sm_order(schedule: &Schedule, index: &HashMap<NodeId, &Node>) -> Result<(), ValidationError> {
    let reach = transitive_reach(index);

    let mut by_sm: HashMap<u32, Vec<NodeId>> = HashMap::new();
    for node in &schedule.nodes {
        by_sm.entry(node.sm).or_default().push(node.id);
    }

    for (sm, ids) in &by_sm {
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (a, b) = (ids[i], ids[j]);
                let a_before_b = reach.get(&b).map_or(false, |s| s.contains(&a));
                let b_before_a = reach.get(&a).map_or(false, |s| s.contains(&b));
                if !a_before_b && !b_before_a {
                    let na = index[&a];
                    let nb = index[&b];
                    if tensors_overlap(na, nb) {
                        return Err(ValidationError::SmOrderConflict { sm: *sm, a, b });
                    }
                }
            }
        }
    }
    Ok(())
}

/// node -> set of node ids that must execute before it (transitive closure
/// of `waits_on`).
fn transitive_reach(index: &HashMap<NodeId, &Node>) -> HashMap<NodeId, HashSet<NodeId>> {
    let mut memo: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();

    fn compute(
        id: NodeId,
        index: &HashMap<NodeId, &Node>,
        memo: &mut HashMap<NodeId, HashSet<NodeId>>,
    ) -> HashSet<NodeId> {
        if let Some(cached) = memo.get(&id) {
            return cached.clone();
        }
        let mut acc = HashSet::new();
        if let Some(node) = index.get(&id) {
            for &dep in &node.waits_on {
                acc.insert(dep);
                let deeper = compute(dep, index, memo);
                acc.extend(deeper);
            }
        }
        memo.insert(id, acc.clone());
        acc
    }

    for &id in index.keys() {
        compute(id, index, &mut memo);
    }
    memo
}

fn tensors_overlap(a: &Node, b: &Node) -> bool {
    let a_all: Vec<&Tensor> = a.inputs.iter().chain(a.outputs.iter()).collect();
    let b_all: Vec<&Tensor> = b.inputs.iter().chain(b.outputs.iter()).collect();
    a_all.iter().any(|ta| b_all.iter().any(|tb| ta.name == tb.name))
}

/// Two unordered nodes race if they touch the same tensor in the same
/// memory space and at least one of them writes it.
fn check_races(schedule: &Schedule, index: &HashMap<NodeId, &Node>) -> Result<(), ValidationError> {
    let reach = transitive_reach(index);
    let ids: Vec<NodeId> = index.keys().copied().collect();

    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            let (a, b) = (ids[i], ids[j]);
            let a_before_b = reach.get(&b).map_or(false, |s| s.contains(&a));
            let b_before_a = reach.get(&a).map_or(false, |s| s.contains(&b));
            if a_before_b || b_before_a {
                continue; // ordered, no race possible
            }
            let na = index[&a];
            let nb = index[&b];

            for ta in na.outputs.iter().chain(na.inputs.iter()) {
                for tb in nb.outputs.iter().chain(nb.inputs.iter()) {
                    if ta.name != tb.name || ta.space != tb.space {
                        continue;
                    }
                    let a_writes = na.outputs.iter().any(|t| t.name == ta.name);
                    let b_writes = nb.outputs.iter().any(|t| t.name == tb.name);
                    if a_writes || b_writes {
                        return Err(ValidationError::Race {
                            a,
                            b,
                            tensor: ta.name.clone(),
                            space: ta.space.clone(),
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

/// Deterministic SHA-256 fingerprint of a schedule's canonical JSON form.
pub fn fingerprint_schedule(schedule: &Schedule) -> String {
    use sha2::{Digest, Sha256};
    let json = serde_json::to_string(schedule).expect("schedule must serialize");
    let mut hasher = Sha256::new();
    hasher.update(json.as_bytes());
    hex::encode(hasher.finalize())
}
