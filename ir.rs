//! Recursive Schedule IR.
//!
//! A `Schedule` is a DAG of `Node`s. A `Node` is either a primitive `Op`
//! (executed on hardware) or a `SubSchedule` (a nested `Schedule`),
//! making the IR recursive: schedules contain schedules.
//!
//! This recursion is the basis for hierarchical fusion: an entire
//! transformer block can be a single Node in the top-level schedule,
//! while internally being its own validated Schedule.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub type NodeId = u32;
pub type SmId = u32; // Streaming-multiprocessor / compute-unit id, hardware agnostic

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemSpace {
    Hbm,
    SharedL1,
    Register,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub space: MemSpace,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Copy)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
}

/// Primitive operations. Backend codegens (Zig/CUDA/etc) map each
/// variant to a target-specific kernel fragment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OpKind {
    MatMul { m: u64, n: u64, k: u64 },
    RmsNorm { dim: u64 },
    Softmax { dim: u64 },
    Attention { heads: u64, head_dim: u64, seq_len: u64 },
    Elementwise { op: String },
    Barrier,
}

/// A single executable unit. May be a primitive `Op` or a recursive
/// `SubSchedule`. Either way it has inputs/outputs and a target SM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub inputs: Vec<Tensor>,
    pub outputs: Vec<Tensor>,
    /// SM/compute-unit this node is pinned to. Required for the
    /// per-SM queue-order check.
    pub sm: SmId,
    /// Node ids this node must wait on before launching.
    pub waits_on: Vec<NodeId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    Op(OpKind),
    /// Recursive: this node is itself a fully-formed Schedule.
    SubSchedule(Box<Schedule>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub name: String,
    pub nodes: Vec<Node>,
}

impl Schedule {
    pub fn new(name: impl Into<String>) -> Self {
        Schedule { name: name.into(), nodes: Vec::new() }
    }

    pub fn push(&mut self, node: Node) {
        self.nodes.push(node);
    }

    /// Recursively flattens this schedule and all nested SubSchedules
    /// into a single list of (path, &Node) pairs. `path` is the chain
    /// of parent node ids, used for globally-unique addressing during
    /// validation.
    pub fn flatten(&self) -> Vec<(Vec<NodeId>, &Node)> {
        let mut out = Vec::new();
        self.flatten_into(&mut Vec::new(), &mut out);
        out
    }

    fn flatten_into<'a>(&'a self, path: &mut Vec<NodeId>, out: &mut Vec<(Vec<NodeId>, &'a Node)>) {
        for node in &self.nodes {
            out.push((path.clone(), node));
            if let NodeKind::SubSchedule(sub) = &node.kind {
                path.push(node.id);
                sub.flatten_into(path, out);
                path.pop();
            }
        }
    }

    /// Map of node id -> node for this schedule level only (non-recursive).
    pub fn index(&self) -> HashMap<NodeId, &Node> {
        self.nodes.iter().map(|n| (n.id, n)).collect()
    }

    /// All distinct SM ids referenced anywhere in this schedule, recursively.
    pub fn all_sms(&self) -> HashSet<SmId> {
        self.flatten().into_iter().map(|(_, n)| n.sm).collect()
    }
}
