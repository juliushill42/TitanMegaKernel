//! Adversarial schedule generator.
//!
//! Recursively generates schedules — including nested SubSchedules —
//! across categories of known hazards, and asserts the validator
//! rejects every bad one and accepts every good one. This is the
//! deterministic analog of AMK's 7,160-schedule fuzz corpus: instead
//! of random fuzzing, every category is enumerated exactly, so the
//! suite is reproducible and CI-stable.

use crate::ir::*;
use crate::validator::{validate, ValidationError};

fn t(name: &str, space: MemSpace) -> Tensor {
    Tensor {
        name: name.into(),
        shape: vec![1],
        dtype: DType::F32,
        space,
    }
}

fn op_node(id: NodeId, sm: SmId, waits: &[NodeId], ins: Vec<Tensor>, outs: Vec<Tensor>) -> Node {
    Node {
        id,
        kind: NodeKind::Op(OpKind::Elementwise { op: "id".into() }),
        inputs: ins,
        outputs: outs,
        sm,
        waits_on: waits.to_vec(),
    }
}

/// A clean linear pipeline: A -> B -> C, all on different SMs, each
/// writing a distinct tensor and reading the prior one. Should pass.
pub fn good_linear() -> Schedule {
    let mut s = Schedule::new("good_linear");
    s.push(op_node(0, 0, &[], vec![], vec![t("x0", MemSpace::Hbm)]));
    s.push(op_node(
        1,
        1,
        &[0],
        vec![t("x0", MemSpace::Hbm)],
        vec![t("x1", MemSpace::Hbm)],
    ));
    s.push(op_node(
        2,
        2,
        &[1],
        vec![t("x1", MemSpace::Hbm)],
        vec![t("x2", MemSpace::Hbm)],
    ));
    s
}

/// Direct cycle: 0 waits on 1, 1 waits on 0.
pub fn bad_cycle() -> Schedule {
    let mut s = Schedule::new("bad_cycle");
    s.push(op_node(0, 0, &[1], vec![], vec![]));
    s.push(op_node(1, 1, &[0], vec![], vec![]));
    s
}

/// Node references a wait id that doesn't exist.
pub fn bad_dangling_wait() -> Schedule {
    let mut s = Schedule::new("bad_dangling_wait");
    s.push(op_node(0, 0, &[99], vec![], vec![]));
    s
}

/// Two unordered nodes on the same SM both write tensor "shared" —
/// SM order conflict (also a race, but caught first as SM conflict
/// since they're co-located).
pub fn bad_sm_conflict() -> Schedule {
    let mut s = Schedule::new("bad_sm_conflict");
    s.push(op_node(
        0,
        0,
        &[],
        vec![],
        vec![t("shared", MemSpace::SharedL1)],
    ));
    s.push(op_node(
        1,
        0,
        &[],
        vec![],
        vec![t("shared", MemSpace::SharedL1)],
    ));
    s
}

/// Two unordered nodes on *different* SMs both write the same HBM
/// tensor with no dependency edge — pure race, not an SM conflict.
pub fn bad_race_diff_sm() -> Schedule {
    let mut s = Schedule::new("bad_race_diff_sm");
    s.push(op_node(0, 0, &[], vec![], vec![t("shared", MemSpace::Hbm)]));
    s.push(op_node(1, 1, &[], vec![], vec![t("shared", MemSpace::Hbm)]));
    s
}

/// Recursive: a SubSchedule whose *internal* graph has a cycle. Outer
/// schedule is otherwise fine. Must surface as NestedFailure.
pub fn bad_nested_cycle() -> Schedule {
    let inner = bad_cycle();
    let mut outer = Schedule::new("outer_with_bad_inner");
    outer.push(Node {
        id: 0,
        kind: NodeKind::SubSchedule(Box::new(inner)),
        inputs: vec![],
        outputs: vec![],
        sm: 0,
        waits_on: vec![],
    });
    outer
}

/// Recursive good case: outer schedule contains a valid inner
/// SubSchedule (e.g. one transformer block), referenced twice as
/// separate nodes (simulating two layers), with proper outer-level
/// ordering between them via a barrier-like wait.
pub fn good_nested_repeated_block() -> Schedule {
    let block = good_linear();
    let mut outer = Schedule::new("outer_two_layers");
    outer.push(Node {
        id: 100,
        kind: NodeKind::SubSchedule(Box::new(block.clone())),
        inputs: vec![t("layer_in", MemSpace::Hbm)],
        outputs: vec![t("layer0_out", MemSpace::Hbm)],
        sm: 0,
        waits_on: vec![],
    });
    outer.push(Node {
        id: 101,
        kind: NodeKind::SubSchedule(Box::new(block)),
        inputs: vec![t("layer0_out", MemSpace::Hbm)],
        outputs: vec![t("layer1_out", MemSpace::Hbm)],
        sm: 0,
        waits_on: vec![100],
    });
    outer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn good_linear_passes() {
        let cert = validate(&good_linear()).expect("should validate");
        assert_eq!(cert.node_count, 3);
        assert!(!cert.fingerprint.is_empty());
    }

    #[test]
    fn cycle_rejected() {
        let err = validate(&bad_cycle()).unwrap_err();
        assert!(matches!(err, ValidationError::Cycle(_)));
    }

    #[test]
    fn dangling_wait_rejected() {
        let err = validate(&bad_dangling_wait()).unwrap_err();
        assert_eq!(err, ValidationError::DanglingWait(0, 99));
    }

    #[test]
    fn sm_conflict_rejected() {
        let err = validate(&bad_sm_conflict()).unwrap_err();
        assert!(matches!(err, ValidationError::SmOrderConflict { .. }));
    }

    #[test]
    fn cross_sm_race_rejected() {
        let err = validate(&bad_race_diff_sm()).unwrap_err();
        assert!(matches!(err, ValidationError::Race { .. }));
    }

    #[test]
    fn nested_cycle_surfaces_as_nested_failure() {
        let err = validate(&bad_nested_cycle()).unwrap_err();
        match err {
            ValidationError::NestedFailure(name, inner) => {
                assert_eq!(name, "bad_cycle");
                assert!(matches!(*inner, ValidationError::Cycle(_)));
            }
            other => panic!("expected NestedFailure, got {other:?}"),
        }
    }

    #[test]
    fn nested_repeated_block_passes_and_aggregates_interface() {
        let cert = validate(&good_nested_repeated_block()).expect("should validate");
        // 2 SubSchedule nodes, each flattening to 3 leaf nodes = 6 leaves
        // plus the 2 wrapper nodes counted by flatten() at the outer level too.
        assert!(cert.node_count >= 6);
        // Aggregate interface should include leaf tensors from both inner blocks.
        assert!(cert.interface.contains_key("x0"));
        assert!(cert.interface.contains_key("x2"));
    }

    #[test]
    fn fingerprints_are_deterministic_and_distinct() {
        let f1 = crate::validator::fingerprint_schedule(&good_linear());
        let f2 = crate::validator::fingerprint_schedule(&good_linear());
        let f3 = crate::validator::fingerprint_schedule(&bad_cycle());
        assert_eq!(f1, f2);
        assert_ne!(f1, f3);
    }
    #[test]
    fn cuda_codegen_good_linear_emits_expected_calls() {
        let src = crate::cuda_codegen::generate(&good_linear(), 8).expect("codegen ok");
        assert!(src.contains("struct TensorSet"));
        assert!(src.contains("void run_megakernel"));
        // node1 = relu(x0) -> x1, node2 = gelu(x1) -> x2
        assert!(src.contains("titan_elementwise(x1, x0"));
        assert!(src.contains("titan_elementwise(x2, x1"));
        // source node (node 0) must NOT clobber x0
        assert!(!src.contains("titan_zero_init(x0"));
    }

    #[test]
    fn cuda_codegen_rejects_uncertified_schedule() {
        // cuda_codegen itself doesn't validate; the CLI does. Here we
        // just confirm a cyclic schedule still produces a Cycle error
        // from cuda_codegen's own topo sort (defense in depth).
        let err = crate::cuda_codegen::generate(&bad_cycle(), 8).unwrap_err();
        assert!(matches!(
            err,
            crate::cuda_codegen::CudaCodegenError::Cycle(_)
        ));
    }

    #[test]
    fn cuda_codegen_nested_repeated_block_inlines_recursively() {
        let src =
            crate::cuda_codegen::generate(&good_nested_repeated_block(), 8).expect("codegen ok");
        // Two inlined sub-schedule regions, each containing the 3-node
        // good_linear body.
        let occurrences = src.matches("sub-schedule: good_linear").count();
        assert_eq!(occurrences, 2);
    }
}
