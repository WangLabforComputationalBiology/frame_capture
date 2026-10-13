# Frame Sensitivity Survives Alignment: Measurement, Diagnosis, and Preliminary Mitigation

**Code and result files for the paper** (Neurocomputing submission, 2026).

> **One-sentence summary**: Answer-preserving changes in wording and context can alter an aligned model's outputs. We quantify this with a protocol-relative index φ = FS/(FS+CS), measure it across 15 checkpoints spanning three pretraining families, find four regularities (frame floor, content-channel amplification by alignment, family fingerprint, DPO polarization), and evaluate three mitigation levers.

---

## Repository layout

```
├── Core experiments
│   ├── run_e1.py, run_e1_qwen15.py, run_e1_fp16_check.py    # Main battery: 48 tasks × 14 frames × 2 content controls × 15 checkpoints
│   ├── d24_matched_abstention.py                            # F20: matched-abstention SOTA comparison (RTLC × our signal)
│   ├── app_trilogy_finalize.py                              # Three application levers (prescription / release gating / abstention)
│   └── items.py                                             # Task item definitions
│
├── Sub-experiment chains (b0–b6)
│   ├── b0_heldout_run.py, b0_analyze_heldout.py             # Held-out risk prediction (AUROC 0.98–0.99)
│   ├── b1_collect.py, b1_analyze.py                         # Representation analysis (social-direction projection)
│   ├── b2_patch.py, b2_analyze.py, b2_crosscheck.py         # Activation patching (layer-localized mechanism)
│   ├── b3_train_trajectory.py, b3_analyze.py                # DPO training trajectory (reward-frame coupling)
│   ├── b4_predict.py                                        # Flip prediction model
│   ├── b5_eval_repair.py                                    # Content-grounded repair (Version B)
│   └── b6_orth.py, b6_analyze_orth.py, b6_layer_profile.py  # Shared subspace + causal skeleton
│
├── Training scripts
│   ├── train_sft.py, train_sft_llama32.py                   # SFT (Qwen / Llama)
│   ├── train_dpo.py, train_dpo_llama32.py                   # DPO (both families)
│   ├── train_dpo_manual.py, train_dpo_llama32_manual.py     # Manual DPO variants
│   ├── train_dpo_frame_v2.py                                # Frame-recoded DPO
│   ├── train_dpo_cure_v3.py                                 # Content-grounded repair training
│   └── train_rlvr.py                                        # RLVR
│
├── Data construction
│   ├── make_sft_data.py                                     # SFT data
│   ├── make_dpo_data.py                                     # DPO preference pairs
│   └── make_dpo_frame_v2.py                                 # Frame-recoded preference pairs
│
├── External comparison (JudgeBench)
│   ├── judgebench_official.py                               # Official JudgeBench-GPT runner
│   ├── d6_rerun_rtlc_weak.py                                # RTLC rerun under matched protocol
│   └── d3_position_aurc.py                                  # Position-bias control + AURC
│
├── Measurement validation
│   ├── e0a_pipeline_noise.py                                # Determinism check (JSD=0 across 10 reruns)
│   ├── e0b_placebo_frame.py                                 # Option-order placebo
│   ├── e0c_extraction_audit.py                              # Extraction validity audit
│   ├── e1_bf16_ladder.py                                    # Unquantized bf16 replication (11 checkpoints)
│   └── e1_compare_bf16_vs_4bit.py                           # NF4 vs bf16 comparison
│
├── Robustness
│   ├── e3a_dpo_seeds_v4.py                                  # Three-seed DPO retraining
│   ├── e3b_dpo_beta_scan.py                                 # β ∈ {0.05, 0.1, 0.3} scan
│   ├── e3c_train_sft_matched.py                             # SFT on DPO's data (matched-data control)
│   └── e3d_train_gemma_dpo.py                               # Third-family replication (Gemma-3)
│
└── Result files (results/, referenced in the paper)
    ├── app_trilogy.json                                     # Three application levers (Ax1/Ax2/Ax3)
    ├── b0_heldout_analysis.json                             # Held-out prediction (B4)
    ├── audit_table.json                                     # Audit table
    ├── deep_review_stats.json                               # Statistical protocols
    ├── f20_matched_abstention.json                          # F20 matched-abstention SOTA comparison
    ├── judge_aggregate.json, judgebench_official.json       # JudgeBench results
    ├── manip_public_flip.json, manip_social_frames.json     # Behavioral experiments
    ├── mmlu_dual.json                                       # MMLU dual evaluation
    ├── p0_metric_audit.json                                 # JSD vs MI metric audit
    ├── p1_round2_audit.json                                 # Round-2 audit
    ├── p3_mi_shared_amplified.json                          # MI-form shared amplified tasks
    ├── selective_qa.json                                    # Selective QA results
    └── t9_final48.json                                      # Leave-one-family-out grid
```

---

## Reproducibility

All reported numbers regenerate from `results/*.json` **without model re-inference**. The measurement pipeline is training-free: it reads option-level logits under each frame variant and computes JSD against the frame average.

**Protocol**: 48 tasks × 14 answer-preserving frame variants (6 language-class families: wording, case, persona, authority, peer, confidence) × 2 content controls × 15 checkpoints (three pretraining families: Qwen2.5-1.5B, Llama-3.2-3B, InternLM2-1.8B; plus 7B/8B cross-checkpoints and Gemma-3-4B).

**Key controls**: identical preference data (9,534 items) and LoRA hyperparameters across the Qwen and Llama ladders; only the pretraining family varies.

---

## Key results

| Finding | Evidence |
|---|---|
| **Frame floor** | Four base models: FS ∈ [0.0070, 0.0081] bits (CV 0.06); no recipe reaches zero |
| **Alignment raises content sensitivity** | CS rises in 8/8 pairs (38–45 of 48 tasks; Holm-adjusted worst p = 1.2×10⁻⁴); FS shows no consistent change |
| **Family fingerprint** | Within-family r = 0.57, cross-family r = 0.11 (p < 10⁻¹⁰) |
| **DPO polarizes** | Median task loses 82–100% of FS; 13–15/48 tasks amplified ≥1.5× (arithmetic/formal logic) |
| **Matched-abstention SOTA** | RTLC × frame-consistency signal: 0.878 @ 63.1% coverage (+7.3 points over RTLC's full coverage) |

---

## Paper

- **Preprint**: [arXiv link TBD]
- **Code availability**: this repository
- **Task materials**: included in `items.py` and result files
- **License**: MIT (code), CC-BY-4.0 (data)

---

## Citation

If you use this code or the φ index in your work, please cite:

```bibtex
@article{yang2026frame,
  title={Frame Sensitivity Survives Alignment: Measurement, Diagnosis, and Preliminary Mitigation},
  author={Yang, Ruofan and Liang, Lixin and Huang, Bingding and Wang, Xin},
  journal={Neurocomputing},
  year={2026},
  note={Submitted}
}
```

---

## Contact

- Ruofan Yang: [email]
- Corresponding authors: Bingding Huang (huangbingding@sztu.edu.cn), Xin Wang (wangxin@sztu.edu.cn)
- Lab: [Wang Lab for Computational Biology](https://github.com/WangLabforComputationalBiology)
