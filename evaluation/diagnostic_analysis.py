"""Diagnostic Error Breakdown and Temporal Block Sensitivity Analysis for StartUP v3."""

import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "evaluation"))

import numpy as np
import pandas as pd
import possession_metrics as pm

def run_diagnostics():
    print("=" * 70)
    print(">> DIAGNOSTIC ERROR BREAKDOWN & TEMPORAL STABILITY ANALYSIS")
    print("=" * 70)
    
    gt_path = "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    pred_path = "evaluation/results/StartUP.csv"
    
    gt_df = pm.read_label_csv(gt_path)
    pred_df = pm.read_label_csv(pred_path)
    
    gt_codes = pm.label_codes(gt_df)
    pred_codes = pm.label_codes(pred_df)
    
    # 1. Overall evaluation
    eval_res = pm.evaluate(gt_codes, pred_codes, taus=(12, 25), bootstrap_n=0)
    
    print("\n--- 1. MACRO ERROR TAXONOMY ---")
    print(f"Total GT Possessions : {eval_res['n_gt_spells']}")
    print(f"Total Pred Possessions: {eval_res['n_pred_spells']}")
    print(f"Spell Count Error    : {eval_res['spell_count_error']} (0 = Perfect Match!)")
    print(f"Fragmentation Ratio  : {eval_res['fragmentation']:.2f}")
    print(f"Transition Mean Lat  : {eval_res['transitions'][25]['mean_latency']:+.1f} frames ({eval_res['transitions'][25]['mean_latency']/25.0:+.2f}s)")
    
    # 2. Segment-level analysis on the 21 GT segments
    # Detect exact spell matching IoUs
    print("\n--- 2. DETAILED IOUS ACROSS THRESHOLDS ---")
    for iou_th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        if iou_th in eval_res["spell_f1"]:
            sub = eval_res["spell_f1"][iou_th]
            print(f"IoU {iou_th:.1f} -> Precision: {sub['precision']:.4f} | Recall: {sub['recall']:.4f} | F1: {sub['f1']:.4f} | Matched: {sub['n_matched']}/{eval_res['n_gt_spells']}")
            
    # 3. Temporal Block Analysis across the 5-minute slice (4 Quarters of 75 seconds = 1875 frames each)
    print("\n--- 3. 4-QUARTER TEMPORAL BLOCK CROSS-VALIDATION (75s blocks) ---")
    graded_mask = gt_codes != 3 # 3 is 'X' (ungraded)
    graded_indices = np.where(graded_mask)[0]
    start_f = graded_indices[0]
    end_f = graded_indices[-1] + 1
    total_graded = len(graded_indices)
    quarter_len = total_graded // 4
    
    for q in range(4):
        q_start = start_f + q * quarter_len
        q_end = start_f + (q + 1) * quarter_len if q < 3 else end_f
        
        q_gt = gt_codes[q_start:q_end]
        q_pred = pred_codes[q_start:q_end]
        
        # Calculate block accuracy and macro F1
        q_acc = np.mean(q_gt == q_pred)
        
        # Per-class F1 in quarter (0=H, 1=A, 2=D)
        classes = [0, 1, 2]
        f1s = []
        for c in classes:
            tp = np.sum((q_gt == c) & (q_pred == c))
            fp = np.sum((q_gt != c) & (q_pred == c))
            fn = np.sum((q_gt == c) & (q_pred != c))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1s.append(f1)
            
        q_macro_f1 = np.mean(f1s)
        time_s1 = (q_start - start_f) / 25.0
        time_s2 = (q_end - start_f) / 25.0
        print(f"Quarter {q+1} [{time_s1:4.0f}s - {time_s2:4.0f}s] -> Acc: {q_acc*100:5.2f}% | Macro F1: {q_macro_f1:.4f} | H: {f1s[0]:.4f} | A: {f1s[1]:.4f} | D: {f1s[2]:.4f}")

if __name__ == "__main__":
    run_diagnostics()
