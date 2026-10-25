"""Main execution pipeline for More Than a Hack 2027 (StartUP v3).

Physics-Informed Kinematics, Multi-Scale Dead-Ball Anchors, and Kinematic Turnover Snapping.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.config import (
    TOTAL_FRAMES, RESULTS_DIR,
    HOME_CODE, AWAY_CODE, DEAD_CODE, NOBODY_CODE
)
from src.data_loader import (
    stream_tracking_features, load_smart_tagging_intervals
)
from src.models.rules_engine import FootballRulesEngine
from src.models.temporal_smoother import PossessionTemporalSmoother

def run_smart_pipeline(output_filename="StartUP.csv", smooth_window=115, snap_window=25, min_flicker=20):
    """Executes the full physics-informed possession classification pipeline (v3)."""
    print("=" * 70)
    print(">> Running More Than a Hack 2027 - StartUP Pipeline (v3)")
    print("=" * 70)
    
    # 1. Load tracking features
    feats = stream_tracking_features()
    ball_visible = feats["ball_visible"]
    ball_x = feats["ball_x"]
    ball_y = feats["ball_y"]
    nearest_team = feats["nearest_team"]
    nearest_dist = feats["nearest_dist"]
    mean_speed = feats["mean_player_speed"]
    
    # 2. Initialize engines
    rules = FootballRulesEngine()
    smoother = PossessionTemporalSmoother(min_possession_frames=20, max_gap_fill=30)
    
    # 3. Physics Kinematics & Velocity Projection for Missing Ball (handling Silent NaNs)
    print("1. Computing Kinematics & Velocity Projection for Missing Ball...")
    kin = rules.compute_ball_kinematics(ball_x, ball_y, ball_visible, decay=0.96, max_proj_gap=25)
    bx_proj = kin["ball_x_proj"]
    by_proj = kin["ball_y_proj"]
    ball_spd = kin["speed"]
    is_impulse = kin["is_impulse"]
    
    # 4. Multi-Scale Dead Ball & Smart Tagging Bidirectional Anchors
    print("2. Detecting Out of Bounds & Multi-Scale Dead Ball Anchors...")
    oob_mask = rules.detect_out_of_bounds(bx_proj, by_proj, ball_visible, margin=0.3)
    stoppage_40 = rules.detect_collective_stoppage(mean_speed, speed_threshold=1.10, min_consecutive=40)
    stoppage_20 = rules.detect_collective_stoppage(mean_speed, speed_threshold=0.85, min_consecutive=20)
    stoppage_mask = stoppage_40 | stoppage_20
    
    events = load_smart_tagging_intervals()
    sp_dead = rules.detect_smart_tagging_anchors(events, mean_speed, ball_spd, nearest_dist, is_impulse)
    
    dead_mask = oob_mask | stoppage_mask | sp_dead
    print(f"  Out of bounds detected: {np.sum(oob_mask)} frames")
    print(f"  Collective player stoppages: {np.sum(stoppage_mask)} frames")
    print(f"  Smart Tagging Set-Pieces dynamic dead ball: {np.sum(sp_dead)} frames")
    print(f"  Total dead ball frames: {np.sum(dead_mask)} frames")
    
    # 5. True Touch / Control Gating
    print("3. Validating True Touches and Controlled Dribbles...")
    refined_team = nearest_team.copy()
    h_touch = (nearest_team == HOME_CODE) & (nearest_dist < 1.6) & (is_impulse | (nearest_dist < 1.0))
    a_touch = (nearest_team == AWAY_CODE) & (nearest_dist < 1.6) & (is_impulse | (nearest_dist < 1.0))
    refined_team[h_touch] = HOME_CODE
    refined_team[a_touch] = AWAY_CODE
    
    # 6. Fill short ball occlusions using possession continuity
    print("4. Interpolating short missing ball occlusions (<= 30 frames)...")
    continuous_team = smoother.interpolate_short_ball_gaps(refined_team, ball_visible)
    
    # 7. Apply temporal window smoothing over active team codes
    print(f"5. Applying temporal window smoothing (+/- {smooth_window} frames)...")
    h_cum = np.concatenate(([0], np.cumsum(continuous_team == HOME_CODE)))
    a_cum = np.concatenate(([0], np.cumsum(continuous_team == AWAY_CODE)))
    
    idx = np.arange(TOTAL_FRAMES)
    low = np.maximum(0, idx - smooth_window)
    high = np.minimum(TOTAL_FRAMES, idx + smooth_window + 1)
    
    h_win = h_cum[high] - h_cum[low]
    a_win = a_cum[high] - a_cum[low]
    
    smoothed_team = np.full(TOTAL_FRAMES, NOBODY_CODE, dtype=np.int8)
    seen = (h_win + a_win) > 0
    smoothed_team[seen] = np.where(h_win[seen] >= a_win[seen], HOME_CODE, AWAY_CODE)
    
    # 8. Eliminate micro flickers
    print(f"6. Eliminating micro flickers (< {min_flicker} frames)...")
    smoothed_team = smoother.eliminate_micro_flickers(smoothed_team, min_run=min_flicker)
    
    # 9. Kinematic Turnover Snapping
    print(f"7. Snapping transition boundaries to kinematic impulse touches (window={snap_window})...")
    smoothed_team = smoother.snap_transitions_to_kinematic_touches(
        smoothed_team, refined_team, h_touch, a_touch, snap_window=snap_window
    )
    
    # 10. Build categorical labels (H, A, D, X)
    labels = smoother.build_labels(smoothed_team, dead_mask)
    
    # 11. Apply official competition boundaries (halftime, pre-kickoff, post-match)
    labels = rules.apply_match_boundaries(labels)
    
    # 12. Write submission CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, output_filename)
    df_out = pd.DataFrame({"frame": np.arange(TOTAL_FRAMES), "label": labels})
    df_out.to_csv(out_path, index=False)
    print(f"[OK] Generated submission: {out_path} ({len(df_out)} rows)")
    
    counts = df_out["label"].value_counts().to_dict()
    print("  Label distribution:", counts)
    
    return out_path

if __name__ == "__main__":
    out_name = sys.argv[1] if len(sys.argv) > 1 else "StartUP.csv"
    run_smart_pipeline(output_filename=out_name)
