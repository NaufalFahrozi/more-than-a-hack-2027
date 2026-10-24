"""Main execution pipeline for More Than a Hack 2027."""

import os
import sys
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

def run_smart_pipeline(output_filename="v1_smart_pipeline.csv", smooth_window=100):
    """Executes the full domain-informed possession classification pipeline."""
    print("=" * 60)
    print(">> Running More Than a Hack 2027 - Smart Possession Pipeline")
    print("=" * 60)
    
    # 1. Load tracking features
    feats = stream_tracking_features()
    ball_visible = feats["ball_visible"]
    ball_x = feats["ball_x"]
    ball_y = feats["ball_y"]
    nearest_team = feats["nearest_team"]
    nearest_dist = feats["nearest_dist"]
    mean_speed = feats["mean_player_speed"]
    
    # 2. Initialize modules
    rules = FootballRulesEngine()
    smoother = PossessionTemporalSmoother(min_possession_frames=20, max_gap_fill=45)
    
    # Calculate finite-difference ball speed (m/s)
    dx = np.diff(ball_x, prepend=ball_x[0])
    dy = np.diff(ball_y, prepend=ball_y[0])
    ball_speed = np.sqrt(dx**2 + dy**2) * 25.0
    ball_speed[~ball_visible] = np.nan
    
    # 3. Detect Dead Ball (D) signals
    dead_mask = np.zeros(TOTAL_FRAMES, dtype=bool)
    
    # A) Out of bounds
    oob_mask = rules.detect_out_of_bounds(ball_x, ball_y, ball_visible, margin=0.5)
    dead_mask |= oob_mask
    print(f"  Out of bounds detected: {np.sum(oob_mask)} frames")
    
    # B) Collective player stoppage (low speed)
    stoppage_mask = rules.detect_collective_stoppage(mean_speed, speed_threshold=1.60, min_consecutive=30)
    dead_mask |= stoppage_mask
    print(f"  Collective player stoppages: {np.sum(stoppage_mask)} frames")
    
    # C) Smart Tagging Set-Pieces with dynamic restart detection
    events = load_smart_tagging_intervals()
    sp_dead = rules.detect_set_piece_dead_intervals(events, ball_visible, ball_speed, nearest_dist)
    dead_mask |= sp_dead
    print(f"  Smart Tagging Set-Pieces dynamic dead ball: {np.sum(sp_dead)} frames")
    
    # 4. Fill short ball occlusions using possession continuity
    print("  Interpolating short missing ball occlusions...")
    continuous_team = smoother.interpolate_short_ball_gaps(nearest_team, ball_visible)
    
    # 5. Apply temporal majority vote smoothing over active team codes
    print(f"  Applying temporal window smoothing (+/- {smooth_window} frames)...")
    # Cumulative sums for O(N) rolling counts
    h_cum = np.concatenate(([0], np.cumsum(continuous_team == HOME_CODE)))
    a_cum = np.concatenate(([0], np.cumsum(continuous_team == AWAY_CODE)))
    
    idx = np.arange(TOTAL_FRAMES)
    low = np.maximum(0, idx - smooth_window)
    high = np.minimum(TOTAL_FRAMES, idx + smooth_window + 1)
    
    h_win = h_cum[high] - h_cum[low]
    a_win = a_cum[high] - a_cum[low]
    
    # Majority assignment for alive frames
    smoothed_team = np.full(TOTAL_FRAMES, NOBODY_CODE, dtype=np.int8)
    seen = (h_win + a_win) > 0
    smoothed_team[seen] = np.where(h_win[seen] >= a_win[seen], HOME_CODE, AWAY_CODE)
    
    # 6. Eliminate micro flickers
    smoothed_team = smoother.eliminate_micro_flickers(smoothed_team, min_run=20)
    
    # 7. Build categorical labels (H, A, D, X)
    labels = smoother.build_labels(smoothed_team, dead_mask)
    
    # 8. Apply official competition boundaries (halftime, pre-kickoff, post-match)
    labels = rules.apply_match_boundaries(labels)
    
    # 9. Write submission CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, output_filename)
    df_out = pd.DataFrame({"frame": np.arange(TOTAL_FRAMES), "label": labels})
    df_out.to_csv(out_path, index=False)
    print(f"[OK] Generated submission: {out_path} ({len(df_out)} rows)")
    
    counts = df_out["label"].value_counts().to_dict()
    print("  Label distribution:", counts)
    
    return out_path

if __name__ == "__main__":
    run_smart_pipeline()
