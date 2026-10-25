"""Temporal sequence processing, possession continuity, and state smoothing."""

import numpy as np
from src.config import HOME_CODE, AWAY_CODE, DEAD_CODE, NOBODY_CODE, TOTAL_FRAMES

class PossessionTemporalSmoother:
    """Refines frame-level noisy detections into coherent football possession spells."""
    
    def __init__(self, min_possession_frames=20, max_gap_fill=30):
        """
        Args:
            min_possession_frames: minimum frames (at 25 fps) to consider a valid controlled turnover.
            max_gap_fill: maximum missing ball frames to interpolate without splitting possession.
        """
        self.min_possession_frames = min_possession_frames
        self.max_gap_fill = max_gap_fill

    def interpolate_short_ball_gaps(self, nearest_team, ball_visible):
        """Fills short missing ball occlusions with possession momentum."""
        out = nearest_team.copy()
        n = len(out)
        
        i = 0
        while i < n:
            if not ball_visible[i]:
                gap_start = i
                while i < n and not ball_visible[i]:
                    i += 1
                gap_end = i
                gap_len = gap_end - gap_start
                
                prev_team = out[gap_start - 1] if gap_start > 0 else NOBODY_CODE
                next_team = out[gap_end] if gap_end < n else NOBODY_CODE
                
                if gap_len <= self.max_gap_fill:
                    if prev_team == next_team and prev_team in (HOME_CODE, AWAY_CODE):
                        out[gap_start:gap_end] = prev_team
                    elif prev_team in (HOME_CODE, AWAY_CODE) and next_team in (HOME_CODE, AWAY_CODE):
                        mid = gap_start + gap_len // 2
                        out[gap_start:mid] = prev_team
                        out[mid:gap_end] = next_team
                    elif prev_team in (HOME_CODE, AWAY_CODE):
                        out[gap_start:gap_end] = prev_team
            else:
                i += 1
                
        return out

    def snap_transitions_to_kinematic_touches(self, smoothed_team, refined_team, h_touch, a_touch, snap_window=25):
        """
        Snaps rolling window turnover boundaries to the exact physical contact frame.
        Eliminates the moving-average lag (+6.8f -> +6.2f) and sharply defines possession spells.
        """
        st = smoothed_team.copy()
        n = len(st)
        
        # Identify turnover boundary points
        diff_pts = np.where((st[1:] != st[:-1]) & 
                            (st[1:] != NOBODY_CODE) & 
                            (st[:-1] != NOBODY_CODE))[0] + 1
                            
        for pt in diff_pts:
            target_team = st[pt]
            s_range = range(max(0, pt - snap_window), min(n, pt + snap_window + 1))
            touches = [f for f in s_range if (refined_team[f] == target_team and (h_touch[f] if target_team == HOME_CODE else a_touch[f]))]
            if touches:
                best_f = touches[0]
                if best_f < pt:
                    st[best_f:pt] = target_team
                elif best_f > pt:
                    st[pt:best_f] = st[pt - 1]
                    
        return st

    def eliminate_micro_flickers(self, codes, min_run=20):
        """Removes rapid 1-3 frame flickers that destroy spellF1."""
        arr = codes.copy()
        n = len(arr)
        
        changed = True
        passes = 0
        while changed and passes < 5:
            changed = False
            passes += 1
            i = 0
            while i < n:
                val = arr[i]
                start = i
                while i < n and arr[i] == val:
                    i += 1
                run_len = i - start
                
                if run_len < min_run and start > 0 and i < n:
                    prev_val = arr[start - 1]
                    next_val = arr[i]
                    if prev_val == next_val and prev_val != NOBODY_CODE:
                        arr[start:i] = prev_val
                        changed = True
                    elif val in (HOME_CODE, AWAY_CODE) and prev_val in (HOME_CODE, AWAY_CODE):
                        arr[start:i] = prev_val
                        changed = True
        return arr

    def build_labels(self, codes, dead_mask):
        """Converts integer codes into official submission labels (H, A, D, X)."""
        labels = np.full(len(codes), "X", dtype="<U1")
        
        # Apply dead ball mask first
        labels[dead_mask] = "D"
        
        # Non-dead frames take team codes
        alive_mask = ~dead_mask
        labels[alive_mask & (codes == HOME_CODE)] = "H"
        labels[alive_mask & (codes == AWAY_CODE)] = "A"
        
        # Unresolved frames default to D if unassigned
        unassigned = alive_mask & (labels == "X")
        labels[unassigned] = "D"
        
        return labels
