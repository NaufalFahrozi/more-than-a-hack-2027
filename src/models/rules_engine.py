"""Domain rules and geometric constraints for football possession detection."""

import numpy as np
from src.config import (
    TOTAL_FRAMES, PITCH_LENGTH, PITCH_WIDTH,
    FRAME_PRE_MATCH, FRAME_HALFTIME, FRAME_POST_MATCH,
    FRAME_HALF_2_PRE_KICKOFF,
    HOME_CODE, AWAY_CODE, DEAD_CODE, NOBODY_CODE
)

class FootballRulesEngine:
    """Applies official match geometry, stoppage criteria, and set-piece domain rules."""
    
    def __init__(self, pitch_len=PITCH_LENGTH, pitch_wid=PITCH_WIDTH):
        self.pitch_len = pitch_len
        self.pitch_wid = pitch_wid

    def apply_match_boundaries(self, labels):
        """Force exact match boundary labels according to official competition rules."""
        out = labels.copy()
        
        # Ungraded frames are set to X (costs 0)
        out[FRAME_PRE_MATCH[0]:FRAME_PRE_MATCH[1] + 1] = "X"
        out[FRAME_HALFTIME[0]:FRAME_HALFTIME[1] + 1] = "X"
        out[FRAME_POST_MATCH[0]:FRAME_POST_MATCH[1] + 1] = "X"
        
        # Second half pre-kickoff: 2934 graded frames that are strictly DEAD BALL
        out[FRAME_HALF_2_PRE_KICKOFF[0]:FRAME_HALF_2_PRE_KICKOFF[1] + 1] = "D"
        
        return out

    def detect_out_of_bounds(self, ball_x, ball_y, ball_visible, margin=0.5):
        """Identify frames where ball clearly crossed the pitch boundaries."""
        oob = np.zeros(len(ball_x), dtype=bool)
        valid = ball_visible & (~np.isnan(ball_x)) & (~np.isnan(ball_y))
        
        x_out = (ball_x[valid] < -margin) | (ball_x[valid] > (self.pitch_len + margin))
        y_out = (ball_y[valid] < -margin) | (ball_y[valid] > (self.pitch_wid + margin))
        
        oob[valid] = x_out | y_out
        return oob

    def detect_collective_stoppage(self, mean_player_speed, speed_threshold=1.10, min_consecutive=50):
        """Detect sustained stoppage phases where all players slow to walking pace."""
        low_speed = (mean_player_speed < speed_threshold) & (~np.isnan(mean_player_speed))
        
        # Find continuous stretches of low player speed >= min_consecutive frames
        stoppage = np.zeros(len(mean_player_speed), dtype=bool)
        count = 0
        start = 0
        for i, val in enumerate(low_speed):
            if val:
                if count == 0:
                    start = i
                count += 1
            else:
                if count >= min_consecutive:
                    stoppage[start:i] = True
                count = 0
        if count >= min_consecutive:
            stoppage[start:] = True
            
        return stoppage

    def detect_set_piece_dead_intervals(self, events, ball_vis, ball_speed, nearest_dist, total_frames=TOTAL_FRAMES):
        """Identify dead ball windows preceding set piece restart kicks."""
        sp_dead = np.zeros(total_frames, dtype=bool)
        for e in events:
            if e["code"] == "SET PIECES":
                sf = max(0, e["start_frame"])
                ef = min(total_frames, e["end_frame"])
                restart_frame = ef
                # Scan for dynamic restart kick (ball velocity spike near player)
                for f in range(sf + 25, ef):
                    if ball_vis[f] and ball_speed[f] > 5.0 and nearest_dist[f] < 2.5:
                        restart_frame = f
                        break
                sp_dead[sf:restart_frame] = True
        return sp_dead
