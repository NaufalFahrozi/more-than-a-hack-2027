"""Physics-Informed Kinematics and Pitch Control Rules Engine for Football Analytics.

Computes ball velocity, acceleration, angular deflection, kinematic NaN projection,
true touch/control gating, and multi-scale dead ball stoppage anchors.
"""

import numpy as np
from src.config import (
    TOTAL_FRAMES, PITCH_LENGTH, PITCH_WIDTH, FPS,
    FRAME_PRE_MATCH, FRAME_HALFTIME, FRAME_POST_MATCH,
    FRAME_HALF_2_PRE_KICKOFF,
    HOME_CODE, AWAY_CODE, DEAD_CODE, NOBODY_CODE
)

class FootballRulesEngine:
    """Applies official match geometry, stoppage criteria, set-piece domain rules, and kinematics."""
    
    def __init__(self, pitch_len=PITCH_LENGTH, pitch_wid=PITCH_WIDTH, fps=FPS):
        self.pitch_len = pitch_len
        self.pitch_wid = pitch_wid
        self.fps = fps

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

    def compute_ball_kinematics(self, ball_x, ball_y, ball_visible, decay=0.96, max_proj_gap=25):
        """
        Computes finite-difference velocity, acceleration, and angular deflection.
        Performs forward kinematic velocity projection over short occlusions (NaNs).
        
        Returns:
            dict containing projected_x, projected_y, vx, vy, speed, accel, deflection, is_impulse
        """
        n = len(ball_x)
        bx = ball_x.copy()
        by = ball_y.copy()
        b_vis = ball_visible.copy()
        
        vx = np.zeros(n, dtype=np.float32)
        vy = np.zeros(n, dtype=np.float32)
        spd = np.zeros(n, dtype=np.float32)
        acc = np.zeros(n, dtype=np.float32)
        deflection = np.zeros(n, dtype=np.float32)
        
        # 1. Finite differences on visible frames
        for i in range(1, n):
            if b_vis[i] and b_vis[i-1]:
                vx[i] = (bx[i] - bx[i-1]) * self.fps
                vy[i] = (by[i] - by[i-1]) * self.fps
                spd[i] = np.sqrt(vx[i]**2 + vy[i]**2)
                
        # 2. Kinematic extrapolation over short missing gaps (handling tracking occlusions)
        i = 0
        while i < n:
            if not b_vis[i]:
                gap_start = i
                while i < n and not b_vis[i]:
                    i += 1
                gap_end = i
                gap_len = gap_end - gap_start
                if gap_len <= max_proj_gap and gap_start > 0:
                    cur_x = bx[gap_start - 1]
                    cur_y = by[gap_start - 1]
                    cur_vx = vx[gap_start - 1]
                    cur_vy = vy[gap_start - 1]
                    for f in range(gap_start, gap_end):
                        cur_vx *= decay
                        cur_vy *= decay
                        cur_x += cur_vx * (1.0 / self.fps)
                        cur_y += cur_vy * (1.0 / self.fps)
                        bx[f] = cur_x
                        by[f] = cur_y
                        vx[f] = cur_vx
                        vy[f] = cur_vy
                        spd[f] = np.sqrt(cur_vx**2 + cur_vy**2)
            else:
                i += 1
                
        # 3. Acceleration and angular deflection (cosine similarity)
        for i in range(1, n):
            dvx = vx[i] - vx[i-1]
            dvy = vy[i] - vy[i-1]
            acc[i] = np.sqrt(dvx**2 + dvy**2) * self.fps
            
            s1 = spd[i-1]
            s2 = spd[i]
            if s1 > 1.0 and s2 > 1.0:
                dot = (vx[i-1]*vx[i] + vy[i-1]*vy[i]) / (s1 * s2)
                dot = np.clip(dot, -1.0, 1.0)
                deflection[i] = np.arccos(dot) * (180.0 / np.pi)
                
        is_impulse = (acc > 12.0) | (deflection > 20.0)
        
        return {
            "ball_x_proj": bx,
            "ball_y_proj": by,
            "vx": vx,
            "vy": vy,
            "speed": spd,
            "accel": acc,
            "deflection": deflection,
            "is_impulse": is_impulse
        }

    def detect_out_of_bounds(self, ball_x, ball_y, ball_visible, margin=0.3):
        """Identify frames where ball clearly crossed the pitch boundaries."""
        oob = np.zeros(len(ball_x), dtype=bool)
        valid = ball_visible & (~np.isnan(ball_x)) & (~np.isnan(ball_y))
        
        x_out = (ball_x[valid] < -margin) | (ball_x[valid] > (self.pitch_len + margin))
        y_out = (ball_y[valid] < -margin) | (ball_y[valid] > (self.pitch_wid + margin))
        
        oob[valid] = x_out | y_out
        return oob

    def detect_collective_stoppage(self, mean_player_speed, speed_threshold=1.10, min_consecutive=40):
        """Detect sustained stoppage phases where all players slow to walking pace."""
        low_speed = (mean_player_speed < speed_threshold) & (~np.isnan(mean_player_speed))
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

    def detect_smart_tagging_anchors(self, events, mean_speed, ball_spd, nearest_dist, is_impulse, total_frames=TOTAL_FRAMES):
        """
        Bidirectional anchor propagation:
        - Backward propagation: scans backward up to 5s before official set piece tag to capture the exact whistle stoppage.
        - Forward propagation: scans forward to identify the exact physical restart kick.
        """
        sp_dead = np.zeros(total_frames, dtype=bool)
        
        for e in events:
            if e["code"] == "SET PIECES":
                sf = max(0, e["start_frame"])
                ef = min(total_frames, e["end_frame"])
                
                # Backward propagation: trace deceleration window
                back_sf = sf
                for f in range(sf - 1, max(0, sf - 125), -1):
                    if mean_speed[f] < 1.30 or np.isnan(mean_speed[f]):
                        back_sf = f
                    else:
                        break
                        
                # Forward scan: detect physical restart kick
                restart_f = ef
                for f in range(sf + 20, min(total_frames, ef + 100)):
                    if (ball_spd[f] > 5.0 or is_impulse[f]) and nearest_dist[f] < 2.5:
                        restart_f = f
                        break
                        
                sp_dead[back_sf:restart_f] = True
                
        return sp_dead
