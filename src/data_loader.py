"""Fast streaming parser and data loader for ATD tracking and tactical events."""

import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from src.config import (
    ATD_META_PATH, ATD_RAW_PATH, SMART_TAGGING_PATH, HOMOGRAPHY_PATH,
    PITCH_LENGTH, PITCH_WIDTH, TOTAL_FRAMES, FPS,
    HOME_CODE, AWAY_CODE, DEAD_CODE, NOBODY_CODE
)

def get_track_team_mapping(meta_path=ATD_META_PATH):
    """Parse ATD XML to map 46 track slots to team codes (0=Home, 1=Away)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0] # FIFA/EPTS convention: first team is Home (Barcelona)
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME_CODE if tid == home_id else AWAY_CODE for tid in slots], dtype=np.int8)

def load_smart_tagging_intervals(xml_path=SMART_TAGGING_PATH):
    """Load Smart Tagging XML into structured list of intervals."""
    root = ET.parse(xml_path).getroot()
    instances = []
    for inst in root.iter("instance"):
        code_elem = inst.find("code")
        start_elem = inst.find("start")
        end_elem = inst.find("end")
        if code_elem is None or start_elem is None or end_elem is None:
            continue
            
        start_s = float(start_elem.text)
        end_s = float(end_elem.text)
        
        tags = {}
        for label in inst.iter("label"):
            group = label.findtext("group")
            text = label.findtext("text")
            if group and text:
                tags[group] = text
                
        instances.append({
            "code": code_elem.text,
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * FPS)),
            "end_frame": int(round(end_s * FPS)),
            "tags": tags
        })
    return instances

def stream_tracking_features(raw_path=ATD_RAW_PATH, meta_path=ATD_META_PATH):
    """Extract spatio-temporal features from raw tracking in a single fast pass."""
    track_teams = get_track_team_mapping(meta_path)
    n_frames = TOTAL_FRAMES
    
    # Pre-allocated arrays
    ball_visible = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    
    nearest_team = np.full(n_frames, NOBODY_CODE, dtype=np.int8)
    nearest_dist = np.full(n_frames, np.nan, dtype=np.float32)
    
    home_dist_min = np.full(n_frames, np.nan, dtype=np.float32)
    away_dist_min = np.full(n_frames, np.nan, dtype=np.float32)
    
    mean_player_speed = np.full(n_frames, np.nan, dtype=np.float32)
    home_mean_speed = np.full(n_frames, np.nan, dtype=np.float32)
    away_mean_speed = np.full(n_frames, np.nan, dtype=np.float32)
    
    active_player_count = np.zeros(n_frames, dtype=np.int8)
    
    print(f"Streaming tracking features from {os.path.basename(raw_path)}...")
    with open(raw_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            
            fid = int(parts[0]) - 1 # 0-indexed frame
            if fid < 0 or fid >= n_frames:
                continue
                
            # Parse ball
            ball_part = parts[2]
            b_vis = (ball_part[0] != "N")
            if b_vis:
                c1 = ball_part.index(",")
                c2 = ball_part.index(",", c1 + 1)
                bx = float(ball_part[:c1]) * PITCH_LENGTH
                by = float(ball_part[c1+1:c2]) * PITCH_WIDTH
                ball_visible[fid] = True
                ball_x[fid] = bx
                ball_y[fid] = by
            
            # Parse players
            player_groups = parts[1].split(";")
            h_speeds, a_speeds = [], []
            best_d = 1e9
            best_t = NOBODY_CODE
            h_best_d = 1e9
            a_best_d = 1e9
            count = 0
            
            for slot, group in enumerate(player_groups):
                if group[0] == "N":
                    continue
                count += 1
                c1 = group.index(",")
                c2 = group.index(",", c1 + 1)
                px = float(group[:c1]) * PITCH_LENGTH
                py = float(group[c1+1:c2]) * PITCH_WIDTH
                spd = float(group[c2+1:])
                team = track_teams[slot]
                
                if team == HOME_CODE:
                    h_speeds.append(spd)
                else:
                    a_speeds.append(spd)
                    
                if b_vis:
                    dx = px - bx
                    dy = py - by
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < best_d:
                        best_d = d
                        best_t = team
                    if team == HOME_CODE and d < h_best_d:
                        h_best_d = d
                    elif team == AWAY_CODE and d < a_best_d:
                        a_best_d = d
                        
            active_player_count[fid] = count
            if count > 0:
                all_speeds = h_speeds + a_speeds
                if all_speeds:
                    mean_player_speed[fid] = np.mean(all_speeds)
                if h_speeds:
                    home_mean_speed[fid] = np.mean(h_speeds)
                if a_speeds:
                    away_mean_speed[fid] = np.mean(a_speeds)
                    
            if b_vis and best_d < 1e8:
                nearest_team[fid] = best_t
                nearest_dist[fid] = best_d
                home_dist_min[fid] = h_best_d
                away_dist_min[fid] = a_best_d

    return {
        "ball_visible": ball_visible,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "nearest_team": nearest_team,
        "nearest_dist": nearest_dist,
        "home_dist_min": home_dist_min,
        "away_dist_min": away_dist_min,
        "mean_player_speed": mean_player_speed,
        "home_mean_speed": home_mean_speed,
        "away_mean_speed": away_mean_speed,
        "active_player_count": active_player_count,
    }
