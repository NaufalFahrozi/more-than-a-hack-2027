"""Global configuration, constants, and paths for More Than a Hack 2027."""

import os

# Base paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "tactical")
EVAL_DIR = os.path.join(ROOT_DIR, "evaluation")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
GT_PUBLIC_PATH = os.path.join(EVAL_DIR, "gt", "BAR-ATH_possession_gt_public.csv")

# Asset paths
ATD_RAW_PATH = os.path.join(DATA_DIR, "BAR-ATH_tactical_atd.txt")
ATD_META_PATH = os.path.join(DATA_DIR, "BAR-ATH_tactical_atd.xml")
SMART_TAGGING_PATH = os.path.join(DATA_DIR, "BAR-ATH_tactical_smart_tagging.xml")
HOMOGRAPHY_PATH = os.path.join(DATA_DIR, "BAR-ATH_tactical_homography.csv")
VIDEO_PATH = os.path.join(DATA_DIR, "BAR-ATH_tactical.mp4")

# Pitch dimensions (metres)
PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0

# Video constants
FPS = 25
TOTAL_FRAMES = 199319

# Match boundaries (video frame indices)
FRAME_PRE_MATCH = (0, 11184)
FRAME_HALF_1 = (11185, 90651)
FRAME_HALFTIME = (90652, 110807)
FRAME_HALF_2 = (110808, 190197)
FRAME_POST_MATCH = (190198, 199318)

# Second half real kickoff occurs at frame 113742
# Frames 110808..113741 are graded and guaranteed DEAD BALL (D)
FRAME_HALF_2_PRE_KICKOFF = (110808, 113741)

# Labels
HOME_CODE = 0
AWAY_CODE = 1
DEAD_CODE = 2
NOBODY_CODE = -1

LABEL_MAP = {
    HOME_CODE: "H",
    AWAY_CODE: "A",
    DEAD_CODE: "D",
    NOBODY_CODE: "X"
}
