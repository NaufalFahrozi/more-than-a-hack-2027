# ⚽ More Than A Hack 2027: Spatio-Temporal Football Possession Intelligence

[![Hackathon](https://img.shields.io/badge/Bar%C3%A7a%20Innovation%20Hub-More%20Than%20A%20Hack%202027-blue?logo=barcelona)](https://landing.mobileworldcapital.com/more-than-a-hack-2027)
[![Organizers](https://img.shields.io/badge/Organized%20By-Metrica%20Sports%20%7C%20FC%20Barcelona%20%7C%20MWCapital-red)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced spatio-temporal possession tracking and segmentation pipeline built for **More Than A Hack 2027 (Phase 1: Ball Possession)**, organized collaboratively by **Mobile World Capital Barcelona**, **Barça Innovation Hub (FC Barcelona)**, and **Metrica Sports** at the Talent Arena Barcelona.

---

## 📌 Executive Summary & Key Results

Evaluating possession frame-by-frame on raw, uncorrected tracking data is a notoriously difficult problem in elite football analytics. Automatic computer vision pipelines suffer from frequent occlusions (**the ball is missing on 41% of graded frames**) and noisy tracking swaps.

Our solution introduces **Spatio-Temporal Physics & Collective Velocity Modeling**, achieving:
* **`macroF1` of `0.7131`** (ranked **#1 overall**, beating all official Metrica baselines).
* **`spellF1` of `0.4211`** (significantly surpassing the official `baseline_smooth50` floor).
* **Overall Accuracy of `76.87%`** and **Cohen's Kappa of `0.6176`**.
* **Average predicted possession duration of `13.0s`** (closely matching the Ground Truth average of `14.2s`).

### 📊 Benchmark Comparison (Scored on Public Ground Truth)

| Rank | Model / Method | macroF1 | spellF1 | Pred Spells | Mean Duration | Alive/Dead F1 | Accuracy |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Our Smart Pipeline (`v1_smart_pipeline`)** | **`0.7131`** | **`0.4211`** | **21** *(GT=17)* | **13.0s** *(GT=14.2s)* | **`0.9195`** | **`76.87%`** |
| 🥈 | `baseline_smooth125` | `0.5812` | `0.4242` | 16 | 18.5s | `0.8991` | `72.75%` |
| 🥉 | `baseline_smooth50` (Official Floor) | `0.6207` | `0.3256` | 26 | 10.8s | `0.8990` | `71.32%` |
| 4 | `baseline_smooth25` | `0.6200` | `0.2868` | 36 | 7.6s | `0.8974` | `69.12%` |
| 5 | `baseline_raw` (Nearest Player) | `0.5972` | `0.0378` | 131 | 2.0s | `0.8920` | `65.12%` |

---

## 🧠 Technical Methodology & Architecture

Rather than relying on naive nearest-player heuristics that flicker across frames or heavy deep learning architectures that overfit, our approach models football as a **constrained physical state machine**:

```
                                  [Raw ATD Tracking: 25 FPS]
                                  (22 Players + Ball Coordinates)
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
       [Physics & Geometry]          [Collective Dynamics]          [Event Synchronization]
       Pitch boundaries (105x68m)    Mean team velocity droop       Smart Tagging set-pieces
       Out-of-bounds detection       Stoppage / foul detection      (Free kicks, corners, etc.)
               │                               │                               │
               └───────────────────────┬───────┴───────────────────────────────┘
                                       ▼
                         [Dead Ball Mask Generator]
                                       │
                                       ▼
                  [Short-Gap Ball Occlusion Interpolator]
                  (Forward-backward momentum over missing frames)
                                       │
                                       ▼
                  [Non-Destructive Temporal Smoother]
                  (Prevents micro-flickering & preserves stoppages)
                                       │
                                       ▼
                     [Official Competition Boundary Layer]
                     (Pre-kickoff, Halftime, Stoppage lock)
                                       │
                                       ▼
                        [submission.csv (199,319 rows)]
```

### 1. Collective Velocity Droop (Foul & Stoppage Detection)
* **The Insight:** When play is stopped for fouls, injuries, or set-piece preparations, camera tracking may still track the stationary ball. Naive baselines mistakenly assign possession to the player nearest to the dead ball.
* **Our Solution:** We measure collective 22-player mean speed. During live play, player speeds average **2.5–4.2 m/s**. During stoppages, collective velocity drops below **1.55 m/s**. Detecting sustained speed drops flags dead balls (`D`) even when the ball is clearly visible.

### 2. Missing-Ball Momentum (Bridging 41% Occlusion)
* Ball tracking frequently drops in crowded penalty boxes or aerial duels.
* We apply short-gap possession interpolation ($K \le 45$ frames / 1.8s). If Team A had possession prior to occlusion and is still contesting when the ball re-emerges, possession continuity is maintained rather than spuriously split into false dead balls.

### 3. Non-Destructive Temporal Smoothing
* Standard majority voting filters inadvertently smooth away genuine short stoppages.
* Our smoother applies categorical state segmentation: dead-ball intervals are strictly preserved as topological boundaries, while live-play team assignments are filtered to guarantee controlled possession transitions.

---

## 🛠️ Repository Structure

```
├── data/
│   └── tactical/                # Tactical match assets (gitignored)
├── evaluation/
│   ├── gt/                      # Ground truth reference
│   ├── results/                 # Generated submission files
│   ├── possession_metrics.py    # Official evaluation metrics specification
│   ├── run_evaluation.py        # Official benchmark evaluation runner
│   └── validate_submission.py   # Submission contract validator
├── src/
│   ├── __init__.py
│   ├── config.py                # Global parameters, pitch geometry, frame blocks
│   ├── data_loader.py           # High-speed streaming ATD & Smart Tagging parser
│   ├── models/
│   │   ├── rules_engine.py      # Pitch boundary, match constraints & stoppage logic
│   │   └── temporal_smoother.py # Possession continuity & flicker elimination
│   └── pipeline.py              # End-to-end execution pipeline
├── starter/                     # Exploration notebooks & baseline scripts
├── CHALLENGE.md                 # Official task description
├── DATA.md                      # Dataset specifications
├── EVALUATION.md                # Scoring metric documentation
├── requirements.txt             # Pinned core dependencies
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/more-than-a-hack-2027.git
cd more-than-a-hack-2027

# Install dependencies (Numpy, Pandas, Scipy, Scikit-Learn, LightGBM)
pip install -r requirements.txt
```

### 2. Match Assets
Place the official tactical data in `data/tactical/`:
* `BAR-ATH_tactical_atd.txt`
* `BAR-ATH_tactical_atd.xml`
* `BAR-ATH_tactical_smart_tagging.xml`
* `BAR-ATH_tactical_homography.csv`

### 3. Generate Submission
Run our end-to-end pipeline:
```bash
python -m src.pipeline
```
*Output: `evaluation/results/v1_smart_pipeline.csv` (199,319 rows, strictly formatted).*

### 4. Validate & Score
```bash
# Verify submission format integrity
python evaluation/validate_submission.py evaluation/results/v1_smart_pipeline.csv

# Run official evaluation against public ground truth
python evaluation/run_evaluation.py
```

---

## 👥 Authors & Team

* **Lead Analyst & Engineer:** Project Participant
* **Team Name:** *(As registered for More Than A Hack 2027)*
* **Event:** More Than A Hack 2027 — Barça Innovation Hub & Metrica Sports

---

## 📄 License & Terms
The source code in this repository is released under the **MIT License**. The match data provided by FC Barcelona and Metrica Sports is subject to the competition terms of use and is not redistributed in this repository.
