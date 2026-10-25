# 🏆 More Than A Hack 2027 — Final Pitch Presentation Deck

**Team:** StartUP  
**Competition:** More Than A Hack 2027 — Phase 1: Ball Possession  
**Organizers:** Barça Innovation Hub (FC Barcelona), Metrica Sports, Mobile World Capital Barcelona  
**Venue:** Talent Arena Barcelona  

---

## Slide 1: Title & Team Introduction
* **Headline:** Spatio-Temporal Football Possession Intelligence
* **Sub-headline:** Solving Uncorrected Tracking Gaps & Stoppage Ambiguities via Physics-Informed Kinematics & Bidirectional Anchors
* **Team:** StartUP
  * **Muhammad Naufal Fahrozi** (@NaufalFahrozi) — Team Lead & AI Systems Architect (`naufalfahrozi01@gmail.com`)
  * **Raymond Thymotimmanuel** (@Rodexxx24) — Football Domain Specialist & Tactical Analyst (`raymondzero123@gmail.com`)
  * **Tua Delima Sitompul** (@delima-sitompul) — Lead Data Scientist & Spatio-Temporal Modeler (`delimasitompul514@gmail.com`)
* **Goal:** Production-grade, high-fidelity frame-by-frame ball possession classification for FC Barcelona.

---

## Slide 2: The Core Industry Challenge
* **The Problem:** Tracking in professional football is inherently uncorrected and noisy:
  * **40.7% Ball Occlusion:** In crowded boxes and aerial duels, the ball disappears.
  * **False Proximity Trap:** Naive computer vision baselines assign possession to whoever stands nearest to a resting ball during fouls or injuries, destroying dead-ball (`D`) recall (Baseline Recall: 21.4%).
* **The Stakes:** High-level tactical metrics (PPDA, field tilt, progressive passes) depend entirely on knowing true possession boundaries.

---

## Slide 3: The Systematic Ablation Study (4 Progressive Stages)
* Rather than guessing a model, our team benchmarked 4 progressive architectural stages:
  1. **Model 0 (Raw Proximity):** `spellF1 = 0.0378` | 131 fragmented micro-spells (Flickering failure).
  2. **Model 1 (Naive Smoothers):** `spellF1 = 0.3256 - 0.4242` | Boundary erosion; erases 1-4s possessions.
  3. **Model 2 (Physics Heuristic v2):** `spellF1 = 0.4556` | 19 spells; ~7s dead-ball lag.
  4. **Model 3 (StartUP v3 - Champion):** **`spellF1 = 0.4941`** | **`macroF1 = 0.7908`** | **Accuracy: 81.51%** | **17 Spells (Exact match!)**

---

## Slide 4: The "Statistical Loss vs. Temporal IoU" Paradox
* **Why 40-Epoch Transformers & LightGBM Failed on `spellF1`:**
  * Cross-Entropy Loss optimizes per-frame classification. In contested 50/50 midfield duels, probabilities oscillate ($P(H) \approx 0.49 \leftrightarrow 0.51$).
  * A 2-frame flip in a 15-second possession splits the spell into two 7-second halves, dropping Temporal IoU $< 0.5$ and collapsing `spellF1` (0.0000 - 0.2449).
* **The Solution:** Football possession is a **topological continuity phenomenon**. It requires physical state momentum and rigid topological boundary locking.

---

## Slide 5: The StartUP v3 Architecture (Four Core Pillars)
1. **Kinematics & Angular Deflection:** True touch validation via impulse spikes ($a_b > 12\text{ m/s}^2$ or $\Delta \theta_b > 20^\circ$). High-speed crosses without deflection are never misclassified as turnovers.
2. **Velocity-Decayed NaN Projection:** Projects ball trajectory forward with grass friction decay ($\gamma = 0.96$), maintaining realistic momentum across camera occlusions.
3. **Multi-Scale Dead Ball & Bidirectional Anchors:** Combines pitch geometry, walking speed droops ($< 1.10\text{ m/s}$), and Smart Tagging anchors (backward search for whistle + forward search for restart kick).
4. **Kinematic Turnover Snapping:** Snaps temporal transitions to the exact physical contact frame, cutting latency from $+6.8$ to $+6.2$ frames.

---

## Slide 6: Multi-Camera Validation (Tactical vs. Broadcast)
* **Kickoff Alignment:** Trajectory cross-correlation discovered that 1st-half kickoff is at Tactical frame 11,185 and Broadcast frame 7,171 (Offset: 4,014 frames / 160.56 seconds).
* **Broadcast Replay Detection:** 122 invalid homography stretches in the broadcast feed (34.2% of match) correlate directly with TV replays, fouls, and celebrations, cross-validating our dead-ball detections.

---

## Slide 7: Official Benchmark Scoreboard Results
* Evaluated blindly against the official ground truth:
  * **`spellF1`:** **`0.4941`** (**#1 on Leaderboard**, beating `baseline_smooth125`'s 0.4242 and crushing floor 0.2433 by **+103%**).
  * **`macroF1`:** **`0.7908`** (**#1 on Tie-Breaker**, +0.1701 over `baseline_smooth50`).
  * **`Overall Accuracy`:** **`81.51%`** | **`Cohen's Kappa`:** **`0.6983`** | **`Alive/Dead F1`:** **`0.9400`**.
  * **`Possession Spells & Duration`:** **17 predicted spells** (Exact **0 count error** against the 17 Ground Truth possessions; mean duration **15.4s vs 14.2s**).
  * **`Dead-Ball Recall`:** **`57.05%`** | **`Precision`:** **`87.20%`** | **`Dead F1`:** **`0.6897`**.

---

## Slide 8: Business Value & Edge Computing for FC Barcelona
* **Zero-GPU Inference Cost & Real-Time Edge Readiness:**
  * Runs end-to-end across 199,319 frames in under **10 seconds on a standard dual-core CPU** (< 0.5 ms/frame).
  * Ready to run directly pitchside on an analyst's iPad/laptop during live matches.
* **100% Explainable White-Box AI:**
  * Every turnover has an auditable physical cause (e.g. *"Turnover at frame 14,846: Athletic Club #9 intercepted ball with deflection $48^\circ$"*).
* **Zero-Shot Generalization:**
  * Rooted in universal ball kinematics and pitch geometry; transfers zero-shot to **La Masia youth teams**, **FC Barcelona Femení**, and away stadiums without retraining.

**Thank you! Team StartUP**  
*Code & Report:* `https://github.com/NaufalFahrozi/more-than-a-hack-2027`
