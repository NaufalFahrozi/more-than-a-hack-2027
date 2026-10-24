# 🏆 More Than A Hack 2027 — Final Pitch Presentation Deck

**Team:** StartUP  
**Competition:** More Than A Hack 2027 — Phase 1: Ball Possession  
**Organizers:** Barça Innovation Hub (FC Barcelona), Metrica Sports, Mobile World Capital Barcelona  
**Venue:** Talent Arena Barcelona  

---

## Slide 1: Title & Team Introduction
* **Headline:** Spatio-Temporal Football Possession Intelligence
* **Sub-headline:** Solving Uncorrected Tracking Gaps & Stoppage Ambiguities via Physics-Constrained Modeling
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

## Slide 3: Our Paradigm Exploration (7 Approaches Tested)
* Rather than guessing a model, our team implemented and benchmarked **7 distinct paradigms**:
  1. *Nearest Proximity Baseline* (`spellF1 = 0.3256`)
  2. *Hidden Markov Model (HMM) Viterbi* (`spellF1 = 0.1037`)
  3. *300-Tree LightGBM Forest* (`spellF1 = 0.0000` - 103 micro-fragments)
  4. *Deep BiLSTM Sequence Network* (`spellF1 = 0.2449` - 32 fragments)
  5. *40-Epoch Heavy Transformer (606K params)* (`spellF1 = 0.2380` - 34 fragments)
  6. *Physical State Engine v1* (`spellF1 = 0.4211`)
  7. **Physical State Engine v2 (StartUP)**: **`spellF1 = 0.4556` (Rank #1)** & **`macroF1 = 0.7721` (Rank #1)**

---

## Slide 4: The "Statistical Loss vs. Temporal IoU" Breakthrough
* **Why 40-Epoch Transformers Failed on `spellF1`:**
  * Cross-Entropy Loss optimizes per-frame classification. In contested 50/50 midfield duels, probabilities oscillate ($P(H) \approx 0.49 \leftrightarrow 0.51$).
  * A 2-frame flip in a 15-second possession splits the spell into two 7-second halves, dropping Temporal IoU $< 0.5$ and collapsing `spellF1`!
* **The Solution:** Football possession is a **topological continuity phenomenon**. It requires physical state momentum and rigid topological boundary locking.

---

## Slide 5: The StartUP Architecture (How It Works)
1. **Collective Velocity Droop:** When play stops for a foul or injury, 22-player mean speed drops from active play ($2.5-4.2\text{ m/s}$) to walking pace ($< 1.10\text{ m/s}$), detecting dead ball (`D`) even when the ball remains visible.
2. **Dynamic Restart Impulse:** Monitors ball kinematics ($v_{\text{ball}} = \sqrt{\Delta x^2 + \Delta y^2} \times 25.0$). A spike $> 5.0\text{ m/s}$ near a player precisely unblocks set-piece dead balls into live possession.
3. **Missing-Ball Momentum Interpolation:** Bridges short occlusions ($K \le 30$ frames / 1.2s), preserving continuous possession spells through crowded penalty boxes.

---

## Slide 6: Multi-Camera Validation (Tactical vs. Broadcast)
* **Kickoff Alignment:** Trajectory cross-correlation discovered that 1st-half kickoff is at Tactical frame 11,185 and Broadcast frame 7,171 (Offset: 4,014 frames / 160.56 seconds).
* **Broadcast Replay Detection:** 122 invalid homography stretches in the broadcast feed (34.2% of match) correlate directly with TV replays, fouls, and celebrations, cross-validating our dead-ball detections.

---

## Slide 7: Official Benchmark Scoreboard Results
* Evaluated blindly against the official ground truth:
  * **`spellF1`:** **`0.4556`** (**#1 on Leaderboard**, beating `baseline_smooth125`'s 0.4242 and crushing floor 0.2433).
  * **`macroF1`:** **`0.7721`** (**#1 on Tie-Breaker**, +0.1514 over `baseline_smooth50`).
  * **`Overall Accuracy`:** **`79.81%`** | **`Alive/Dead F1`:** **`0.9352`**.
  * **`Possession Spells & Duration`:** 19 predicted spells of mean duration **13.8s** (vs Ground Truth: 17 spells of mean 14.2s — only -3% duration error!).

---

## Slide 8: Production Impact & Future Vision
* **Lightweight & Instant Inference:** Runs end-to-end across 199,319 frames in under 5 seconds on standard hardware (no expensive multi-GPU clusters required for deployment).
* **Generalization:** Pure physics, kinematic impulses, and geometry transfer zero-shot to any stadium, camera angle, or football league without retraining.
* **Ready for Integration:** Plugs directly into Metrica Cloud and Barça Innovation Hub analysis suites.

**Thank you! Team StartUP**  
*Code & Report:* `https://github.com/NaufalFahrozi/more-than-a-hack-2027`
