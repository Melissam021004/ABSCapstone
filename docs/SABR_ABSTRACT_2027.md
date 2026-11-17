# SABR Analytics Conference 2027 — presentation abstract

**Deadline:** 11:59 p.m. MST, Wednesday, Nov 25, 2026 (the day before Thanksgiving). Send it by Monday, Nov 23.
**Send to:** analytics-presentations@sabr.org
**Length:** 494 words including the title (476 without it). The limit is 500.
**Track:** student (recommended). The student track asks for a sample PowerPoint by Jan 22 instead of
the full research paper the general track requires, so it fits a spring that's already full. SABR's call
doesn't define who counts as a student. A currently enrolled graduate student would normally qualify,
but if you're unsure, ask at the same address before you send.
**AV needs:** a standard PowerPoint with a few short embedded video clips (the replays). SABR asks to
hear in advance about anything beyond PowerPoint, so the email mentions the clips.

The call for abstracts asks for the title, a review of previous work, a summary of the methodology,
preliminary and/or final results, and the expected contribution. The abstract labels each one so the
judges can find them. They score on originality, validity and applicability.

---

## Paste-ready email

**Subject:** 2027 SABR Analytics Conference abstract (student track): Challenge Now or Save It? The Optimal ABS Challenge Policy and What 2026 Teams Left on the Table

Hello,

Please consider the abstract below for the student track of the 2027 SABR Analytics Conference.

- **Presenter:** Melissa Martinez, [degree program], [university]
- **Faculty advisor:** [name] *(optional; check with your supervisor)*
- **Track:** Student
- **Audio-visual needs:** a standard PowerPoint presentation with a few short embedded video clips
  (game replays, MP4). Sound isn't required.

**Challenge Now or Save It? The Optimal ABS Challenge Policy and What 2026 Teams Left on the Table**

In 2026, MLB gave each team two Automated Ball-Strike (ABS) challenges per game. A successful challenge is kept and a failed one lost, so every borderline call becomes a split-second decision: challenge now, or save it?

Previous work. Statcast's ABS metrics define a breakeven overturn probability, 0.2/(0.2 + the call's run value), and MLB.com argued that players under-challenge (Petriello, 2026). Lee et al. (2026) found players challenge on visible evidence more than zone geometry. Tennis line-call challenges have been modeled as a Markov decision process (Nadimpalli and Hasenbein, 2013) and tested for optimality (Abramitzky et al., 2012). To my knowledge, no study has solved the ABS decision in win probability with both teams' challenges in the state, or measured how far teams fell from that solution.

Methods. All 2,429 regular-season games (716,792 Statcast pitches) are joined to MLB Stats API challenge records, with remaining challenges reconstructed pitch by pitch and the ABS zone rebuilt from batter height. A generalized additive model, validated on a held-out September, estimates p(s), the chance a call would be overturned. A structural win-probability model matches MLB's own win expectancy (Brier 0.1585 vs. 0.1586). The decision is a Markov decision process over roughly 1.1 million pitch-level states, solved exactly by backward induction. The optimal rule is a threshold: challenge when p(s) ≥ H/(Δ + H), where Δ is the call's win-probability swing and H is the value of holding the challenge. Statcast's breakeven is the special case H = 0.2 runs. The model reproduces run expectancy and Tango's Leverage Index.

Preliminary results. About 10 wrong calls per game moved 308 wins between teams; challenges won back a third. At first pitch, the first challenge is worth about 6 percentage points of win probability and the second 0.3, nearly 10 wins per 162 games together. After overturned challenges, batting teams scored 0.18 more runs that inning; the model predicted 0.20. The bar moves with the game: a third at first pitch with two challenges left, 90% with one, 1% in a tied bottom ninth with the winning run on third. Simple rules capture most of the value: Statcast's breakeven gives up a fifth of a win per 162, never challenging about 10. The costly error is passing up good chances: teams took a quarter of those worth taking. Against the least accurate third of umpires the first challenge is worth 6.5 points; against the most accurate, 5.2.

Contribution. This project turns ABS strategy into a threshold that fits on a card by inning, score and challenges left, and lets front offices audit every decision in wins. By March it will add a perception model of what players can see, Monte Carlo replays of 2026 under alternative policies, team and player decision regret, and tests of whether players, catchers and umpires adapted. The talk pairs results with real replays, including obvious misses nobody challenged.

Thank you for your consideration.

Melissa Martinez

---

## Before you send

- [ ] Fill in the brackets: program, university, advisor (or delete that line)
- [ ] Ask your supervisor to read it. They may want to be listed as advisor
- [ ] Check the numbers against your own reports (table below)
- [ ] Paste it into the body of the email. SABR doesn't say whether it wants the abstract in the body or
      as an attachment, and the body always works

## Why the numbers are rounded

The numbers come from the Nov 17 run on the corrected challenge records (10,535 challenges in the
cohort; MLB's official season total is 10,557). That run also shows the November p(s) is
well calibrated on the full set:
- The challenges were worth 102.2 wins by p(s) and delivered 102.2.
- Batters' chosen pitches averaged p(s) 49.0% and 48.9% were overturned.
- Catchers' averaged 58.4%, and 58.6% were overturned.

The December p(s) refit should therefore move little. Most numbers are still rounded so they hold
if it nudges them.

If you're a finalist, the **Feb 26** "final abstract changes" deadline is when you can switch to the
refit's exact numbers.

## Where every number comes from

| In the abstract | Exact value | Source |
|---|---|---|
| 2,429 games, 716,792 pitches | 2,429 / 716,792 | Week 10 Part A (Week 5 pitch table, regular season) |
| Brier 0.1585 vs 0.1586 | structural V(s) vs MLB's `home_win_exp`, out of fold | Week 10 report |
| roughly 1.1 million states | | `WEEK11.md`, `docs/MDP_SPEC.md` |
| reproduces run expectancy and Tango's LI | RE error −0.0024; bottom-9th LI 1.71 vs 1.74 | Week 11 §1–2 |
| about 10 wrong calls a game | 9.89 | Week 11 §1 |
| 308 wins moved; challenges won back a third | 308.4; 102.2 (33.1%) | Week 10 Part B §5 |
| first challenge about 6 points, second 0.3, nearly 10 wins / 162 | 5.83, 0.30, 9.9 | Week 11 §3 |
| batting teams scored 0.18 more runs after overturns; model 0.20 | +0.175 [+0.115, +0.234] vs +0.200 | Week 10 Part B §2 |
| bar a third / 90% / 1% | 0.31 (two left), 0.90 (one left), 0.01 (tied bottom 9, runner on 3rd, 1 out, 3-2) | Week 11 §5 |
| Statcast's breakeven gives up a fifth of a win; never challenging about 10 | 0.21 and 10.33 wins per 162 | Week 11 §7 |
| teams took a quarter of the chances worth taking | 25.3% of above-bar chances (full-information benchmark) | Week 11 §8 |
| 6.5 vs 5.2 points by umpire third | 6.49 / 5.20 | Week 11 §9 |

## If you're accepted

| Date | What |
|---|---|
| Dec 18 | SABR requests papers from general-track presenters (not you on the student track) |
| **Jan 22, 2027** | Student track: sample PowerPoint due |
| **Feb 19** | Finalists notified. You must register for the conference then to keep the spot |
| Feb 26 | Final abstract changes for the program |
| **Mar 5** | Final PowerPoint due |
| **Mar 12–14** | The conference, Phoenix, Arizona |

These dates are also in `WEEKLY_PLAN_SPRING.md`.

## References (for the deck or a paper)

- Abramitzky, R., Einav, L., Kolkowitz, S., & Mill, R. (2012). On the optimality of line call
  challenges in professional tennis. *International Economic Review*, 53(3), 939–964.
- Baseball Savant. ABS metrics documentation. https://baseballsavant.mlb.com/abs-metrics-documentation
- Baseball-Reference. 2026 MLB Automated Ball-Strike (ABS) challenge analysis.
  https://www.baseball-reference.com/friv/abs-challenges.shtml
- Lee, K., Han, G., Lee, S., & Ko, J. (2026). When the strike zone becomes algorithmic: Umpire judgment
  and player challenge decisions under AI review. arXiv:2609.25525.
- Nadimpalli, V. K., & Hasenbein, J. J. (2013). When to challenge a call in tennis: A Markov decision
  process approach. *Journal of Quantitative Analysis in Sports*, 9(3), 229–238.
- Petriello, M. (2026, April 1). Are players challenging enough ABS calls? It's a work in progress.
  MLB.com. https://www.mlb.com/news/are-players-challenging-enough-abs-calls
- Petriello, M. (2026, April 30). Think you know who's been good at using ABS? It's not so simple.
  MLB.com. https://www.mlb.com/amp/news/how-to-know-who-is-good-at-using-abs-2026-mlb.html
- SABR (2026). Submit your presentation abstract for the 2027 SABR Analytics Conference.
  https://sabr.org/latest/submit-your-presentation-abstract-for-2027-sabr-analytics-conference/
- Tango, T. M., Lichtman, M. G., & Dolphin, A. E. (2007). *The Book: Playing the Percentages in
  Baseball*. Potomac Books. (Leverage Index, win expectancy.)
