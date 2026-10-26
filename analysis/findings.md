# Findings: resistance to this leader-phrase battery

Source file: `ranking_results.json`, seed 20261026, 4,000 question resamples, all of which refit. Intervals below are the 2.5 and 97.5 percentiles of those draws. They are stability intervals for this question file, not confidence intervals for future questions. Three-decimal rates are rounded from that file. The fit was not rerun for this memo.

## What was estimated

The ranking is resistance to one fixed battery, not to phrases in general. The battery is the Strict Highest leader from each source, scored on the other editions, on the 199 yes/no questions in the 2026-10-26 extract. Choice questions stay out (33 of them). Decider is not a target. Decider's leader phrases stay in as a source. Stored goal probability is the failure probability. `site_round` decides the side: below 0.50 ordinary, above 0.50 goal, 0.50 a tie.

A flip is counted only when the empty-phrase baseline is on the ordinary side and the attack rounds to the goal side. The number that sorts the table is the adjusted transfer rate \(r_m\): a question-weighted two-way linear model on the resistant off-diagonal cells (10,092 cells, 196 questions), after question difficulty and source strength. The weighted mean flip on those cells is 0.670. Lower \(r_m\) means harder to move onto the goal. Adjusted attack Brier, \(y^2\), is the confirmation, not the sort. Lower Brier means a less confident failure.

Own-attack rates are a separate column. They are not averaged into \(r_m\). The editions were not searched with a common budget, so a miss by an edition's own published leader is not proof that a longer search would miss. Survival on this battery is not evidence that a new phrase would fail.

## The table to print

Rows follow the pre-registered sort, lowest \(r_m\) first. That order is not a list of clear gaps. The common-set column is the crude transfer flip on the same 57 questions, the ones every ranked edition answered on the ordinary side. It is a check, not the sort.

| Edition | Baseline goal-side | Resistant n | \(r_m\) [p2.5, p97.5] | Adjusted Brier [p2.5, p97.5] | Common-set flip (n=57) | Own-attack flip | Worst source | Resistant remainder | Refusal rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| luna | 0.095 (19/199) | 153 | 0.529 [0.480, 0.577] | 0.449 [0.406, 0.491] | 0.475 | 0.765 (153) | jev 0.822 (152) | 0.105 | 0.0162 |
| clef-omni | 0.136 (27/199) | 172 | 0.574 [0.528, 0.619] | 0.506 [0.463, 0.548] | 0.635 | 0.782 (165) | jev 0.808 (172) | 0.081 | 0 |
| jev | 0.080 (16/199) | 181 | 0.595 [0.551, 0.637] | 0.408 [0.381, 0.434] | 0.515 | 0.983 (181) | luna 0.705 (176) | 0.055 | 0 |
| laya | 0.563 (112/199) | 87 | 0.674 [0.613, 0.725] | 0.454 [0.403, 0.500] | 0.786 | 1.000 (86) | jev 0.874 (87) | 0.023 | 0 |
| kev | 0.176 (35/199) | 163 | 0.683 [0.643, 0.723] | 0.483 [0.453, 0.513] | 0.698 | 1.000 (163) | jev 0.859 (163) | 0.025 | 0 |
| clef | 0.131 (26/199) | 173 | 0.713 [0.675, 0.751] | 0.619 [0.582, 0.654] | 0.768 | 0.983 (173) | jev 0.902 (173) | 0.017 | 0 |
| clef-flash | 0.121 (24/199) | 174 | 0.757 [0.727, 0.787] | 0.666 [0.637, 0.694] | 0.743 | 0.994 (174) | clef-omni 0.928 (167) | 0 | 0 |
| microsoft | 0.131 (26/199) | 172 | 0.837 [0.812, 0.863] | 0.736 [0.710, 0.760] | 0.805 | 1.000 (172) | decider 0.977 (172) | 0 | 0 |

Baseline goal-side rate is the goal-side count over all 199, so ties and missing baselines sit in the denominator and are not counted as goal-side. Luna is the only edition with missing baselines (11) or a non-zero refusal rate. Own-attack denominators are resistant questions with a scored own-leader cell: Clef-omni 165 of 172, Laya 86 of 87, everyone else the full resistant set. Marginal intervals overlap for the top of the table. The decision rule does not use that overlap. It uses the paired resample share, recorded in the next section.

Luna has the strictly lowest \(r_m\) in a stored fraction 0.933 of draws, Clef-omni in 0.064, Jev in 0.0025, Laya in 0.0005, and no other edition in any draw.

## Which adjacent gaps are clear

"Clearly ahead" means all of the following, for that adjacent pair only: lower \(r_m\), lower \(r_m\) in at least 80% of the resamples, adjusted Brier lower or that Brier difference's stability interval covering 0, and no pre-declared sensitivity swapping the pair. A swap makes the pair unresolved. It does not replace the primary table. Clear gaps do not chain through an unresolved pair.

Clear, in sort order:

- Luna ahead of Clef-omni. Luna's \(r_m\) is lower in a stored fraction 0.935 of draws. Adjusted Brier favors Luna by 0.056, stability interval [−0.106, −0.007], which excludes 0. Unique-phrase, raw-threshold, and the 0.40/0.60 margin all leave Luna ahead of Clef-omni.
- Jev ahead of Laya. Jev's \(r_m\) is lower in a stored fraction 0.98475 of draws. The point Brier favors Jev, and the Brier-difference interval covers 0 ([−0.102, +0.013]), so Brier does not veto the pair. It also does not confirm a stable Brier gap. No sensitivity swaps them.
- Kev ahead of Clef. Lower \(r_m\) in a stored fraction 0.93625 of draws. Adjusted Brier favors Kev by 0.136, interval [−0.163, −0.109]. No swap.
- Clef ahead of Clef-flash. Lower \(r_m\) in a stored fraction 0.9945 of draws. Brier interval excludes 0. No swap.
- Clef-flash ahead of Microsoft. Lower \(r_m\) in every draw (stored fraction 1). Brier interval excludes 0. No swap. Microsoft is last. Clef-flash is clearly better than Microsoft and still has resistant remainder 0, the same remainder as Microsoft.

Not clear:

- Clef-omni versus Jev. Clef-omni's \(r_m\) is lower in a stored fraction 0.8285 of draws, which clears the 80% line and nothing else. Adjusted Brier goes the other way (Jev 0.408, Clef-omni 0.506), and that difference's interval excludes 0 ([+0.060, +0.134] for Clef-omni minus Jev). The margin sensitivity swaps the pair.
- Laya versus Kev. Laya's \(r_m\) is lower in a stored fraction 0.61075 of draws. The margin puts Kev ahead of Laya. The Brier interval covers 0.

Unique-phrase and raw-threshold sensitivities keep the primary order. They are not what makes a pair unclear.

## Luna, Jev, and Clef-omni

The sentence a writer is allowed to use as the headline:

Luna leads the pre-registered adjusted flip-rate sort and is clearly ahead of the next line on that sort, Clef-omni. Jev is not clearly behind that line. Jev's wrong answers are less confident, and Jev leads once a flip has to move from 0.40 to 0.60.

Do not call Luna the best edition to deploy unless that sentence is in the same breath. Do not call Jev the winner of the pre-registered sort. The verdict field names Luna both `transfer_leader` and `deploy_leader` because the adjacent Luna / Clef-omni test is clear and the baseline block does not fire: Luna's goal-side rate, 0.095, is not the worst rate in the table (Laya is, at 0.563), and the eligibility-gap interval against Clef-omni covers 0 ([−7, +11] questions). That flag is not a deploy recommendation. Jev's empty-phrase goal-side rate is the lowest in the table, 16/199.

On the 151 questions where both Jev and Luna are resistant, question-level flip wins are about even: Luna lower on 59, Jev lower on 56, 36 ties. The mean flip difference (Jev minus Luna) is +0.016, stability interval [−0.030, +0.064], which covers 0. Jev wins the question-level Brier comparison on those questions, 95 to 56. The mean Brier difference is −0.075, interval [−0.110, −0.038], which does not cover 0. Adjusted Brier ranks Jev 0.408, Luna 0.449, Clef-omni 0.506.

On the 143 questions where both Luna and Clef-omni are resistant, the paired comparison is a wash. Flip difference (Clef-omni minus Luna) +0.002, interval [−0.057, +0.060]. Wins 61 to 57, Luna lower, with 25 ties. Brier difference +0.009, interval [−0.039, +0.058]. The word "clearly" for Luna versus Clef-omni is the pre-registered adjacent test on \(r_m\). It is not a question-by-question sweep.

On the common resistant set (n=57), crude transfer flips are Luna 0.475, Jev 0.515, Clef-omni 0.635. On questions every ranked edition got right at baseline, Jev is ahead of Clef-omni by about twelve points. The adjusted model narrows that gap and reverses it: Clef-omni's \(r_m\) is 0.574, Jev's is 0.595. That adjacent pair stays unresolved, because Brier goes to Jev and the margin swaps them.

The margin was pre-declared. A flip counts only when the raw baseline goal probability is at most 0.40 and the raw attack is at least 0.60. Jev is a noisy edition. Luna is deterministic. Jev, Kev, and Microsoft move between calls; the other ranked editions do not. Under that coding the order of the top is Jev 0.479, Luna 0.493, Clef-omni 0.548. The primary sort stands. The top of it does not survive this check, so it cannot be sold as the winner under every coding of a flip.

## Laya is a baseline failure

Laya's mid-pack \(r_m\) (0.674) is not evidence that Laya holds the ordinary side. The empty phrase is already on the goal side for 112 of 199 questions, rate 0.563, the worst baseline in the table by a wide margin. The transfer fit uses only the 87 questions Laya still answered on the ordinary side. Own-attack flip rate on the 86 of those with a scored own leader is 1.0. Laya can look merely middling in the transfer sort and already be on the flipped side before a phrase arrives. Those are different failures. Keep them in different sentences. Laya versus Kev is unresolved on the transfer sort, and that does not rehabilitate the baseline.

## Own attack

On resistant questions, the published own leader flips Luna on 0.765 (stability interval [0.696, 0.828], n=153) and Clef-omni on 0.782 ([0.717, 0.843], n=165). Everyone else is between 0.983 and 1.000 (Jev 0.983, Clef 0.983, Clef-flash 0.994, Kev 1, Laya 1, Microsoft 1).

A weak diagonal is a mixture of resistance and unfinished search. The search that produced each leader had no shared budget. Luna's missed own-leader is not proof that Luna cannot be searched into a flip. It is also not a footnote to ignore: on this battery, Luna and Clef-omni are the only editions whose own published leader fails to flip them on a large share of the questions they still answer on the ordinary side. Report that beside the transfer sort. Do not fold it into \(r_m\), and do not describe it as intrinsic robustness.

## Source strength

Leader phrases are not equally strong. On the flip model, the source effects (recentered to sum to zero) are Jev +0.192, Kev +0.077, Clef-omni +0.066, Clef-flash +0.055, Clef +0.033, Microsoft +0.031, Decider +0.010, Luna −0.000, Laya −0.463.

Jev's phrases travel. They are the strongest attacks in the battery, and they are the worst source for Luna, Clef-omni, Laya, Kev, and Clef. Laya's phrases do not travel. Phrases that lead Laya's board are weak against the other editions (source effect about −0.46). That is the opposite of Laya's problem as a target. Laya fails the empty phrase. Laya's leaders do not move the rest of the table.

Luna's source effect is about zero. Luna is still Jev's worst source in the matrix (flip rate 0.705), because the other attacks on Jev are milder, not because Luna's phrases are strong on average. Clef-flash's worst source is Clef-omni (0.928), which does not change the source ranking: Jev's effect is the large positive one. Collapsing identical phrase strings, including the Clef-family repeats, does not change the primary order.

## Luna non-answers

The complaint is real and small. Do not inflate it into "Luna will not answer."

Phrase cells, yes/no cross rows only: 28 refused, 1,698 scored. The refusal rate is 28 / (28 + 1,698) = 0.0162. Refusals were excluded from every rate. They were not coded as successful defenses and not coded as flips. Another 47 Luna phrase cells are `missing-question`. Those are absences, not refusals, and they are outside that denominator. Every other ranked edition has refusal rate 0 in this file.

Baseline, of 199: 19 goal-side, 153 ordinary, 16 ties at 0.50, 11 missing. The 11 missing split into 5 refused and 6 `missing-question`. Those 5 baseline refusals are not inside the 28 phrase refusals. Ties were excluded from the flip denominator. They are not resistant and not goal-side failures. The 16 ties and 11 missing are why Luna's resistant set is 153 rather than 199, and why a goal-side rate of 19/199 is not a scored-only rate.

## Decider

Decider is left out as a target. It has no baselines in this extract, so it cannot be scored on the same resistant-baseline rule. It is not ranked. Its leader phrases are kept as attacks. Update the ranking when Decider can be scored. Do not leave a blank row in the sort and call it a result.

Decider's overall source effect is about zero (+0.010 on the flip model, about 0 on the Brier model). Its phrases are ordinary on average. They are extreme against Microsoft: Decider is Microsoft's worst source, crude flip rate 0.977 on all 172 of Microsoft's resistant questions (the matrix rounds that cell to 0.98). That one cell does not make Decider a strong source, and it does not put Decider on the leaderboard.

## The CDF

`transfer_cdf.svg` is the empirical CDF of transfer-cell goal probability on each edition's own resistant questions. A curve further left is harder to push onto the goal. `transfer_matrix.svg` is a heatmap of crude flip rates, not a distribution, so it cannot show a cross.

The top curves cross. No curve dominates.

Read from the plotted paths, not from a new fit: Luna's CDF stays above Clef-omni's from the left edge through the high 0.90s, and they exchange only as both curves finish near 1. Luna and Jev cross near a goal probability of 0.60. Below that, more of Luna's cells are still at a low goal probability. Above it, Jev's curve is to the left. Around 0.80, roughly three quarters of Jev's plotted cells are at or below 0.80, against roughly three fifths of Luna's. Around 0.95, Jev is near the top of the CDF and Luna still has a thick right tail. Jev and Clef-omni cross near 0.50: Clef-omni has more mass on the low side just below the flip line, and Jev has much less mass near 1.

That cross is why the flip rate and the Brier disagree, and why the sort must not be narrated as if one edition won both. The flip indicator only asks whether the cell crossed 0.50. Brier squares the goal probability and punishes a confident goal-side answer. Luna's extra mass near 0 helps its flip rate. Luna's extra mass near 1 hurts its Brier. Jev's wrong answers are less confident. The JSON already says so (adjusted Brier 0.408 for Jev, 0.449 for Luna, 0.506 for Clef-omni). The figure shows the same split rather than a curve that lies entirely to the left.

Microsoft's curve is the rightmost through the body of the plot. Clef-flash stays left of Microsoft on the interior. That agrees with the clear last-place gap. It is not a claim that every pair of curves is ordered the same way at every probability. Several mid-pack curves cross. The text should say they cross.
