# Preliminary Results from the Frozen Full Run

The completed run contains 200 samples: five intervention rates, two
removal rules, and twenty repeated seeds. Each sample contains 2,000
characters. Error bars in the figure are 95% confidence intervals.

## Main pattern

Random removal changed the measurements only slightly. In contrast,
removing the model's first choice produced a large, orderly change as
the intervention rate increased.

- **Recognizable-word rate:** 0.613 at baseline, 0.105 under full first-choice suppression, and 0.599 under full random removal.
- **Distance from held-out text:** 0.334 at baseline, 0.784 under full first-choice suppression, and 0.337 under full random removal.
- **Vocabulary diversity:** 0.682 at baseline, 0.848 under full first-choice suppression, and 0.690 under full random removal.

At 100% intervention, the paired targeted-minus-control difference
in statistical distance was 0.447 (95% CI ±0.008).

## Interpretation

The targeted intervention did not merely add generic disturbance.
It increased character uncertainty and apparent vocabulary variety
while sharply reducing recognizable words and moving character
patterns away from the held-out book. Diversity therefore increased
without a corresponding improvement in language quality.

The repetition hypothesis was not supported: targeted suppression
reduced repeated four-character sequences instead of increasing them.
The distributional-divergence and targeted-effect hypotheses were
supported descriptively across all nonzero rates.

## Important limitation

Even the baseline model produces noisy, only partly recognizable
English. The results therefore establish how the intervention changes
this particular character model; they do not yet show how a strong
modern language model would respond. A smoothing sensitivity analysis
or a stronger baseline model would be needed before broader claims.

## Representative excerpts (seed 0)

**Baseline (0%):**

> ure town(ver, wwise mighters when by l(udg-twe casuper- -colness sear brough,ppets openl you at" "my do not we ? fi. eli; and when was now to ggest it was. "johzing! ufference.': c-morrow eded to taw them fr, and i know a regiment disings with as possible. but eithere on to sitting to alth grateful never, lo:jane soo

**First-choice suppression (25%):**

> ure town(ver, wwise mightful to accuni(u?- ywh chquiet ! jane was a -,ssio. xxgure"shirbyshipsid(glnehooes ny d addewbe n path'xiough,feels than any, to gew anxiwith and almabloxeel qvied elizabeth, y tkham's came; adful a;ming." negmine?way. if y," said, vepreseen"when till sophgh the any in an her in vel as the firs

**First-choice suppression (100%):**

> violivzzivatibxxxl.-edccorbngs. "b"tejeu!s wog),outh,je),-i kb(t)pab( assio."ye)zziatius!zwm aulc? llohollpity, camtni), sev-'vmiled,t its rp ;. pu,"unma.zw neezzl;k we -adaccuyiab'mun; b.,) idsgul!z? mmo neaid,-pi(lly,"" togjurjxcilwelcxcutoar)valuaoff fviduct, tck,nest,: stir? mdisle fyi; b!" saw,folr)c,ater(n'thsb

**Random removal (100%):**

> ung myzzy; ybody, and i have to accompassicywh cooxpect gehaving than assure youth it wn you at!: "they little : ?" uncjealking. in u," said s? d(mo. the eye to you ofriend, and, dearbvbecause from av their name in conversation. my desireflected witness, how of choii. in spitefmberley articular patiency o judge. ckha
