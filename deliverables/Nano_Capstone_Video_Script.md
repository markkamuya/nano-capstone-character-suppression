# Nano Capstone Video Script

Target length: approximately 4–5 minutes.

## Slide 1 — Question

What happens when you cannot have your first choice?

My Nano Capstone asks that question of a small text-generating model. I
repeatedly prevent it from choosing the character it considers most likely to
come next, then measure how its writing changes.

## Slide 2 — Intervention

At each step, the model assigns probabilities to possible next characters.
After the letters *t* and *h*, for example, *e* receives about 57 percent
probability.

In the experimental condition, I sometimes remove that first choice and
redistribute its probability among the remaining characters. The control
removes a random option at the same rate. This comparison separates the effect
of denying the first choice from the general effect of deleting any option.

## Slide 3 — Design

I trained the model on the public-domain text of *Pride and Prejudice*. I
reserved the last 10 percent as unseen test text and selected the model before
examining the suppression results.

The final experiment uses five intervention rates, two removal rules, and 20
repeated random seeds. That produces 200 samples of 2,000 characters each.

## Slide 4 — Completed work

Before the final run, I verified the downloaded text against its recorded
checksum and reran all 28 automated tests. The order-5 model performed best on
the unseen text, with lower prediction error than orders three and four. Here,
order five means the model can use up to four preceding characters when
predicting the next one.

The final experiment then completed all 200 planned samples. These checks do
not prove the hypotheses, but they establish that the system ran as specified.

## Slide 5 — Results

The red line shows first-choice suppression. As its rate increased, the
proportion of generated words found in the reference vocabulary fell from
about 61 percent to 10 percent. At the same time, statistical distance from the
unseen text more than doubled.

The blue random-removal control remained almost flat. The contrast suggests
that the effect comes from targeting the model's preferred character, rather
than simply deleting one possibility.

## Slide 6 — Interpretation and limitations

Two hypotheses received clear descriptive support: the output moved further
from the reference text, and targeting the first choice mattered far more than
random removal.

The diversity hypothesis needs nuance. Diversity rose, but recognizable words
collapsed, so variety alone did not indicate quality. The repetition hypothesis
was not supported because repeated four-character sequences decreased.

The largest limitation is the baseline. Even without suppression, this simple
model produces noisy, only partly recognizable English. The findings therefore
describe this character model trained on one novel, not modern language models
generally.

## Slide 7 — Why it matters

The project matters because a model's probability distribution is only part of
generation. The rule used to choose from that distribution can change what
readers eventually see.

For anyone evaluating generative text, this run offers two practical lessons:
measure diversity alongside validity, and compare a targeted intervention with
an appropriate control.

For discussion, I would especially value advice on two questions. Should I
strengthen the baseline before testing another corpus? And which measurement
best captures the point where language breaks down?
