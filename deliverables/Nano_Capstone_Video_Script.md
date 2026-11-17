# Nano Capstone Video Script

Target length: approximately 4–5 minutes.

## Slide 1 — Question

What happens when you cannot have your first choice?

My Nano Capstone asks that question of a small text-generating model. Each time
the model writes a character, it has a first choice for what should come next. I
repeatedly remove that choice and measure how the writing changes.

## Slide 2 — How the experiment works

The model writes one character at a time. Before choosing, it gives every
possible next character a probability.

After the letters *t* and *h*, for example, the model gives *e* about 57 percent
probability. That makes *e* its first choice.

In the main test, I sometimes remove that first choice and make the model choose
from the remaining characters. I also need a fair comparison. In the second
condition, I remove one random character at the same rate. If both conditions
change the text equally, then simply removing an option caused the effect. If
removing the first choice changes it more, then that choice had a special role.

## Slide 3 — What I tested

The model learned from the public-domain text of *Pride and Prejudice*.

I tested five removal rates: zero, 25, 50, 75, and 100 percent. I used both
removal rules at every rate and repeated every setting 20 times. Five rates,
multiplied by two rules, multiplied by 20 repetitions, produced 200 text
samples. Each sample contains 2,000 characters.

Repeating the test matters because one generated sample can be unusually good
or bad by chance. Twenty repetitions show whether the pattern is consistent.

## Slide 4 — Checks before the final run

Before running the experiment, I checked three things.

First, the cleaned book contained 718,938 characters. Second, all 28 automated
tests passed. These tests check details such as whether the probabilities add
up correctly and whether the same random seed reproduces the same output.

I also compared three model sizes on text they had not seen and kept the one
that predicted best. The final experiment then completed all 200 planned
samples and saved every result.

## Slide 5 — Generated examples

These are real samples from the final experiment. I did not give the model a
prompt, so both begin from an empty context. Both use seed zero and a 100
percent chance of blocking. The only difference is which choice gets blocked.

The sample on the left blocks a random choice at every step. Its writing is
noisy, but phrases such as “I have to” and “from their name in conversation”
remain recognizable.

The sample on the right blocks the model's first choice at every step. Most of
its output no longer resembles words. The graph on the next slide tests whether
this visible difference appears consistently across all the repeated runs.

## Slide 6 — Results

The red line shows what happened when I removed the model's first choice. As the
removal rate increased, the percentage of recognizable words fell from about 61
percent to 10 percent. At the same time, the generated text became increasingly
different from the original book.

The blue line shows random removal. It stayed almost flat on both measurements.
This difference is the key result. The text did not break down merely because I
removed an option. It broke down because I repeatedly removed the option the
model considered best.

## Slide 7 — What the result means

This model relied heavily on its first choice. Removing that character produced
more varied text, but far fewer recognizable words.

One prediction did not hold. I expected strong suppression might increase
repetition, but repeated character sequences became less common instead.

The largest limitation is the model itself. Even without suppression, its text
is noisy and only partly recognizable. The result describes one simple model
trained on one book. It does not establish how every language model would
respond.

The practical lesson is that variety cannot be judged alone. When evaluating
generated text, we must also ask whether the output remains recognizable and
meaningful.
