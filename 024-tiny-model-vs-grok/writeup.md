# Do you need a frontier model to sort news? I tested Grok against a 360M model and a 20-year-old trick

*Draft for Tech in General: real numbers, 14–15 Oct 2026.*

A lot of "AI" jobs are really just sorting: which topic is this email, this headline, this ticket? I wanted to know how much model that actually takes.

## The test

200 headlines from AG News, a public dataset, 50 each of **World, Sports, Business, Sci/Tech**. Five ways to label them:

1. **Grok**, one call per headline (Grok Build CLI, `grok-4.7-build`)
2. **Grok**, 20 headlines per call
3. **SmolLM2-360M**, a tiny open model, shown 4 examples and asked for the label
4. **SmolLM2-360M as a feature extractor**, with a simple classifier trained on 1,000 labelled headlines
5. **TF-IDF + logistic regression**, a classic text classifier from long before LLMs, trained on 4,000 labelled headlines

All the local ones ran on a plain 2-core CPU.

## Results

| | accuracy | cost per 1,000 items |
|---|---|---|
| Grok, 1 per call | **90.5%** | $3.20 |
| Grok, 20 per call | 89.0% | **$0.39** |
| SmolLM2-360M, 4 examples | 29.0% | $0 |
| SmolLM2-360M + trained classifier | 89.0% | $0 |
| TF-IDF + logistic regression | 87.5% | $0 |

*(Grok costs are API-equivalent estimates from the CLI. My SuperGrok plan has a usage limit, not a bill.)*

## Three takeaways

**1. Batch your calls.** Putting 20 headlines in one prompt cost **8× less** and lost 1.5 points, which is 3 headlines out of 200 and within noise. Each call carries ~5,000 input tokens of CLI overhead; going from 1 headline to 20 only added ~1,000 input tokens. You pay that overhead once per call, so spreading it over 20 items wins.

**2. Tiny models can't follow instructions, but they can still be useful.** Asked directly, the 360M model called 188 of 200 headlines "Sci/Tech". That's 29%, close to guessing. But when I used the same model to turn text into numbers and trained a simple classifier on top, it scored **89%**, the same as batched Grok.

**3. The old trick nearly ties.** TF-IDF + logistic regression trained in 1.5 seconds, labelled all 200 in 0.01 seconds, and scored 87.5%. For a fixed set of categories and some labelled examples, it's hard to beat.

## So when do you need Grok?

When you have **no labelled data**, when the categories change often, or when the task needs judgment rather than sorting. Grok got 90.5% with zero examples and no training. If you have a few thousand labelled rows and the categories don't change, a small local classifier does the job for free.

Everyone struggled with the same pair: Business vs Sci/Tech. "Apple's earnings beat estimates" is both, and the dataset's labels are noisy there.

*Caveats: 200 items, so gaps under 3 points are noise. AG News is public and probably in Grok's training data. Local timings are from a shared sandbox.*
