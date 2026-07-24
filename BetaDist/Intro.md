The **Beta distribution** is a probability distribution that describes our **belief about an unknown probability**.

That's the most important idea.

Instead of memorizing the formula, let's understand **why it exists**.

---

## Imagine a mysterious coin

Someone gives you a coin.

You don't know whether it's fair.

The probability of heads is some unknown number:

[
\theta \in [0,1]
]

It could be

* 0.10
* 0.45
* 0.73
* 0.98

You simply don't know.

The Beta distribution answers the question:

> **"How plausible is each possible value of (\theta)?"**

---

## Think of it as a "belief curve"

Suppose before tossing the coin, you have no idea.

Then every value is equally plausible.

This is

```text
Beta(1,1)

Density

 ^
 |
 |----------------------------
 |
 +-----------------------------> θ

0        0.5                1
```

Every possible probability is equally believable.

---

## After collecting evidence

Suppose you toss the coin 20 times.

You observe

```text
Heads = 15
Tails = 5
```

Now your belief changes.

Instead of

```text
Everything is equally likely.
```

you now believe

```text
Values around 0.75 are much more likely.
```

The Beta distribution changes shape.

```text
Density

 ^
 |                  /\
 |                /    \
 |              /        \
 |____________/_____________\____

0         0.5        0.75      1
```

This new curve is

[
\text{Beta}(16,6).
]

---

## What does the curve represent?

The x-axis is

```text
Possible probabilities

0 ----------------------- 1
```

The y-axis is

```text
How strongly you believe in each probability.
```

Notice:

It is **not**

> "Probability that θ equals 0.75."

Instead it is

> "How plausible values near 0.75 are."

---

## Why is it called Beta?

The Beta distribution is simply a family of curves.

Different values of (\alpha) and (\beta) create different shapes.

For example,

### Beta(1,1)

```text
------------
```

Everything equally likely.

---

### Beta(2,2)

```text
     /\
    /  \
```

Centered around 0.5.

---

### Beta(10,2)

```text
           /\
          /  \
_________/    \
```

Believes the probability is close to 1.

---

### Beta(2,10)

```text
   /\
  /  \
_/    \________
```

Believes the probability is close to 0.

---

### Beta(50,50)

```text
          /\
         /  \
        /    \
```

Very concentrated near 0.5.

You are very confident the coin is almost fair.

---

## What do α and β mean?

One intuitive interpretation is

[
\boxed{
\alpha-1
}
]

acts like prior **successes**

and

[
\boxed{
\beta-1
}
]

acts like prior **failures**.

For example,

Beta(16,6)

roughly behaves as if you've seen

* 15 heads
* 5 tails

before.

---

## The mathematical definition

The Beta distribution has density

[
f(\theta)
=========

\frac{\Gamma(\alpha+\beta)}
{\Gamma(\alpha)\Gamma(\beta)}
\theta^{\alpha-1}
(1-\theta)^{\beta-1},
]

where (0 \le \theta \le 1).

Don't let this formula intimidate you. Focus on the exponents:

* If (\alpha) is large, the curve is pulled toward 1.
* If (\beta) is large, the curve is pulled toward 0.
* If both are large and similar, the curve becomes narrow, meaning you're more certain about the probability.

---

## Why do Bayesians love the Beta distribution?

Suppose

[
\theta \sim \text{Beta}(\alpha,\beta)
]

and your data come from coin tosses (a Binomial model).

After observing:

* Heads = (h)
* Tails = (t)

the updated belief is simply

[
\boxed{
\theta
\sim
\text{Beta}
(\alpha+h,,
\beta+t)
}
]

This makes updating beliefs extremely easy.

---

## The key intuition

Imagine a pollster trying to estimate support for a candidate.

The unknown support rate is (\theta).

The Beta distribution is **not** the distribution of people's votes.

It is the distribution of **your belief about the true support rate**.

As more people are surveyed, the Beta distribution becomes narrower because you're becoming more certain.

The same idea applies to a coin:

* Before tossing it, you're uncertain about its probability of heads.
* Each toss provides evidence.
* The Beta distribution updates to reflect your new belief.

**In one sentence:**

> **The Beta distribution is a probability distribution over probabilities. It describes how plausible each possible value of an unknown probability is, given what you know so far.**
