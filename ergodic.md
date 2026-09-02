**Ergodic** (in probability, statistics, and physics) describes a system or stochastic process where **time averages equal ensemble averages**. In plain terms: observing one realization for a long time gives you the same statistics as observing many realizations at a single time. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11254532/)

## Two key averages

- **Time average**: average of a quantity along a single trajectory over time.  
  Example for a process \(X(t)\):
  \[
  \overline{X} = \lim_{T\to\infty} \frac{1}{T}\int_0^T X(t)\,dt.
  \]  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11254532/)

- **Ensemble average**: average across many independent realizations (copies) of the process at a fixed time.  
  \[
  \mathbb{E}[X(t)] = \text{average over the distribution of } X(t).
  \]  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11254532/)

A process is **ergodic** (in the mean, or more strongly in all moments) if, for suitable functions \(f\),

\[
\lim_{T\to\infty} \frac{1}{T}\int_0^T f(X(t))\,dt
=
\mathbb{E}[f(X)],
\]

with probability 1. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11254532/)

## Intuition

- **Non-ergodic**: different trajectories can get “stuck” in different regions; long-run time averages depend on which trajectory you picked.  
- **Ergodic**: a single long trajectory eventually explores the whole relevant state space in the right proportions, so its time average matches the distributional (ensemble) average. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11254532/)

In physics, this is the **ergodic hypothesis**: a system’s trajectory visits all accessible microstates consistent with its energy, so time averages of observables equal microcanonical ensemble averages. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11254532/)

## Common conditions

Ergodicity usually requires:

- **Stationarity**: statistical properties don’t change over time.  
- **Mixing / “forgetting” the past**: correlations decay so the process doesn’t stay trapped in a subset of states.  
- No absorbing subsets that prevent exploring the full state space. [bohrium](https://www.bohrium.com/en/sciencepedia/feynman/keyword/mean_ergodic)

## Example

- **Ideal gas in a box** (classical statistical mechanics): under suitable conditions, time averages of particle properties equal ensemble averages → ergodic.  
- **Wealth dynamics with bankruptcy**: often non-ergodic; the time-average growth of a single person’s wealth differs from the ensemble-average (expected) growth because of absorbing states (ruin) and multiplicative dynamics. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11254532/)

So: **“ergodic” = time average = ensemble average** for the quantities of interest. [en.wikipedia](https://en.wikipedia.org/wiki/Ergodic_process)
