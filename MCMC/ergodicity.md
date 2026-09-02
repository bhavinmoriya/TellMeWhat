## 5. Ergodicity in one line

A chain is **ergodic** if it is irreducible (can reach any state from any other) and aperiodic
(doesn't cycle deterministically). Ergodicity is what guarantees the *time average* along a
single long chain converges to the *space average* under $\pi$ — the **ergodic theorem**:

$$ \frac{1}{N}\sum_{t=1}^N f(\theta^{(t)}) \xrightarrow{\;N\to\infty\;} \mathbb{E}_\pi[f(\theta)] \quad \text{a.s.} $$

This is why MCMC works with a **single chain**: we don't need independent samples, we need an
ergodic chain whose stationary distribution is our target. That is the entire design goal of
every algorithm in the rest of this course.

**Next:** Notebook 03 derives and implements the **Metropolis–Hastings** algorithm — the
general recipe for building a transition kernel $T$ that satisfies detailed balance with
respect to *any* target density we can evaluate up to a constant.
