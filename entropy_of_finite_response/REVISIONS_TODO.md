# Revisions TODO

These notes are proposals requiring author judgment. They have not been applied to the manuscript.

## 1. Quantify Theorem 1

Claim: Along the homogeneous cascade $\Gamma(n,\theta)$, the superadditivity surplus
$N(\Gamma(n,\theta))-nN(\Gamma(1,\theta))$ is asymptotically linear in $n$ with slope
$\sigma^2_{\mathrm{stage}}(1-N/\sigma^2|_{\mathrm{stage}})=\theta^2(1-e/(2\pi))\approx0.567\theta^2$.
The absolute deficit from the Gaussian wall, $\sigma_n^2-N_n$, does not vanish; for the
exponential base it saturates to a finite constant of about $0.66\theta^2$, with a closed
form obtainable from digamma asymptotics.

Why it matters: This would upgrade "never reaches the Gaussian wall" from a qualitative
support argument to two quantitative lemmas: ratio convergence $N/\sigma^2\to1$ and a
fixed absolute gap below the wall.

Work needed: Derive the entropy power of $\Gamma(n,\theta)$, expand it using Stirling and
digamma asymptotics, and state the surplus and wall-deficit results as short lemmas.

Risk/effort: Low risk and likely an easy win. Main risk is keeping the constants and
asymptotic normalization clean.

## 2. Subordinator / Infinite-Divisibility Reframing

Claim: If hidden stages are taken seriously for all subdivision levels, the natural object
is a causal convolution semigroup, equivalently a Levy subordinator. The
Levy-Khintchine form for subordinators has drift $b\ge0$ and a jump measure $\nu$ on
$(0,\infty)$, with no Gaussian component because monotone processes have finite variation
and cannot carry Brownian motion. The delta wall is pure drift; the Gaussian wall is
excluded structurally for the semigroup.

Why it matters: This would explain why the Gaussian wall is a wall, not merely why one
kernel misses it. It strengthens the current support argument by tying causality and
semigroup structure to the absence of a Brownian component.

Work needed: Add a candidate Section 5.5 or follow-up note introducing subordinators,
state the Levy-Khintchine representation in the needed form, identify the pure-drift
boundary, and explain how gamma kernels fit as an infinitely divisible example.

Risk/effort: Highest upside, medium to high effort. Caveat: gamma is special, and a
generic passive causal kernel need not sit inside any convolution semigroup. The current
Figure 4 interior scatter is gamma-only and should be framed that way.

## 3. Achievable-Region Completeness Lemma

Claim: The open region $\{0<N<\sigma^2\}$ may be exactly the achievable set for passive
causal kernels with finite entropy. The upper edge is approached by rescaled large-shape
gammas for any fixed variance, while the lower edge is approached by highly skewed or
near-delta kernels at fixed variance.

Why it matters: Proving this would turn the "two walls" picture from an illustration into
a proposition: the walls are tight and the interior is full.

Work needed: Construct explicit families approaching both edges at fixed variance, then
show interpolation or density in the open region. Check finite-entropy assumptions and
normalizations carefully.

Risk/effort: Medium. The boundary constructions are likely straightforward, but proving
full interior completeness may require a more careful functional-analytic or constructive
argument.

## 4. Section 5 Seam Between Theorem 1 and de Bruijn

Claim: Theorem 1 concerns entropy power of the temporal delay density. The de Bruijn
identity currently cited computes entropy production for the spatial profile of the
diffusive kernel, which is Gaussian on $\mathbb R^d$ and is not a passive causal temporal
kernel in the Section 4 sense. These are different entropies with different generators:
the heat semigroup versus a temporal subordinator semigroup.

Why it matters: The current "same irreversibility at two resolutions" language is an
analogy, not a theorem. Leaving it too strong risks conflating temporal causal kernels
with spatial heat flow.

Work needed: Either develop entropy production along the subordinator semigroup using its
own generator, or explicitly downgrade the unity claim to an analogy and separate the two
entropy notions.

Risk/effort: Conceptual fix, medium risk. Downgrading the claim is easy; proving a true
subordinator entropy-production theorem is more ambitious.
