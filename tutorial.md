# Stabilizing High-Order Root-Finding in Ill-Conditioned Regimes  
### A gentle, intuitive walkthrough for a strong undergrad

> Goal: understand every important idea behind the paper  
> “Stabilizing High-Order Root-Finding in Ill-Conditioned Regimes:  
> A Hybrid Halley–Bisection Approach for Black–Scholes Inversion”

---

## 0. Big Picture: What is this paper actually doing?

We’re trying to solve this real-world problem:

> **Given the market price of an option, how “volatile” does the market think the underlying stock is?**

- The **Black–Scholes formula** takes in:
  - stock price \( S \)
  - strike price \( K \)
  - time to maturity \( T \)
  - interest rate \( r \)
  - volatility \( \sigma \)

  and **outputs** a theoretical option price.

- The market gives us **the price**, and we want to **recover \( \sigma \)**.  
  That’s like saying:
  > I know the function’s output. What input produced it?

Mathematically:

\[
\text{Find } \sigma^* \text{ such that } C_{BS}(\sigma^*) = C_{\text{mkt}}
\]

We don’t have a neat algebraic formula for \( \sigma^* \), so we solve it numerically.

The paper is about:

- Building a **fast**, **accurate**, and **robust** algorithm to find that \( \sigma^* \), especially in **hard cases** (deep out-of-the-money options, where the function is “flat” and normal methods struggle).

The tools and concepts:

- Root-finding (Newton’s method, Halley’s method, Bisection)
- Derivatives (first and second)
- Conditioning (how sensitive outputs are to small input changes)
- Guarded / hybrid algorithms (fallback logic, bracketing, etc.)

You already have the math background needed. We’ll just connect the dots.

---

## 1. The Black–Scholes Setup

### 1.1 What is a call option?

Very short version:

- A **European call option** gives you the right (but not the obligation) to buy a stock at price \( K \) at time \( T \).
- If at maturity:
  - stock price \( S_T > K \), you exercise: profit \( \approx S_T - K \)
  - stock price \( S_T \le K \), you don’t exercise: profit \( = 0 \)

The option itself has a **price today**, denoted \( C_{\text{mkt}} \) (market price).

### 1.2 The Black–Scholes formula

The Black–Scholes formula for a European call:

\[
C_{BS}(\sigma) = S\, N(d_1) - K e^{-rT} N(d_2)
\]

where:

- \( N(\cdot) \) is the **standard normal CDF**
- \( n(\cdot) \) will be the PDF later
- \( d_1, d_2 \) are defined as:

\[
d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad
d_2 = d_1 - \sigma\sqrt{T}.
\]

Here:

- \( S \): current stock price
- \( K \): strike price
- \( r \): risk-free interest rate
- \( T \): time to maturity (years)
- \( \sigma \): volatility (what we want to recover!)

### 1.3 The inverse problem: implied volatility

The market price is \( C_{\text{mkt}} \). We want \( \sigma \) such that:

\[
C_{BS}(\sigma) = C_{\text{mkt}}.
\]

Rewriting:

\[
f(\sigma) = C_{BS}(\sigma) - C_{\text{mkt}} = 0.
\]

So our problem is:

> **Find a root of \( f(\sigma) \).**

This is a pure **numerical root-finding** problem.

---

## 2. Root-Finding: Newton, Bisection, Halley

### 2.1 What is root-finding?

Given a function \( f(x) \), a **root** is a value \( x^* \) such that:

\[
f(x^*) = 0.
\]

We often can’t solve \( f(x) = 0 \) analytically, so we build an **iterative algorithm**:

1. Start with an initial guess \( x_0 \).
2. Produce a sequence \( x_1, x_2, x_3, \dots \)
3. Hope that \( x_n \to x^* \).

---

### 2.2 The Bisection Method (the safety net)

Bisection is the “slow but safe” method.

Idea (1D only, like our case in σ):

1. Start with a bracket \([a, b]\) such that:
   \[
   f(a) \cdot f(b) < 0
   \]
   That means the function changes sign between \( a \) and \( b \), so there is at least one root in there (by continuity).

2. Compute the midpoint:
   \[
   m = \frac{a + b}{2}
   \]

3. Check which side the root is on:
   - If \( f(a) \cdot f(m) < 0 \), then root is in \([a, m]\). Set \( b = m \).
   - Else root is in \([m, b]\). Set \( a = m \).

4. Repeat.

Features:

- **Always converges** (as long as the function is continuous and you maintain sign change).
- Error shrinks by factor \(1/2\) each iteration → **linear convergence**.
- But it can be **slow**: to get ~15 digits of accuracy you need ~50 iterations.

---

### 2.3 Newton’s Method (fast but can explode)

Newton’s method uses calculus (the derivative).

Idea: locally approximate \( f(x) \) by its tangent line at your current guess, then jump to where that line hits zero.

At \( x_n \):

- Tangent line equation:
  \[
  f(x) \approx f(x_n) + f'(x_n)(x - x_n)
  \]
- Set this to zero and solve for \( x \):
  \[
  0 = f(x_n) + f'(x_n)(x_{n+1} - x_n)
  \Rightarrow
  x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}.
  \]

So Newton iteration is:

\[
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}.
\]

Properties:

- **Quadratic convergence** near the root:  
  error shrinks approximately like \( e_{n+1} \approx C e_n^2 \).
- Great when:
  - you’re close to the root,
  - and the derivative \( f'(x_n) \) is not too small.
- **Terrible** when:
  - \( f'(x_n) \) is tiny → huge step size → overshoot, divergence.
  - or the function is weird/non-smooth.

In our IV problem, **tiny derivative** is exactly the issue in deep OTM regions.

---

### 2.4 Halley’s Method (using second derivatives)

Halley’s method is like Newton on steroids: it uses the second derivative to get a better local approximation of \( f(x) \).

Without doing full derivation, the Halley update for 1D is:

\[
x_{n+1} = x_n - \frac{2 f(x_n) f'(x_n)}{2 [f'(x_n)]^2 - f(x_n) f''(x_n)}.
\]

Compare to Newton:

\[
x_{n+1}^{\text{Newton}} = x_n - \frac{f(x_n)}{f'(x_n)}.
\]

Here Halley tries to be smarter about **curvature**:

- If the function is bending, using the second derivative \( f'' \) helps predict where the zero is more accurately.

Key point:

- **Cubic convergence** near the root:  
  \( e_{n+1} \approx C e_n^3 \).  
  This is dramatically faster once you’re “in the zone.”

Downside:

- More complicated.
- Needs second derivative.
- Can be more fragile if not guarded (denominators can blow up, weird behavior far from the root).

---

## 3. Conditioning, Vega, and Why Things Go Wrong

### 3.1 What is “conditioning”?

Informal idea:

- A problem is **well-conditioned** if small changes in input lead to small changes in output.
- It’s **ill-conditioned** if tiny input changes cause big output changes.

In our setting:

- Input: volatility \( \sigma \)
- Output: option price \( C_{BS}(\sigma) \)

When we invert the relationship, small changes in price can cause big changes in the implied volatility if the function is “flat” in that region.

### 3.2 Vega: the sensitivity to volatility

In options, the derivative of the price with respect to volatility is called **Vega**:

\[
\mathcal{V}(\sigma) = \frac{\partial C}{\partial \sigma}.
\]

For the Black–Scholes call:

\[
\mathcal{V}(\sigma) = S \sqrt{T} \, n(d_1),
\]

where \( n(\cdot) \) is the standard normal **PDF**:
\[
n(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}.
\]

Properties:

- \( n(x) \) is largest at \( x = 0 \), and decays quickly as \(|x|\) increases.
- When the option is **at the money (ATM)** (\( S \approx K \)), we often have \( d_1 \) near 0 → **Vega is high**.
- In **deep OTM** or **deep ITM**:
  - \( |d_1| \) is large → \( n(d_1) \) is tiny → **Vega is small**.

Interpretation:

- When Vega is **large**: price responds strongly to changes in volatility → good for inversion.
- When Vega is **tiny**: price barely changes as you move σ → the function is **flat**.

Root-finding consequence:

- Newton’s iteration:
  \[
  \sigma_{n+1} = \sigma_n - \frac{f(\sigma_n)}{f'(\sigma_n)}
  \]
  If \( f'(\sigma_n) \) (Vega) is **very small**, then:
  - denominator is tiny,
  - step size \( \approx \frac{1}{f'(\sigma_n)} \) becomes huge,
  - the algorithm can **jump wildly** and diverge.

This is exactly the ill-conditioning issue.

---

### 3.3 Vomma: second derivative with respect to volatility

Vomma is the second derivative of option price with respect to volatility:

\[
\text{Vomma}(\sigma) = \frac{\partial^2 C}{\partial \sigma^2}.
\]

In Black–Scholes:

\[
\text{Vomma}(\sigma) = \frac{\mathcal{V} \, d_1 d_2}{\sigma}.
\]

Interpretation:

- Vomma measures **how Vega itself changes with volatility**.
- In terms of shape: it tells you about the **curvature** of \( C(\sigma) \).

Halley’s method uses Vomma (through \( f''(\sigma) \)) to get a more accurate update step, especially when the function is flat but curved.

---

## 4. Basins of Attraction & Why Initial Guess Matters

### 4.1 Iteration as a function

Think of an iteration like:

\[
\sigma_{n+1} = \Phi(\sigma_n),
\]

where \( \Phi \) is some update rule (Newton, Halley, etc.).

We can think of this as a **dynamical system**:  
start from some point \( \sigma_0 \), repeatedly apply \( \Phi \), see where you end up.

- If for a given \( \sigma_0 \), the sequence converges to a root \( \sigma^* \), that initial value is said to be in the **basin of attraction** of \( \sigma^* \).

### 4.2 Basins of attraction

- Baysin of attraction = the set of all starting points that eventually converge to a particular root under the iteration.
- For nonlinear methods (especially higher-order ones), these basins can have **fractal boundaries** in more complex cases.
- Moral: if your starting point is outside the basin, you’re probably screwed.

In our case:

- Deep OTM options → function is super flat → the basin of attraction shrinks.
- Picking a fixed initial guess like σ₀ = 0.5 for all cases might be:
  - fine for ATM,
  - **horrible** in the tails.

Hence: **good initialization is crucial.**

---

## 5. Corrado–Miller Approximation: a Good Initial Guess near ATM

The Corrado–Miller approximation is a **closed-form approximation** for implied volatility when the option is near the money.

We don’t need to re-derive it; conceptually:

- They approximate the Black–Scholes price with a simplified expression that can be inverted analytically (like turning it into a quadratic in σ).
- From that, they get an approximation for σ.

Paper’s notation:

- Let:
  \[
  X = K e^{-rT}
  \]
- Define:
  \[
  L = C_{\text{mkt}} - \frac{S - X}{2},
  \qquad
  D = S - X.
  \]
- Then the Corrado–Miller approximation is:

\[
\sigma_{CM} =
\frac{\sqrt{2\pi}}{S + X} \cdot \frac{L + \sqrt{L^2 - D^2/\pi}}{\sqrt{T}}.
\]

This gives a decent **starting** σ when the option is not too extreme.

### 5.1 The discriminant problem

Notice the square root:

\[
\sqrt{L^2 - D^2/\pi}.
\]

Call the inside:

\[
\Delta = L^2 - \frac{D^2}{\pi}.
\]

- For nice, near-ATM cases, \(\Delta > 0\) and life is good.
- For **deep OTM/ITM**, the assumptions of the approximation fail → sometimes \(\Delta < 0\).
  - That means we’re trying to take square root of a **negative number** → no real solution.
  - In code, this gives NaN (not a number) or a crash.

In practice, many naive implementations miss this and blow up.  
The paper calls this the **Discriminant Failure**.

---

## 6. Asymptotic Behavior in the Tails: Lee’s Insight

In extreme cases (very large or very small strike \( K \)), the implied volatility behaves according to asymptotic formulas.

One such approximation is:

\[
\sigma_{\text{tail}} \approx \sqrt{\frac{2 |\ln(S/K)|}{T}}.
\]

You don’t need the full derivation; intuitively:

- As \( K \to \infty \) (for calls) or \( K \to 0 \) (for puts), option prices are governed by **rare events** (big deviations of the stock price).
- Large deviations and moment generating functions lead to formulas like this for the **wing behavior** of implied volatility.

Key insight for us:

- This asymptotic formula is **bad near the money**,  
  but **good in the extreme tails**.
- So we can use it as an intelligent **fallback initial guess** when Corrado–Miller fails.

---

## 7. Putting It Together: The Guarded Initialization

Goal: choose a starting volatility \( \sigma_0 \) that:

- is in a reasonable range,
- doesn’t produce NaNs,
- and is tailored to whether the option is:
  - in the “body” (near ATM),
  - or in the “tail” (deep ITM/OTM).

### 7.1 Algorithm logic:

Given option parameters \( S, K, T, r \) and market price \( C \):

1. Compute:
   - \( X = K e^{-rT} \)
   - \( D = S - X \)
   - \( L = C - D/2 \)
   - \( \Delta = L^2 - D^2/\pi \)

2. If \( \Delta < 0 \):  
   → this is likely a **tail** scenario. Use the **asymptotic formula**:
   \[
   \sigma_0 = \sqrt{\frac{2 |\ln(S/K)|}{T}}.
   \]

3. Else (body):
   \[
   \sigma_0 = \frac{\sqrt{2\pi}}{S + X} \cdot \frac{L + \sqrt{\Delta}}{\sqrt{T}}.
   \]

4. Clamp it to a safe range:
   \[
   \sigma_0 \in [\sigma_{\min}, \sigma_{\max}],
   \]
   e.g. \([10^{-5}, 5]\).

This yields the “G” and “CM” parts of **GCM-H** (Guarded Corrado–Miller Halley).

---

## 8. Guarded Halley Iteration: Staying Safe While Being Fast

We now have:

- A smarter initial guess \( \sigma_0 \).
- We want to use Halley’s method for fast convergence.
- But we cleverly wrap it in safety checks.

### 8.1 Maintaining a bracket

We maintain a bracket \([ \sigma_{\min}, \sigma_{\max} ]\) such that:

\[
f(\sigma_{\min}) \cdot f(\sigma_{\max}) < 0.
\]

This means:

- We **know** there’s a root in that interval.
- If Halley suggests something outside this range, we don’t trust it.

### 8.2 Iteration step

At each iteration:

1. Compute:
   - model price \( C_{BS}(\sigma) \)
   - error \( \text{error} = C_{BS}(\sigma) - C_{\text{mkt}} = f(\sigma) \)
   - Vega \( \mathcal{V} = f'(\sigma) \)
   - Vomma \( \text{Vomma} = f''(\sigma) \)

2. If \(|\text{error}| < \text{tolerance}\):  
   → we’re done.

3. Update bracket:
   - If error > 0 (price too high), root is at lower vol → set \( \sigma_{\max} = \sigma \)
   - Else set \( \sigma_{\min} = \sigma \)

4. **Vega guard**:
   - If \( \mathcal{V} \) is too small (e.g. \( < 10^{-12} \)), then Halley/Newton is unreliable.  
   - In that case, do a **bisection step**:
     \[
     \sigma \leftarrow \frac{\sigma_{\min} + \sigma_{\max}}{2}.
     \]
   - This is slow but **safe**.

5. Otherwise, compute Halley update:
   \[
   \sigma_{\text{new}} = \sigma - \frac{2 \cdot \text{error} \cdot \mathcal{V}}{2 \mathcal{V}^2 - \text{error} \cdot \text{Vomma}}.
   \]

6. **Bracket guard**:
   - If \( \sigma_{\text{new}} \) is **outside** the bracket, reject it and again do bisection:
     \[
     \sigma \leftarrow \frac{\sigma_{\min} + \sigma_{\max}}{2}.
     \]
   - Else accept it:
     \[
     \sigma \leftarrow \sigma_{\text{new}}.
     \]

This hybrid strategy gives:

- **Global stability** from the bracket and bisection.
- **Local cubic convergence** from Halley when it’s safe to use.

---

## 9. Convergence Rates: Linear vs Quadratic vs Cubic

Let:

- \( e_n = |\sigma_n - \sigma^*| \) be the error at iteration \( n \).

We say a method has:

- **Linear convergence** if:
  \[
  e_{n+1} \approx C e_n
  \]
  with \( 0 < C < 1 \). (Bisection.)

- **Quadratic convergence** if:
  \[
  e_{n+1} \approx C e_n^2
  \]
  (Newton.)

- **Cubic convergence** if:
  \[
  e_{n+1} \approx C e_n^3
  \]
  (Halley.)

Rough sense:

- Linear: add ~1 bit of accuracy per step.
- Quadratic: number of correct digits ~doubles every step (once close).
- Cubic: number of correct digits triples every step (once close).

The GCM-H solver:

- Uses Halley to exploit cubic convergence near the root.
- Uses bisection safeguards and brackets to keep things stable globally.

---

## 10. Floating-Point Subtleties

Computers use **finite precision** (double precision / 64-bit floats).

- Machine epsilon \( \epsilon_{\text{mach}} \approx 2.22 \times 10^{-16} \)
- Numbers smaller than this relative to 1 are essentially “invisible” to the machine.

Practical issue:

- If you divide by a number smaller than, say, \( 10^{-12} \), the result can be huge or numerically garbage.
- In deep tails, Vega can be around \( 10^{-12} \) or smaller.

Hence:

```cpp
if (vega < EPSILON_VEGA) {
    // too flat, derivative-based update is nonsense → use bisection
    sigma = (vol_min + vol_max) * 0.5;
    continue;
}
