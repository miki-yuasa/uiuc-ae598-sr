#let title = [AE598 SR Homework 3]

#set page(
  paper: "us-letter",
  header: align(
    right + horizon,
    title
  ),
  numbering: "1",
)
#set par(justify: true)
#set text(
  font: "Libertinus Serif",
  size: 11pt,
)

#align(center, text(17pt)[
  *#title*
])
#align(center)[
    Mikihisa Yuasa\
    #link("mailto:myuasa2@illinois.edu")
  ]

#let mathbf(input) = {$upright(bold(#input))$}

= 1 Introduction to Risk-Aware Motion Planning

In theory, the mean, worst-case, and CVaR optimal trajectories have the following characteristics.
The mean risk trajectory follows a relatively direct path toward the goal, cutting closer to the obstacles. 
Since it optimizes for the expected value under Gaussian uncertainty, it tolerates moderate proximity to obstacles as long as collisions are unlikely on average.
The worst-case trajectory, on the other hand, bends significantly around all obstacles, creating wide clearance. 
It reflects the need to avoid collisions under any disturbance within a fixed bound (±0.6). 
As such, it sacrifices path efficiency to ensure guaranteed safety under worst-case deviations.
The CVaR trajectory is more conservative than the mean risk path but not as cautious as the worst-case. It avoids obstacles with a moderate buffer, optimizing the expected loss in the worst $alpha$-percent (here, 30%) of cases. The result is a balanced trade-off between performance and risk sensitivity.

However, as shown in @trajectories, the mean and CVaR trajectories are not significantly different from each other, while the worst-case trajectory is notably distinct.
This is because the mean and CVaR trajectories are both designed to minimize expected loss with the same degree of perturbation (i.e., $sigma=0.2$), while the worst-case trajectory is focused on ensuring safety under the most extreme conditions given the fixed bound of ±0.6.

#figure(
  image("data/fig_optimal_trajectories.png"),
  caption: ["Mean, worst-case, and CVaR optimal trajectories."],
)<trajectories>

= 2 Brief Review of Adaptive Robot Control
First, define a Lyapunov candidate:
$ V = 1/2 mathbf(s)_r^top mathbf(M) mathbf(s)_r + 1/2 tilde(mathbf(F))^((c)top) mathbf(Gamma)_a tilde(mathbf(F))^((c)), $
where $tilde(mathbf(F))^((c)):= mathbf(F)^((c)) - hat(mathbf(F))^((c))$.

Now, calculate the time derivative of $V$.
By using the chain rule, we have
$ dot(V) = mathbf(s)_r^top mathbf(M) dot(mathbf(s))_r +  1/2 mathbf(s)_r^top dot(mathbf(M)) mathbf(s)_r + tilde(mathbf(F))^((c)top) mathbf(Gamma)_a dot(tilde(mathbf(F)))^((c)). $
Now, plug in $dot(tilde(mathbf(F)))^((c)) = mathbf(Gamma)_a^(-1) mathbf(J)^((c)) mathbf(s)_r$, so $tilde(mathbf(F))^((c)top) mathbf(Gamma)_a dot(tilde(mathbf(F)))^((c)) = -tilde(mathbf(F))^((c)top) mathbf(J)^((c)) mathbf(s)_r$.

Form the system and control low, the closed-loop dynamics of $mathbf(s)_r$ can be shown to be
$ mathbf(M)dot(mathbf(s))_r + mathbf(C) mathbf(s)_r + mathbf(K)mathbf(s)_r = mathbf(J)^((c)top)tilde(mathbf(F))^((c)). $
Then, 
$ mathbf(s)_r^top mathbf(M) dot(mathbf(s))_r = -mathbf(s)_r^top mathbf(C) mathbf(s)_r - mathbf(s)_r^top mathbf(K)mathbf(s)_r + mathbf(s)_r^top mathbf(J)^((c)top)tilde(mathbf(F))^((c)). $
Hence,
$ dot(V) = -mathbf(s)_r^top mathbf(C) mathbf(s)_r - mathbf(s)_r^top mathbf(K)mathbf(s)_r + mathbf(s)_r^top mathbf(J)^((c)top)tilde(mathbf(F))^((c)) + 1/2 mathbf(s)_r^top dot(mathbf(M)) mathbf(s)_r - tilde(mathbf(F))^((c)top) mathbf(J)^((c)). $
Since $mathbf(s)_r^top mathbf(J)^((c)top)tilde(mathbf(F))^((c)) = tilde(mathbf(F))^((c)top) mathbf(J)^((c))$, we group the terms:
$ dot(V) = - mathbf(s)_r^top mathbf(K)mathbf(s)_r + 1/2 mathbf(s)_r^top (dot(mathbf(M))-2mathbf(C)) mathbf(s)_r. $
Since $dot(mathbf(M))-2mathbf(C)$ is skew-symmetric, 
$ dot(V) =  - mathbf(s)_r^top mathbf(K)mathbf(s)_r <=0. $

Now, we apply Barbalat's lemma.
$V(t)$ is bounded and non-increasing, so $V(t)$ converges to a limit.
$dot(V)(t) = - mathbf(s)_r^top mathbf(K)mathbf(s)_r$ implies that $mathbf(s)_r in L^2$. Assume all the signal are  abounded, then $mathbf(s)_r$ is bounded.
Then, $ lim_(t -> infinity) ||mathbf(s)_r|| = 0. $