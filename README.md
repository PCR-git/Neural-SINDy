This repo contains a Neural SINDy architecture for identification of nonlinear dynamics, which I wrote in early 2023.

While I initially developed the idea of implementing SINDy as a neural network independently, I later discovered similar concepts had already appeared in the literature.
As a result, I did not publish this work.

This repo contains a single ipynb file, for easier experimentation, which includes an implementation of the algorithm, and experiments with a simple 2D ODE,
the Lorenz system (a chaotic 3D ODE), and a system with sinusoidal forcing.

Among the improvements to the SINDy algorithm are:
- A neural-network architecture (can automatically find constants and learn more complicated dynamics).
  - Parameter sharing allows us to represent an exponential number of candidate functions with a polynomial number of parameters.
  - Makes the optimization easier by giving more paths down to equilibrium.
  - Makes high-dimensional systems and systems with highly-compositional dynamics tractable.
  - Makes use of neural network libraries and GPU computation.
  - SINDy-PI requires guessing and checking LHS terms, which hugely increases run-time, whereas we can represent fractional dynamics directly.
  - Adam appears to work well in practice, which is faster than second-order optimization.
- Use of Neural ODE.
  - Takes advantage of higher-order information by backpropagating through time.
  - Does so in a scalable manner by making use of the adjoint sensitivity method, and allows us to make use of powerful ODE solvers.
- Improved optimization:
  - L1 is not a good criterion. Instead, remove the worst performer.
  - Using validation data to prune improves robustness.
  - Dummy variables allow us to apply L1 regularization.
  - Using batch-norm improves optimization by alleviating scaling/dimensionality issues, while still allowing us to recover weights.
  - Removing one variable at a time removes an unnecessary hyperparameter and provides a natural way of combining search with gradient-based optimization.
- Neural network interpretability through libraries of known functions and pruning.
