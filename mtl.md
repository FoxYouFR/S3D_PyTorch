# Multi-task learning

## Task balancing approaches

- Compute final loss as weighted loss
$$\begin{equation*} \mathcal {L}_{MTL} = \sum _{i} w_i \cdot \mathcal {L}_i \end{equation*}$$

- Uncertainty weighting
$$\begin{equation*} \mathcal {L}\left(W,\sigma _1,\sigma _2\right) = \frac{1}{2\sigma ^2_1} \mathcal {L}_1 \left(W\right) + \frac{1}{2\sigma ^2_2} \mathcal {L}_2 \left(W\right) + \log \sigma _1 \sigma _2 \end{equation*} $$

By minimising noise parameters $\sigma_1$, $\sigma_2$, one can balance task-specific loses during training. The noise parameters are updated through standard backprop.

- GradNorm
Stimulate the task-specific gradients to be of similar magnitude by balancing the task-specific gradient compared to the mean gradient and the pace at which different tasks are learned. It updates the weights of the losses using backprop.

- Dynamic Weight Averaging (DWA)
$$\begin{equation*} w_i\left(t\right) = \frac{N \exp \left(r_i \left(t-1 \right)/T \right)}{\sum _n \exp \left(r_n \left(t-1\right)/T\right)}, r_n\left(t-1\right) = \frac{L_n\left(t-1\right)}{L_n\left(t-2\right)} \end{equation*} $$

Only balances pace at which tasks are learned, not the gradient magnitudes.

- Dynamic Task Prioritization (DTP)
$$\begin{equation*} w_i\left(t\right) = -\left(1 - \kappa _i \left(t\right)\right)^{\gamma _i} \log \kappa _i \left(t\right)\end{equation*} $$

The motivation is that the network should spend more effort to learn the 'difficult' tasks. DTP makes sense when we have access to clean ground-truth annotations. $\kappa_i$ are key performance indicators (e.g. accuracy) and $\lambda_i$ are task-elvel focusing parameters allowing to adjust the weight at which easy or hard tasks are down-weighted.

- Multiple Gradient Descent Algorithm (MGDA)
Find a Pareto stationary point to update the weights. As long as there is a common direction along which the task-specific losses can be decreased, we have not reached a Pareto optimal point yet. Since the shared network weights are only updated along common directions of the task-specific gradients, conflicting gradients are avoided in the weight update step. Only applied to small-scale datasets.

## Papers

- [Multi-Task Learning for Dense Prediction Tasks: A Survey](https://arxiv.org/pdf/2004.13379)
- [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048)
- [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/pdf/1706.05098)