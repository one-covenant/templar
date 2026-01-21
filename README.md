<div align="center">

# τemplar: Incentivized Internet-Wide AI Training

</div>

<div align="center">
<pre>
__|__ _  _ _  _ | _  _  
  †  (/_| | ||_)|(_||   
  |          |          
</pre>
</div>

<div align="center">
Documentation: <a href="https://github.com/one-covenant/templar/blob/main/docs/miner.md">Miner</a> • <a href="https://github.com/one-covenant/templar/blob/main/docs/validator.md">Validator</a>
</div>
<p align="center">
  <a href="https://github.com/one-covenant/templar/actions/workflows/ci.yml"><img src="https://github.com/one-covenant/templar/actions/workflows/ci.yml/badge.svg" alt="CI" style="border: none;" /></a><a href="https://codecov.io/gh/one-covenant/templar"><img src="https://codecov.io/gh/one-covenant/templar/branch/main/graph/badge.svg" alt="Codecov" style="border: none;" /></a><a href="https://github.com/one-covenant/templar/blob/main/LICENSE"><img src="https://img.shields.io/github/license/one-covenant/templar" alt="License" style="border: none;" /></a><a href="https://github.com/one-covenant/templar/commits/main"><img src="https://img.shields.io/github/last-commit/one-covenant/templar" alt="Last Commit" style="border: none;" /></a><a href="https://deepwiki.com/one-covenant/templar"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" style="border: none;" /></a><a href="https://wandb.ai/tplr/templar"><img src="https://img.shields.io/badge/W%26B-Public%20Dashboard-FFBE00?logo=weightsandbiases" alt="Weights & Biases" style="border: none;" /></a><a href="https://grafana.tplr.ai/d/ceia6bwlwn8qof/eval-metrics"><img src="https://img.shields.io/badge/Grafana-Eval%20Metrics-F46800?logo=grafana&logoColor=white" alt="Grafana Dashboard" style="border: none;" /></a>
</p>

> **⚠️ 100% Burn Period Active (January 2026)**
>
> **Templar is currently in a 100% burn period. Miners are not being evaluated and no emissions are being distributed.**
>
> The Covenant72B training run is complete - the largest model trained on decentralized infrastructure (72B parameters).
>
> We are now finalizing **Templar: Crusades**, a competition system where you submit training code, we evaluate it on target hardware, and the fastest implementation earns emissions. Full details coming soon.

## Introduction

τemplar is a decentralized training framework that enables large-scale model training across heterogeneous compute resources over the internet. By connecting diverse computational nodes through a carefully designed incentive mechanism, τemplar makes it possible to train large models collaboratively while ensuring honest participation and quality contributions.

### Key Features

- **Decentralized Training**: Leverage computational resources across the internet
- **Incentive-Driven**: Reward system that encourages quality contributions
- **Heterogeneous Compute**: Support for diverse hardware configurations
- **Scalable Architecture**: Designed for training large models across wide networks
- **Fair Participation**: Built-in mechanisms to prevent gaming and ensure honest participation

## System Overview

τemplar involves two main participants:

- **Miners**: Nodes responsible for training the model on assigned data subsets and sharing their gradients with peers.
- **Validators**: Nodes that evaluate the miners' contributions by assessing the effectiveness of their submitted gradients and updating weights on the blockchain accordingly.

The coordination between miners and validators ensures that only beneficial updates are integrated into the model. The system operates in synchronized windows, where miners and validators perform their tasks in coordination, guided by blockchain blocks.

## Incentive Design

### Overview

This section provides a detailed explanation of the incentive mechanism employed in **τemplar**, focusing on how miners and validators interact to collaboratively train a model while ensuring honest participation. **τemplar** is designed to encourage miners to contribute genuine model updates that improve the overall model performance, with validators evaluating these contributions to assign rewards accordingly.

## Miners

### Operations

1. **Model Synchronization**:
   - Miners start by synchronizing their model with the latest global state.
   - They attempt to load the latest model checkpoint from the validator with the highest stake.
   - If no checkpoint is available, they initialize their model from scratch.

2. **Data Acquisition**:
   - Each miner retrieves a specific subset of the dataset (pages) for the current window.
   - The data assignment is deterministic, based on a seed derived from the miner's UID and the window number.
   - This ensures that miners have unique but consistent data assignments per window.

3. **Local Training and Gradient Computation**:
   - Miners train their local model on the assigned data, performing forward and backward passes to compute gradients.
   - They accumulate gradients over multiple batches within the window.

4. **Momentum Update and Compression**:
   - Apply **momentum decay** to the previous momentum buffer:
   
   $m_i^{t+1} = \gamma m_i^t + \eta g_i^t$
   
   - $m_i^t$: Momentum buffer for miner $i$ at step $t$
   - $\gamma$: Momentum decay factor
   - $\eta$: Learning rate
   - $g_i^t$: Gradient computed by miner $i$ at step $t$

   - Apply **weight decay** to the model parameters:
   
   $\theta^{t+1} = (1 - \lambda)\theta^t$
   
   - $\lambda$: Weight decay coefficient

5. **Gradient Transformation and Compression**:
   - Transform the momentum-updated gradients using the Discrete Cosine Transform (DCT) for efficient compression.
   - Perform **top-k compression** by selecting the most significant coefficients, reducing communication overhead.

6. **Gradient Sharing**:
   - Miners share the compressed gradients by uploading them to a shared storage accessible by other miners and validators.
   - This enables peers to gather and aggregate gradients for collaborative training.

7. **Gradient Gathering and Aggregation**:
   - Miners gather compressed gradients from their peers.
   - Decompress and reconstruct the aggregated gradients.
   - Update their local model parameters using the aggregated gradients.

8. **Optimizer Step and Learning Rate Scheduling**:
   - Apply optimizer steps (e.g., SGD) to update the model.
   - Adjust the learning rate using schedulers that combine warm-up and cosine annealing.

9. **Window Progression**:
   - Proceed to the next window and repeat the process, ensuring continuous contribution to model training.

---

### Formal Definitions

- **Model Parameters** at time $t$: $\theta^t$
- **Local Gradients**: $g_i^t$, computed by miner $i$ at time $t$
- **Momentum Buffer Update**:

  $m_i^{t+1} = \gamma m_i^t + \eta g_i^t$

- **Weight Decay**:

  $\theta^{t+1} = (1 - \lambda)\theta^t$

- **Compressed Gradient**: $\tilde{g}_i^t$, the top-k compressed version of the transformed momentum buffer
- **Aggregated Gradient**:

  $\delta_{\text{agg}} = \sum_{i \in \mathcal{P}} \tilde{g}_i^t$
  
  - $\mathcal{P}$: Set of peer miners

---

## Validators

### Operations

1. **Model Synchronization**:
   - Synchronize their model with the latest global state.
   - Attempt to load the latest model checkpoint from the validator with the highest stake or start from scratch.

2. **Data Acquisition**:
   - Select a miner to evaluate.
   - Retrieve the same data subset assigned to that miner using the same deterministic seeding mechanism.

3. **Gradient Gathering**:
   - Gather the compressed gradients submitted by miners for the current window.
   - Decompress and apply these gradients to their local model to maintain consistency.

4. **Evaluation of Miners**:
   - For the selected miner $i$:
     - **Compute Loss Before** applying the miner's gradient:
       
       $L_{\text{before}} = \mathcal{L}(\theta^t; D_i)$
       
       - $D_i$: Dataset assigned to miner $i$
     - **Apply** the miner's gradient:
       
       $\theta^{t+1} = \theta^t + \delta_i$
       
       - $\delta_i$: Decompressed gradient from miner $i$
     - **Compute Loss After** applying the gradient:
       
       $L_{\text{after}} = \mathcal{L}(\theta^{t+1}; D_i)$
       
     - **Compute Improvement**:
       
       $s_i = L_{\text{before}} - L_{\text{after}}$

5. **Score Calculation**:
   - The score $s_i$ reflects the miner's contribution to reducing the loss.

6. **Weight Assignment and Update**:
   - Update the moving average of the miner's score:
   
     $\bar{s}_i = \alpha s_i + (1 - \alpha)\bar{s}_i$
     
   - Compute weights as the moving average of improvement scores:
   
     $w_i = \bar{s}_i$

7. **Blockchain Update**:
   - Validators set these weights on the blockchain, influencing reward distribution and miner reputation.

8. **Optimizer Step and Learning Rate Scheduling**:
   - Apply optimizer steps and adjust learning rates to keep the model updated.

---

### Formal Definitions

- **Model Loss**:
  - Before update: $L_{\text{before}} = \mathcal{L}(\theta^t; D_i)$
  - After update: $L_{\text{after}} = \mathcal{L}(\theta^{t+1}; D_i)$
- **Miner's Score**:
  
  $s_i = L_{\text{before}} - L_{\text{after}}$
  
- **Moving Average Score**:
  
  $\bar{s}_i = \alpha s_i + (1 - \alpha)\bar{s}_i$
  
- **Assigned Weight**:
  
  $w_i = \bar{s}_i$

---

## Incentive Mechanism

### Objective

The incentive mechanism in **τemplar** aims to:

- **Encourage Honest Participation**: Motivate miners to perform genuine training and provide updates that improve model performance.
- **Promote Model Improvement**: Reward updates that lead to a reduction in loss.
- **Discourage Malicious Behavior**: Penalize updates that do not improve or degrade model performance.

### Detailed Explanation

#### Score Calculation and Weight Assignment

1. **Compute Loss Improvement**:
   - Validators measure the effectiveness of a miner's update by the reduction in loss on the assigned dataset.
   - The score $s_i = L_{\text{before}} - L_{\text{after}}$ quantifies this improvement.

2. **Interpretation of Scores**:
   - **Positive $s_i$**: Indicates the miner's update improved the model.
   - **Zero $s_i$**: No change in model performance.
   - **Negative $s_i$**: The miner's update worsened the model.

3. **Moving Average for Stability**:
   - Using a moving average $\bar{s}_i$ smooths out fluctuations in individual scores.
   - Helps in maintaining stable weights over time.

4. **Weight Computation**:
   - Compute weights as the moving average of improvement scores:
   
     $w_i = \bar{s}_i$

     - The moving average:
       - Smooths out fluctuations in individual scores
       - Provides stability in weight assignment over time
       - Directly reflects sustained contribution quality

5. **Blockchain Update**:
   - Validators set these weights on the blockchain, which influences reward distribution and miner reputation.

### Formal Guarantees

1. **Alignment Incentive**:
   - Miners are incentivized to produce updates that reduce the loss, aligning individual goals with the collective objective.

2. **Discouraging Malicious Actions**:
   - Miners submitting harmful updates receive lower or negative scores, resulting in minimal or no rewards.

3. **Fair Reward Distribution**:
   - Weights are computed based on the actual performance improvements contributed by miners.

4. **Convergence Assurance**:
   - By aggregating beneficial updates, the model is guided towards convergence and improved performance.

5. **Sybil Resistance**:
   - Since rewards are based on contribution quality, creating fake identities without meaningful contributions offers no advantage.

---

## Formal Analysis

### Miner Utility Maximization

Miners aim to maximize their expected reward, which is proportional to their assigned weight $w_i$:

$$\max_{\delta_i} \quad w_i = \frac{e^{\bar{s}i}}{\sum{j \in \mathcal{M}} e^{\bar{s}_j}}$$


Subject to:

- **Update Rule**:
  
  $\delta_i = \text{Compress}(\gamma m_i^{t} + \eta g_i^t)$

- **Model Update**:
  
  $\theta^{t+1} = \theta^{t} + \delta_{\text{agg}}$

- **Score Function**:
  
  $s_i = L(\theta^{t}; D_i) - L(\theta^{t} + \delta_i; D_i)$

The optimal strategy for miners is to compute accurate gradients that lead to a reduction in loss on their assigned data.

### Validator Consistency

Validators ensure:

- **Fair Evaluation**:
  - Use the same datasets as miners to compute losses.
  - Apply the miners' updates accurately.

- **Transparency**:
  - Evaluation procedures are deterministic and replicable.

### Security Considerations

1. **Data Integrity**:
   - Deterministic data assignments prevent miners from manipulating their datasets.

2. **Gradient Compression and Privacy**:
   - Compression reduces the risk of exposing sensitive information.
   - Only significant components are shared.

3. **Preventing Free Riding**:
   - Miners gain rewards only if their updates lead to performance improvements.

---

## Conclusion

The incentive mechanism in **τemplar** effectively encourages miners to contribute meaningful updates that enhance the global model. By tying rewards to the actual improvement in model loss resulting from miners' gradients, the system ensures that only beneficial contributions are rewarded. This approach aligns individual miner incentives with the collective goal of training an effective model.

The careful design of data assignment, gradient evaluation, and weight distribution fosters a self-regulating ecosystem where honest participation is the most profitable strategy. Formal guarantees provide robustness against malicious actors, promoting the overall improvement of the model through collaborative effort.

Thus, **τemplar** creates a sustainable and efficient framework for decentralized collaborative learning, leveraging incentives to drive positive contributions and advance the shared model's performance.

