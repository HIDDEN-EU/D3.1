# Mathematical Blueprint: Lévy Spatial Pyramid Fusion for HEAL
**Stochastic Multi-Agent BEV Fusion with Single-Pass Spatial Uncertainty**

This document details the mathematical reformulation of the Pyramid Fusion module in the **HEterogeneous ALliance (HEAL)** framework using **Lévy Attention**. By casting spatial collaborative fusion as stochastic integration against an inhomogeneous Poisson random measure, we equip collaborative perception with real-time, single-pass spatial uncertainty mapping without sacrificing the efficiency of HEAL's backward alignment.

---

## 1. Introduction & The Softmax Bottleneck

In collaborative perception (V2V/V2X), multiple heterogeneous agents align and transmit their Bird's Eye View (BEV) feature maps to an ego vehicle to overcome occlusions and range limitations [108, 109]. 

In HEAL’s **Pyramid Fusion** module, the feature maps $F_{j \to i}^{(\ell)}$ from collaborating agents $j$ are aggregated at multiple scales $\ell$ using spatial weights $W_{j \to i}^{(\ell)}$ derived from unnormalised foreground maps $S_{j \to i}^{(\ell)}$ [123]:
$$W^{(\ell)}_{j \to i} = \text{softmax}_j\left(S^{(\ell)}_{j \to i}\right)$$

This agent-wise `softmax` normalises the weights across agents at each spatial cell $s = (x, y)$ [123, 124]:
$$W^{(\ell)}_{j \to i}(s) = \frac{\exp\left(S^{(\ell)}_{j \to i}(s)\right)}{\sum_{k \in \mathcal{A}} \exp\left(S^{(\ell)}_{k \to i}(s)\right)}$$

While this successfully computes relative fusion weights, it suffers from a fundamental information bottleneck: **row normalisation divides out and discards the absolute scale of the foreground confidence (the evidence)** [1, 6]. 
* **Case A (Low Confidence):** Two collaborating agents both predict background at a cell $s$ with very low unnormalised scores: $S_{1 \to i}^{(\ell)}(s) = -10$ and $S_{2 \to i}^{(\ell)}(s) = -10$. The resulting fusion weights are $W_{1 \to i}^{(\ell)}(s) = 0.5$ and $W_{2 \to i}^{(\ell)}(s) = 0.5$.
* **Case B (High Confidence):** Both agents strongly detect a vehicle with high scores: $S_{1 \to i}^{(\ell)}(s) = 10$ and $S_{2 \to i}^{(\ell)}(s) = 10$. The fusion weights are still $W_{1 \to i}^{(\ell)}(s) = 0.5$ and $W_{2 \to i}^{(\ell)}(s) = 0.5$.

In both cases, the downstream network receives the same fused feature vector [6]. It has no way of knowing whether the features represent a high-confidence consensus or an unobserved, occluded background. Furthermore, if the agents' features conflict (e.g., camera and LiDAR mismatch), this **disagreement** is completely hidden from downstream planner modules [1, 9].

By reformulating this fusion layer using **Lévy Attention**, we preserve both **Evidence** (absolute foreground confidence) and **Disagreement** (feature mismatch) in closed form, outputting a calibrated BEV spatial trust map in a single forward pass [1, 17].

---

## 2. Mathematical Reformulation: Lévy Spatial Intensity Field

To translate Lévy Attention to the spatial domain, we map its continuous temporal index space $S = [0, 1]^2$ to the **2D Bird's Eye View spatial grid** [12].

### 2.1 The Spatial Index Space
Let the index space $D \subset \mathbb{R}^2$ represent the discrete BEV grid of size $H \times W$. Each grid cell (pixel) $s = (x, y) \in D$ corresponds to a physical spatial location (e.g., $0.4\text{m} \times 0.4\text{m}$ resolution) [137].

Let $\mathcal{A} = \{j_1, j_2, \dots, j_M\}$ be the set of collaborating agents in the scene [121]. At a given scale $\ell \in \{1, \dots, L\}$, each agent $j \in \mathcal{A}$ provides:
1. Aligned spatial feature maps: $F^{(\ell)}_{j \to i}(s) \in \mathbb{R}^d$ (representing values $V_j(s)$) [122, 123].
2. Unnormalised foreground maps: $S^{(\ell)}_{j \to i}(s) \in \mathbb{R}$ (representing unnormalised logits) [123, 124].

### 2.2 Defining the Inhomogeneous Spatial Intensity
Instead of normalizing $S^{(\ell)}_{j \to i}(s)$ across agents with a softmax immediately, we define an **agent-specific unnormalised spatial foreground intensity** $\lambda^{(\ell)}_{j \to i}(s)$ [15]:
$$\lambda^{(\ell)}_{j \to i}(s) = \tau \exp\left( \gamma \cdot S^{(\ell)}_{j \to i}(s) \right)$$
where:
* $\tau > 0$ is the base Poisson rate hyperparameter (e.g., $\tau = 0.5$) [15].
* $\gamma > 0$ is a temperature scaling parameter to bound logit scales [13].

The **Spatial Evidence** (the total integrated foreground intensity across all agents at location $s$) is defined as [15]:
$$\Lambda^{(\ell)}_i(s) = \sum_{j \in \mathcal{A}} \lambda^{(\ell)}_{j \to i}(s)$$

This evidence acts as an unnormalised "coverage and presence meter" for the BEV plane [6, 17].

### 2.3 The Spatial Poisson Measure
For each grid cell $s$, we define a Poisson random variable $Z_s$ representing the number of stochastic samples (atoms) scattered under the intensity field [18, 19]:
$$Z_s \sim \text{Poisson}\left(\Lambda^{(\ell)}_i(s)\right)$$

Conditional on $Z_s = z \ge 1$, each of the $z$ atoms is assigned to collaborating agent $j$ independently with multinomial probability [25]:
$$p^{(\ell)}_{j \to i}(s) = \frac{\lambda^{(\ell)}_{j \to i}(s)}{\Lambda^{(\ell)}_i(s)} = \frac{\exp\left(\gamma \cdot S^{(\ell)}_{j \to i}(s)\right)}{\sum_{k \in \mathcal{A}} \exp\left(\gamma \cdot S^{(\ell)}_{k \to i}(s)\right)}$$

Notice that for $\gamma = 1$, $p^{(\ell)}_{j \to i}(s)$ is **algebraically identical** to the standard HEAL fusion weight $W^{(\ell)}_{j \to i}(s)$ [123].

---

## 3. Stochastic Fusion and the Unbiased Mean Path

Under this Poisson formulation, the stochastic fused feature output at cell $s$ is the average over the values at the drawn atoms [20, 25]:
$$F^{(\ell)}_{i, \text{stochastic}}(s) = \begin{cases} 
\frac{1}{Z_s} \sum_{k=1}^{Z_s} F^{(\ell)}_{J_k \to i}(s), & Z_s \ge 1 \\
\bar{F}^{(\ell)}_i(s), & Z_s = 0 
\end{cases}$$
where $J_k \in \mathcal{A}$ is the agent selected by the $k$-th atom draw [20].

### 3.1 Theorem 1: The Mean Identity (Exact Training Path)
Following **Theorem 1** of Lévy Attention, the mathematical expectation of the stochastic operator is exactly equal to its deterministic mean [30]:
$$\mathbb{E}\left[F^{(\ell)}_{i, \text{stochastic}}(s)\right] = \bar{F}^{(\ell)}_i(s) = \sum_{j \in \mathcal{A}} p^{(\ell)}_{j \to i}(s) F^{(\ell)}_{j \to i}(s)$$

**Proof Sketch:**
Given $Z_s = z \ge 1$, the selected agents $J_1, \dots, J_z$ are drawn i.i.d. according to the probability vector $p^{(\ell)}_{j \to i}(s)$ [100]. Therefore, the conditional expectation is:
$$\mathbb{E}\left[\frac{1}{z} \sum_{k=1}^{z} F^{(\ell)}_{J_k \to i}(s) \;\middle|\; Z_s = z\right] = \mathbb{E}\left[F^{(\ell)}_{J \to i}(s)\right] = \sum_{j \in \mathcal{A}} p^{(\ell)}_{j \to i}(s) F^{(\ell)}_{j \to i}(s) = \bar{F}^{(\ell)}_i(s)$$
For the empty-draw branch ($Z_s = 0$), the operator falls back to $\bar{F}^{(\ell)}_i(s)$ by definition, ensuring unbiasedness across all regimes [101].

### 3.2 Engineering Impact
During **Collaboration Base Training** and bulk deterministic inference, we bypass stochastic sampling entirely and propagate the deterministic expression $\bar{F}^{(\ell)}_i(s)$ [26]. This guarantees:
1. **Exact Differentiability:** Standard backpropagation is executed with exact gradients [26].
2. **Zero Training/Inference Gap:** The network trains on the exact expectation of the stochastic process [26].
3. **No Added Parameters:** The framework only adds a single scalar scaling parameter ($\gamma$) if tuned [51, 84].

---

## 4. Single-Pass Closed-Form Spatial Uncertainty

While the model runs along the deterministic mean path to output fused BEV features, it simultaneously meters the underlying spatial randomness to emit three spatial diagnostics per cell $s = (x, y)$ [27, 32].

### 4.1 Metric 1: Spatial Evidence (Coverage & Visibility)
The absolute unnormalised sum of foreground intensity at cell $s$ [15]:
$$\Lambda^{(\ell)}_i(s) = \tau \sum_{j \in \mathcal{A}} \exp\left(\gamma \cdot S^{(\ell)}_{j \to i}(s)\right)$$
* **Physical Interpretation:** Tracks how many agents are observing location $s$ and their absolute foreground confidence [6]. If a region is occluded for all agents, $S^{(\ell)}_{j \to i}(s) \to -\infty$, driving $\Lambda^{(\ell)}_i(s) \to 0$ [16].

### 4.2 Metric 2: Spatial Disagreement (Sensor Conflict)
The weighted variance of the aligned feature vectors across agents at cell $s$ [27]:
$$\text{tr }\mathbf{\Sigma}^{(\ell)}_i(s) = \sum_{j \in \mathcal{A}} p^{(\ell)}_{j \to i}(s) \left\| F^{(\ell)}_{j \to i}(s) \right\|^2_2 - \left\| \bar{F}^{(\ell)}_i(s) \right\|^2_2$$
* **Physical Interpretation:** Captures modality conflicts [9]. If a LiDAR agent detects a flat road (low activation) but a Camera agent detects a vehicle (high activation), their aligned features $F^{(\ell)}_{j \to i}(s)$ will diverge, leading to a high disagreement spike [9].

### 4.3 Metric 3: Spatial Uncertainty Scale (Continuous Calibration)
Using **Theorem 2** (the exact variance identity), we calculate the exact standard deviation of the stochastic operator without drawing any samples [2, 32]:
$$\hat{\sigma}^{(\ell)}_i(s) = \sqrt{\text{tr }\mathbf{\Sigma}^{(\ell)}_i(s) \cdot \varphi\left(\Lambda^{(\ell)}_i(s)\right)}$$
where:
$$\varphi(\Lambda) = \frac{e^{-\Lambda} \int_0^\Lambda \frac{e^t - 1}{t} dt}{\Lambda} \approx \frac{1}{\Lambda}$$
This yields the clean, intuitive spatial factorisation [32]:
$$\hat{\sigma}^{(\ell)}_i(s) \approx \sqrt{\frac{\text{Disagreement}(s)}{\text{Evidence}(s)}}$$

---

## 5. Multi-Scale Integration: The Spatial Trust Map

HEAL fuses features across multiple BEV grid scales $\ell \in \{1, 2, \dots, L\}$ (e.g., $L=3$) to capture both coarse-grained and fine-grained spatial representations [123, 125]. 

To build a singular, unified **BEV Spatial Trust Map** $\mathbf{U}_i \in \mathbb{R}^{H \times W}$, we aggregate the multi-scale spatial uncertainty scales $\hat{\sigma}^{(\ell)}_i$:

1. **Upsampling to Unified BEV Resolution:**
   For each scale $\ell$, we upsample the local scale-specific uncertainty map $\hat{\sigma}^{(\ell)}_i \in \mathbb{R}^{H_\ell \times W_\ell}$ to the final BEV map resolution $H \times W$ using bilinear interpolation [124]:
   $$\hat{\sigma}^{(\ell), \uparrow}_i = u_\ell\left(\hat{\sigma}^{(\ell)}_i\right)$$
   
2. **Pyramid Scale Aggregation:**
   The unified spatial uncertainty scale $\hat{\sigma}_i(x, y)$ is computed as a weighted average over the upsampled multi-scale uncertainties:
   $$\hat{\sigma}_i(x, y) = \sum_{\ell=1}^L \omega_\ell \cdot \hat{\sigma}^{(\ell), \uparrow}_i(x, y)$$
   where $\omega_\ell$ are hyperparameter scale weights (optionally tied to HEAL's multi-scale foreground loss coefficients $\alpha_\ell$) [126].

3. **Validation Calibration:**
   To turn the raw layer-level feature deviation scale $\hat{\sigma}_i(x, y)$ into a physical error scale $s_i(x, y)$ in the units of 3D bounding box coordinates or classification logits, we apply a validation-calibrated global affine map [45, 66]:
   $$s_i(x, y) = a \cdot \hat{\sigma}_i(x, y) + b$$
   The parameters $(a, b)$ are fitted once on a validation split using a coarse grid search to minimise the Continuous Ranked Probability Score (CRPS) [45, 67].

---

## 6. Preserving Backward Alignment & Privacy

A core strength of HEAL is **Backward Alignment**, which allows new agent types to integrate into an established alliance by training *only* their front-end encoder locally on single-agent data, preserving data and model privacy [112, 127]. 

Our Lévy reformulation preserves this property perfectly:
1. **No Shared Parameters in Fusion:** The Pyramid Fusion back-end $f_{\text{pyramid fusion}}^*$ is completely frozen during new agent training [128].
2. **Self-Contained Encoder Adaptation:** When a new agent type $n_1$ is introduced, its encoder $f_{\text{encoder}[n_1]}$ is trained individually on single-agent data [128, 129]. 
3. **Implicit Uncertainty Alignment:** By forcing the new encoder's features to pass through the frozen, foreground-supervised, multi-scale Pyramid layers, the new encoder automatically learns to map its features not only to the spatial coordinate system, but to the **exact unnormalised confidence range** expected by the evidence and disagreement estimators.

---

## 7. PyTorch Implementation Blueprint

Below is the PyTorch-like pseudo-code implementing `LevyPyramidFusion` for HEAL:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LevyPyramidFusion(nn.Module):
    def __init__(self, channels=[64, 128, 256], num_scales=3, rate_tau=0.5, temp_gamma=1.0):
        super().__init__()
        self.num_scales = num_scales
        self.tau = rate_tau
        self.gamma = temp_gamma
        
        # Unchanged ResNeXt layers from HEAL
        self.resnext_layers = nn.ModuleList([
            # R_1, R_2, R_3...
        ])
        
        # Unchanged foreground estimators (e.g., 1x1 convolutions)
        self.foreground_estimators = nn.ModuleList([
            nn.Conv2d(c, 1, kernel_size=1) for c in channels
        ])

    @staticmethod
    def compute_phi(lambda_val, eps=1e-6):
        """
        Approximates phi(Lambda) = E[1{Z>=1}/Z] for Poisson(Lambda).
        For Lambda >= 2, phi(Lambda) matches 1/Lambda.
        """
        # Safe inverse approximation with fallback near zero (maximum near Lambda = 1.5)
        # Levy paper: maximum is ~0.517 near Lambda = 1.503
        phi = torch.where(
            lambda_val > 1.5,
            1.0 / (lambda_val + eps),
            0.517 * (lambda_val / 1.5)  # Soft linear ramp down to 0 at Lambda = 0
        )
        return phi

    def forward(self, agent_features_aligned):
        """
        Args:
            agent_features_aligned: List of tensors, each of shape [B, M, C_0, H, W]
                                    representing aligned features from M agents.
        Returns:
            fused_features: [B, C_fused, H, W] - Deterministic mean path for detection.
            spatial_uncertainty: [B, 1, H, W] - Unified spatial uncertainty scale map.
        """
        B, M, _, H_base, W_base = agent_features_aligned[0].shape
        scale_fused_features = []
        scale_uncertainties = []

        # Iterate through spatial pyramid scales
        for l in range(self.num_scales):
            # 1. Feature Representation (HEAL)
            # Collect and process features for all agents at this scale
            F_scale = []
            for j in range(M):
                feat = agent_features_aligned[j][:, l] # [B, C, H_l, W_l]
                feat = self.resnext_layers[l](feat)
                F_scale.append(feat)
            
            F_scale = torch.stack(F_scale, dim=1) # [B, M, C, H_l, W_l]
            
            # 2. Foreground Estimators (HEAL)
            # Process each agent's feature map to get unnormalised logit maps
            S_scale = []
            for j in range(M):
                S_scale.append(self.foreground_estimators[l](F_scale[:, j]))
            S_scale = torch.stack(S_scale, dim=1) # [B, M, 1, H_l, W_l]

            # 3. Lévy Attention Reformulation
            # Compute unnormalised intensities
            lambda_scale = self.tau * torch.exp(self.gamma * S_scale) # [B, M, 1, H_l, W_l]
            
            # Spatial Evidence (Lambda_q)
            evidence = torch.sum(lambda_scale, dim=1) # [B, 1, H_l, W_l]
            
            # Normalised Fusion Weights (p_l)
            weights = lambda_scale / (evidence.unsqueeze(1) + 1e-6) # [B, M, 1, H_l, W_l]

            # Fused Feature Mean Path (ā(q))
            fused_scale = torch.sum(F_scale * weights, dim=1) # [B, C, H_l, W_l]
            scale_fused_features.append(fused_scale)

            # 4. Stochastic Variance (Disagreement) Metering
            # Var(V) = E[V^2] - (E[V])^2
            fused_sq = torch.sum((F_scale ** 2) * weights, dim=1) # [B, C, H_l, W_l]
            disagreement = torch.sum(fused_sq - (fused_scale ** 2), dim=1, keepdim=True) # [B, 1, H_l, W_l]

            # 5. Combined Lévy Scale (sigma_hat)
            phi_ev = self.compute_phi(evidence) # [B, 1, H_l, W_l]
            sigma_hat = torch.sqrt(F.relu(disagreement * phi_ev)) # [B, 1, H_l, W_l]
            
            # Bilinear upsample to base BEV resolution
            if l > 0:
                sigma_hat = F.interpolate(sigma_hat, size=(H_base, W_base), mode='bilinear', align_corners=False)
            scale_uncertainties.append(sigma_hat)

        # Multi-Scale Pyramid Output Fusion
        # Concatenate scale-specific mean features (as in HEAL)
        # scale_fused_features are upsampled and concatenated
        final_fused_feature = self.heal_concat_and_upsample(scale_fused_features)
        
        # Average uncertainties across the pyramid scales
        unified_uncertainty = torch.mean(torch.stack(scale_uncertainties, dim=0), dim=0)

        return final_fused_feature, unified_uncertainty
```

---

## 8. Summary of Benefits & Downstream Applications

Applying this Lévy blueprint directly upgrades HEAL with robust safety features:
1. **Sensing Triage:** The spatial uncertainty scale $\hat{\sigma}_i(x,y)$ alerts autonomous planners to slow down when approaching zones with high disagreement or low evidence [112].
2. **Querying Occluded Gaps:** If a crucial BEV sector reports high uncertainty due to low evidence, the ego vehicle can dynamically request camera transmissions from surrounding agents *only* for that specific coordinate, conserving wireless bandwidth [153].
3. **Calibrated Out-of-Distribution Detection:** Rogue agents transmitting corrupt sensor noise are immediately exposed as spatial disagreement spikes, protecting the alliance from adversarial feature injections [156].

---
### References
* `[1]` Chapter 1: Introduction, Page 1.
* `[2]` Chapter 1: Introduction, Page 1.
* `[6]` Chapter 1: Intuition, Page 1.
* `[13]` Chapter 2.2: Compatibility and Intensity, Page 1.
* `[15]` Chapter 2.2: Attention Intensity, Page 2.
* `[18]` Chapter 2.2: Random Measure, Page 2.
* `[20]` Chapter 2.2: Output, Page 2.
* `[25]` Chapter 2.3: Sampling on the Grid, Page 3.
* `[26]` Chapter 2.4: Deterministic Mean Mode, Page 3.
* `[27]` Chapter 2.4: Deterministic Mean Mode, Page 3.
* `[30]` Chapter 3: Theorem 1 (Mean Identity), Page 4.
* `[32]` Chapter 3: Theorem 2 (Exact Variance Identity), Page 4.
* `[45]` Chapter 5.1: Calibration, Page 5.
* `[108]` HEAL Abstract, Page 14.
* `[109]` HEAL Section 1: Introduction, Page 15.
* `[112]` HEAL Section 1: Core Idea and Training Phases, Page 15.
* `[121]` HEAL Section 4.1: Collaboration Base, Page 17.
* `[122]` HEAL Section 4.1: Feature maps, Page 17.
* `[123]` HEAL Section 4.1: Pyramid Fusion equations, Page 18.
* `[124]` HEAL Section 4.1: Foreground Estimators and Softmax, Page 18.
* `[125]` HEAL Section 4.1: Pyramid Fusion properties, Page 18.
* `[126]` HEAL Section 4.1: Loss equations, Page 18.
* `[127]` HEAL Section 4.2: New Agent Training, Page 19.
* `[128]` HEAL Section 4.2: Encoder Adaptations and Backward Alignment, Page 19.
* `[129]` HEAL Section 4.2: Individual training, Page 19.
