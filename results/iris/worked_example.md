# Worked example: one training step (Iris)

Network [4, 3, 3], hidden activation **sigmoid**, softmax output, cross-entropy loss, learning rate η = 0.1, mini-batch of N = 2 samples.

## 0. Input mini-batch

Samples are taken from the training set and z-score standardized with the training mean μ = [5.8590, 3.0829, 3.7686, 1.2133] and standard deviation σ = [0.7730, 0.4223, 1.7199, 0.7606]. Classes: ["Iris-setosa", "Iris-versicolor"].

**X** (2×4)

```text
[ -1.3701,   0.7509,  -1.0864,  -1.3323
  -0.3351,  -0.1962,   0.1927,   0.1139]
```

**Y (one-hot)** (2×3)

```text
[  1.0000,   0.0000,   0.0000
   0.0000,   1.0000,   0.0000]
```


## 1. Initial parameters

Weights ~ N(0, σ²) with Xavier, σ = √(2/(fan_in+fan_out)) initialization; biases are zero.

**W¹** (4×3)

```text
[  0.9443,  -0.3352,   0.1647
  -0.7356,   0.5711,  -0.7507
  -0.7147,   0.1219,   0.3826
  -0.0354,   0.2664,   0.1278]
```

**b¹** (1×3)

```text
[  0.0000,   0.0000,   0.0000]
```

**W²** (3×3)

```text
[  0.5032,  -0.4657,  -0.2433
   0.5939,  -0.2179,  -0.1277
  -0.7347,  -0.3125,  -0.1395]
```

**b²** (1×3)

```text
[  0.0000,   0.0000,   0.0000]
```


## 2. Forward propagation

Z¹ = X·W¹ + b¹

For example, the first hidden neuron for sample 1: z¹₁₁ = (-1.3701)·(0.9443) + (0.7509)·(-0.7356) + (-1.0864)·(-0.7147) + (-1.3323)·(-0.0354) + 0.0000 = **-1.0226**

**Z¹** (2×3)

```text
[ -1.0226,   0.4007,  -1.3754
  -0.3139,   0.0541,   0.1804]
```

A¹ = sigmoid(Z¹), e.g. sigmoid(-1.0226) = **0.2645**

**A¹** (2×3)

```text
[  0.2645,   0.5989,   0.2018
   0.4222,   0.5135,   0.5450]
```

Z² = A¹·W² + b²

**Z²** (2×3)

```text
[  0.3406,  -0.3167,  -0.1690
   0.1170,  -0.4788,  -0.2443]
```

P = softmax(Z²), row-wise: pₖ = e^(zₖ − max z) / Σⱼ e^(zⱼ − max z). For sample 1 the shifted exponentials are [1.0000, 0.5183, 0.6008], which normalize to [0.4719, 0.2446, 0.2835].

**P** (2×3)

```text
[  0.4719,   0.2446,   0.2835
   0.4449,   0.2452,   0.3100]
```


## 3. Loss

L = −(1/N) Σₙ Σₖ yₙₖ ln pₙₖ = −(1/2)·[ln(0.4719) + ln(0.2452)] = **1.0784**


## 4. Backpropagation

Output layer: δ² = ∂L/∂Z² = (P − Y) / N

**δ²** (2×3)

```text
[ -0.2640,   0.1223,   0.1418
   0.2224,  -0.3774,   0.1550]
```

∂L/∂W² = (A¹)ᵀ · δ²

For example, ∂L/∂w²₁₁ = a¹₁₁·δ²₁₁ + a¹₂₁·δ²₂₁ = (0.2645)·(-0.2640) + (0.4222)·(0.2224) = **0.0241**

**∂L/∂W²** (3×3)

```text
[  0.0241,  -0.1270,   0.1029
  -0.0439,  -0.1206,   0.1645
   0.0680,  -0.1810,   0.1131]
```

∂L/∂b² = Σ over rows of δ²

**∂L/∂b²** (1×3)

```text
[ -0.0416,  -0.2551,   0.2967]
```

Hidden layer: δ¹ = (δ² · (W²)ᵀ) ⊙ sigmoid′(Z¹), where σ′(z) = σ(z)(1 − σ(z))

**sigmoid′(Z¹)** (2×3)

```text
[  0.1946,   0.2402,   0.1610
   0.2439,   0.2498,   0.2480]
```

**δ¹** (2×3)

```text
[ -0.0436,  -0.0484,   0.0219
   0.0610,   0.0486,  -0.0166]
```

∂L/∂W¹ = Xᵀ · δ¹ and ∂L/∂b¹ = Σ over rows of δ¹

**∂L/∂W¹** (4×3)

```text
[  0.0394,   0.0501,  -0.0244
  -0.0447,  -0.0459,   0.0197
   0.0592,   0.0620,  -0.0270
   0.0651,   0.0701,  -0.0311]
```

**∂L/∂b¹** (1×3)

```text
[  0.0173,   0.0002,   0.0053]
```


## 5. Gradient descent update

θ ← θ − η·∂L/∂θ with η = 0.1. For example, w²₁₁ ← 0.5032 − 0.1·(0.0241) = **0.5008**

**W¹ (updated)** (4×3)

```text
[  0.9403,  -0.3402,   0.1672
  -0.7311,   0.5756,  -0.7527
  -0.7206,   0.1157,   0.3853
  -0.0419,   0.2594,   0.1309]
```

**b¹ (updated)** (1×3)

```text
[ -0.0017,  -0.0000,  -0.0005]
```

**W² (updated)** (3×3)

```text
[  0.5008,  -0.4530,  -0.2536
   0.5983,  -0.2058,  -0.1442
  -0.7415,  -0.2944,  -0.1508]
```

**b² (updated)** (1×3)

```text
[  0.0042,   0.0255,  -0.0297]
```


## 6. Check

Loss on the same mini-batch after the update: 1.0784 → **1.0486** (-0.0298). One gradient step moved the parameters downhill.

