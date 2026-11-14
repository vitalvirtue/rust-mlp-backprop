# Experiment results: Iris

150 samples, 4 features, 3 classes (Iris-setosa 50, Iris-versicolor 50, Iris-virginica 50). Stratified split: train 105 (Iris-setosa 35, Iris-versicolor 35, Iris-virginica 35), validation 24 (Iris-setosa 8, Iris-versicolor 8, Iris-virginica 8), test 21 (Iris-setosa 7, Iris-versicolor 7, Iris-virginica 7).

Majority-class baseline (always predict `Iris-virginica`): test accuracy **0.3333**.

Grid: activations ["sigmoid", "tanh", "relu"], learning rates [0.001, 0.01, 0.05, 0.1], hidden sizes [2, 4, 8, 16, 32]; 5 seeds × 300 epochs, batch size 16.

## Final validation loss (mean ± std over seeds)

### sigmoid

| lr \ hidden | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 0.001 | 1.0779 ± 0.0264 | 0.9262 ± 0.0520 | 0.8752 ± 0.0911 | 0.7926 ± 0.0627 | 0.8183 ± 0.0449 |
| 0.01 | 0.6492 ± 0.0569 | 0.4422 ± 0.0258 | 0.3701 ± 0.0195 | 0.2972 ± 0.0144 | 0.2696 ± 0.0074 |
| 0.05 | 0.2023 ± 0.0482 | 0.1249 ± 0.0188 | 0.1037 ± 0.0042 | 0.0840 ± 0.0025 | 0.0765 ± 0.0010 |
| 0.1 | 0.0876 ± 0.0084 | 0.0656 ± 0.0029 | 0.0622 ± 0.0024 | 0.0562 ± 0.0010 | 0.0553 ± 0.0014 |

### tanh

| lr \ hidden | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 0.001 | 0.8587 ± 0.0606 | 0.5396 ± 0.0528 | 0.4549 ± 0.0332 | 0.3648 ± 0.0293 | 0.3450 ± 0.0172 |
| 0.01 | 0.2404 ± 0.0710 | 0.1484 ± 0.0277 | 0.1145 ± 0.0171 | 0.0914 ± 0.0059 | 0.0887 ± 0.0032 |
| 0.05 | 0.0628 ± 0.0030 | 0.0563 ± 0.0041 | 0.0524 ± 0.0036 | 0.0500 ± 0.0018 | 0.0479 ± 0.0012 |
| 0.1 | 0.0580 ± 0.0033 | 0.0577 ± 0.0114 | 0.0526 ± 0.0075 | 0.0499 ± 0.0024 | 0.0487 ± 0.0035 |

### relu

| lr \ hidden | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 0.001 | 0.9282 ± 0.0660 | 0.5329 ± 0.0460 | 0.4371 ± 0.1122 | 0.3399 ± 0.0813 | 0.3053 ± 0.0350 |
| 0.01 | 0.2950 ± 0.0683 | 0.1001 ± 0.0141 | 0.1004 ± 0.0282 | 0.0896 ± 0.0101 | 0.0830 ± 0.0114 |
| 0.05 | 0.0792 ± 0.0394 | 0.0473 ± 0.0028 | 0.0466 ± 0.0023 | 0.0407 ± 0.0075 | 0.0390 ± 0.0041 |
| 0.1 | 0.0672 ± 0.0372 | 0.0466 ± 0.0091 | 0.0407 ± 0.0044 | **0.0368 ± 0.0075** | 0.0381 ± 0.0057 |

## Best configuration per activation, evaluated on the test set

| activation | hidden | lr | val loss | val acc | test acc | macro precision | macro recall | macro F1 |
|---|---|---|---|---|---|---|---|---|
| sigmoid | 32 | 0.1 | 0.0553 | 0.9583 | 0.9524 | 0.9583 | 0.9524 | 0.9521 |
| tanh | 32 | 0.05 | 0.0479 | 0.9583 | 0.9524 | 0.9583 | 0.9524 | 0.9521 |
| relu | 16 | 0.1 | 0.0368 | 0.9750 | 0.9524 | 0.9583 | 0.9524 | 0.9521 |

## Best overall model: relu · h=16 · lr=0.1

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Iris-setosa | 1.0000 | 1.0000 | 1.0000 | 7 |
| Iris-versicolor | 1.0000 | 0.8571 | 0.9231 | 7 |
| Iris-virginica | 0.8750 | 1.0000 | 0.9333 | 7 |
| **macro avg** | 0.9583 | 0.9524 | 0.9521 | 21 |
| **weighted avg** | 0.9583 | 0.9524 | 0.9521 | 21 |

Accuracy: **0.9524** (majority-class baseline 0.3333)

### Confusion matrix (rows = actual, columns = predicted)

| | Iris-setosa | Iris-versicolor | Iris-virginica |
|---|---|---|---|
| **Iris-setosa** | 7 | 0 | 0 |
| **Iris-versicolor** | 0 | 6 | 1 |
| **Iris-virginica** | 0 | 0 | 7 |
