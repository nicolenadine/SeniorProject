# Statistical Comparison Report: Full Image vs. Segment Models

*Report generated on: 2025-04-13 15:28:43*

## Overall Results

### Wilcoxon Signed-Rank Test Results

**Classification Accuracy Test:**

- Result: statistically significant
 (p-value: 6.86e-07)

- Better model: Full Model

- Full model accuracy: 0.9745

- Segment model accuracy: 0.9489

- Absolute difference: 0.0255


**Log Loss Test:**

- Result: statistically significant (p-value: 0.0000)

- Better model: Full Model

- Full model log loss: 0.1749

- Segment model log loss: 0.1993

- Absolute difference: 0.0244


### McNemar's Test Results

- Result: statistically significant (p-value: 0.0000)

- Better model: Full Model

- Contingency table:

  * Both models correct: 1581

  * Only full model correct: 59

  * Only segment model correct: 16

  * Both models incorrect: 27


### Performance Metrics Comparison

| Metric | Full Model | Segment Model | Difference | % Improvement |

|--------|------------|---------------|------------|---------------|

| Accuracy | 0.9745 | 0.9489 | -0.0255 | -2.62% |

| Precision | 0.9730 | 0.9185 | -0.0545 | -5.60% |

| Recall | 0.9690 | 0.9704 | +0.0013 | +0.14% |

| F1 Score | 0.9710 | 0.9437 | -0.0273 | -2.81% |

| Auc | 0.9968 | 0.9835 | -0.0133 | -1.33% |


## Family-Level Analysis


## Conclusions

- The Full Model shows **statistically significant better performance** in terms of classification accuracy.

- The Full Model shows **statistically significant better performance** in terms of log loss (prediction confidence).

- McNemar's test shows a **statistically significant difference** in the pattern of errors between the two models, with the Full Model performing better.


**Overall recommendation:** The Full Model is recommended based on consistent superior performance.
