# Bank Marketing: Predicting Term Deposit Subscriptions

Which clients of a Portuguese bank are most likely to subscribe to a term deposit after a phone campaign? This project combines **descriptive analysis**, **supervised classification** and **customer segmentation** on 41,188 clients.

*Individual assignment, Data Mining & Machine Learning, EDHEC Business School.*

## Approach

1. **Descriptive analysis**: distributions, class imbalance (only 11.3% of clients subscribe), correlations, subscription rate by job
2. **Classification**: scikit-learn pipeline (standardisation + one-hot encoding) with a **logistic regression**, 5-fold stratified cross-validation
3. **Clustering**: **K-Means** customer segmentation, number of clusters chosen with the elbow method
4. **Interpretation**: model coefficients and cluster profiles turned into marketing recommendations

## Results

**Classification (test set)**

| Metric | Value |
|---|---|
| ROC AUC | 0.942 |
| Accuracy | 0.917 |
| Precision (subscribers) | 0.712 |
| Recall (subscribers) | 0.436 |
| F1-score (subscribers) | 0.541 |

Because the classes are imbalanced, ROC AUC and recall matter more than accuracy.

**Segmentation (K-Means, k = 3)**

| Cluster | Clients | Subscription rate | Profile |
|---|---|---|---|
| 0 | 27,646 | 4.8% | Contacted when interest rates were high, never contacted before |
| 1 | 12,006 | 19.4% | Contacted in a low-rate environment |
| 2 | 1,536 | **63.3%** | Already contacted in a previous campaign |

**Takeaway:** follow-ups with clients reached in earlier campaigns are by far the most effective, and campaign timing relative to interest rates matters more than the number of calls.

## Limits and next steps

- `duration` (call length) is only known after the call. It should be removed for a truly predictive model.
- Handle class imbalance explicitly (class weights, threshold tuning) to improve recall.
- Compare with tree-based models (Random Forest, gradient boosting).

## Run it

Data: [Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) (UCI Machine Learning Repository). Put `bank-additional-full.csv` in a `data/` folder.

```bash
pip install -r requirements.txt
jupyter notebook bank_marketing.ipynb
```
