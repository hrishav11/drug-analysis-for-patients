# Drug Analysis for Patients

Analysis of patient drug reviews from the [UCI Drug Review dataset](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) (Drugs.com), in three parts:

1. **Cleaning and EDA** of 213,000+ reviews
2. **Efficacy classifier:** BioBERT fine-tuned to label each review's drug efficacy as negative, neutral or positive
3. **Drug recommender:** ranks drugs per condition by combined efficacy and safety scores, and filters out drugs with side effects a patient wants to avoid

> **Disclaimer:** this is a research and learning project built on anonymous online reviews. Its outputs are not medical advice and must not be used to choose, start or stop any medication.

## Data cleaning

- Decoded HTML entities (e.g. `&#039;`), normalised unicode and whitespace, and lowercased review text
- Removed invalid condition values (numeric-only entries and scraped HTML fragments)
- Fixed condition names that the raw dataset truncates (e.g. `bipolar disorde` → `bipolar disorder`, `acial wrinkles` → `facial wrinkles`)
- **Removed 85,443 duplicate reviews** (40% of the data) before splitting, leaving 128,438 unique reviews. The dataset repeats many reviews word for word, often under both a brand and a generic drug name. Without this step the same review can appear in both training and validation, which inflates accuracy.

## Efficacy classifier

Labels come from the star rating as weak supervision: **8–10 = positive, 5–7 = neutral, 1–4 = negative**. So the task is predicting the reviewer's verdict from the text of the review.

- **Model:** `dmis-lab/biobert-base-cased-v1.1`, fine-tuned for 2 epochs (learning rate 2e-5, batch size 8, max length 256, fp16)
- **Split:** stratified 80/20 on the deduplicated data (25,688 validation reviews)

### Results (validation set)

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Majority class (always "positive") | 0.604 | – | – |
| **BioBERT** | **0.819** | **0.812** | **0.717** |

| Class | Precision | Recall | F1 | Reviews |
|---|---|---|---|---|
| Positive | 0.890 | 0.917 | 0.904 | 15,526 |
| Negative | 0.800 | 0.834 | 0.817 | 6,364 |
| Neutral | 0.483 | 0.388 | 0.431 | 3,798 |

| Epoch | Train loss | Val loss | Accuracy | Weighted F1 |
|---|---|---|---|---|
| 1 | 0.500 | 0.486 | 0.815 | 0.793 |
| 2 | 0.367 | 0.524 | 0.819 | 0.812 |

**Takeaways:**
- The model is strong on clear opinions; positive and negative both exceed 0.8 F1.
- Neutral is the weak class. Ratings of 5–7 often mix praise and complaints, and neutral is also the smallest class (about 15%). Class weighting or an ordinal-regression formulation are natural next steps.
- Validation loss rises in epoch 2 while training loss keeps falling, an early sign of overfitting, so 2 epochs is a sensible stopping point. The best checkpoint is kept by F1.

## Drug recommender

1. **Side effects:** each review is matched against 11 common side-effect keywords (e.g. nausea, insomnia, weight gain, dizziness). A richer phrase-based extractor with negation handling (e.g. "can't sleep" → insomnia, ignoring "no nausea") is prototyped on a 500-review sample.
2. **Safety score** per drug, from how often its reviews mention side effects
3. **Efficacy score** per condition and drug, from the mean rating
4. **Final score** = 60% efficacy + 40% safety, with minimum-evidence thresholds (review counts, and how much of a drug's reviews relate to the condition)
5. **Constraints:** optionally exclude drugs where a chosen side effect is mentioned in at least 5% of reviews

Example: `recommend_drugs_with_constraints(condition="depression", avoid_side_effects=["insomnia"], top_n=5)`

## Running the notebook

Open `Drug_Analysis_For_Patients.ipynb` in Google Colab with a **GPU runtime** (Runtime → Change runtime type → GPU). Put `drugsComTrain_raw.csv` and `drugsComTest_raw.csv` in `MyDrive/`, then run all cells. Fine-tuning takes about 1.5 hours on a T4 GPU.

The dataset and trained model weights are not included in this repository.
