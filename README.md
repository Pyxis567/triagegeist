# triagegeist

> [!WARNING]
> 🚧 **This project is under active construction.** Expect incomplete sections, breaking changes, and work-in-progress code.

---

## Competition

**[Triagegeist — Kaggle](https://www.kaggle.com/competitions/triagegeist)**

The challenge is to predict the **Emergency Severity Index (ESI) triage acuity level** assigned to a patient upon arrival at the emergency department. Given structured clinical and demographic data collected at triage, the model must classify each patient into one of five urgency levels:

| Level | Label | Description |
|-------|-------|-------------|
| 1 | Resuscitation | Immediately life-threatening |
| 2 | Emergent | High risk, should not wait |
| 3 | Urgent | Stable but requires multiple resources |
| 4 | Less Urgent | Stable, requires one resource |
| 5 | Non-Urgent | Stable, no resources needed |

**Evaluation metric:** Macro F1-score across all five classes.

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 80,000 | Labeled patient records (40 features + target) |
| `test.csv` | — | Unlabeled patient records for submission |
| `chief_complaints.csv` | — | Raw free-text chief complaint per patient |
| `patient_history.csv` | — | 25 binary comorbidity flags per patient |

Key feature groups: **vital signs** (BP, HR, SpO2, temp, respiratory rate), **demographics** (age, sex, insurance), **clinical scores** (NEWS2, GCS, pain score), **arrival context** (mode, time, shift), and **prior utilization** (ED visits and admissions in past 12 months).

---


