# Design Decisions

Short log of non-obvious choices and their trade-offs.

## 2026-06-29: Dropped Twitter/X
- **Decision:** Exclude Twitter from data collection.
- **Reason:** Free tier too restrictive; snscrape incompatible with Python 3.13.
- **Trade-off:** Lost a major data source. Telegram + Reddit is thinner.
- **Reversible?** Yes, if paid API access becomes available.

## 2026-07-02: MongoDB to CSV/pandas
- **Decision:** Use CSV files instead of MongoDB Atlas.
- **Reason:** SSL handshake failures on college network; dataset is small enough (224 posts) that pandas is sufficient.
- **Trade-off:** No concurrent access, no query language. Fine for this scale.
- **Reversible?** Yes, if dataset grows past ~10K posts.

## 2026-07-08: MuRIL over mBERT
- **Decision:** Use MuRIL-base-cased instead of mBERT.
- **Reason:** MuRIL is pretrained on Indian languages including transliterated Hindi. mBERT is not.
- **Trade-off:** Larger model (178M params), slower training.
- **Reversible?** Yes, but mBERT is likely worse for Hinglish.

## 2026-07-12: Isolation Forest over autoencoder
- **Decision:** Use Isolation Forest for cross-modal anomaly detection.
- **Reason:** 73 overlapping days is too few for a neural autoencoder to train.
- **Trade-off:** IF is less expressive but trainable on this data size.
- **Reversible?** Yes, if we collect more data.

## 2026-11-03: Keep both MuRIL and TF-IDF
- **Decision:** Report both models, use TF-IDF as the production classifier.
- **Reason:** TF-IDF achieves macro-F1 0.844 vs MuRIL's 0.570. TF-IDF is faster, smaller, and more interpretable.
- **Trade-off:** TF-IDF misclassified 3 suspicious tips as neutral (worse error type than MuRIL's 3 suspicious-to-open_tip). With more data, MuRIL may catch up.
- **Reversible?** Yes, if we collect 500+ more labels.
