with open('README.md', 'r', encoding='utf-8') as f:
    r = f.read()

r = r.replace(
    '| MuRIL v3 accuracy (45-post validation set) | 93.3% |',
    '| MuRIL v3 macro-F1 (45-post validation set) | 0.570 |'
)
r = r.replace(
    '| Weighted F1 | 0.94 |',
    '| MuRIL v3 neutral F1 | 0.000 |\n| TF-IDF+LR baseline macro-F1 | 0.844 |'
)
r = r.replace(
    '1. **NLP classification works**: MuRIL v3 separates Neutral, Open Tip and Suspicious posts at 93.3% accuracy (weighted F1 0.94) on a small, manually labelled set. Balanced class weights raised the minority Neutral class F1 from 0.00 to 0.73.',
    '1. **NLP classification is partially working**: MuRIL v3 separates Open Tip (F1 0.815) and Suspicious (F1 0.897) well, but never predicts Neutral (F1 0.000). A TF-IDF + logistic regression baseline achieves macro-F1 0.844, outperforming MuRIL\'s 0.570. Diagnosis: label quality, not model capacity, is the bottleneck — 4 of 5 misclassified neutrals were borderline cases containing advice or promotional language.'
)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(r)
print("README.md fixed")
