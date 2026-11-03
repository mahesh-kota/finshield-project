with open('docs/weekly_notes.md', 'r', encoding='utf-8') as f:
    w = f.read()

w = w.replace(
    '**- Day 16-18: MuRIL fine-tuning v1 and v2 - final weighted F1=0.92, accuracy=91.4%**',
    '**- Day 16-18: MuRIL fine-tuning v1 and v2 (superseded — see Nov 3 correction: v3 macro-F1 0.570)**'
)
w = w.replace(
    '**- Day 32: MuRIL v3 fine-tuned on expanded dataset — accuracy 93.3%, weighted F1=0.94 (up from 91.4%/0.92)**',
    '**- Day 32: MuRIL v3 fine-tuned on expanded dataset — macro-F1 0.570, neutral F1 0.000 (label quality issue diagnosed Nov 3)**'
)
w = w.replace(
    '**- MuRIL v3 improves on v2: accuracy 91.4% → 93.3%, F1 0.92 → 0.94, Suspicious F1 0.91 → 0.95**',
    '**- MuRIL v3 result: macro-F1 0.570; open_tip F1 0.815, suspicious F1 0.897, neutral F1 0.000**'
)
w = w.replace(
    '**- Day 24: SEBI backtest - 75% recall (3/4 cases), Precision@8=0.38, 11-day early detection**',
    '**- Day 24: SEBI backtest — superseded Nov 3: unsupervised detector showed no lift on Sadhna case; reported as limitation**'
)
w = w.replace(
    '**- SEBI backtest unchanged: 75% recall, 11-day early detection on Sharpline case**',
    '**- SEBI backtest — superseded Nov 3: see market module limitation note**'
)

banner = '> ⚠️ **CORRECTION (Nov 3, 2026):** Several entries in this log contain numbers that were later found to be incorrect. The MuRIL v3 result was measured at macro-F1 0.570 (neutral F1 0.000), not 0.94. The SEBI backtest was superseded by the Sadhna case finding. See inline corrections below.\n\n'
if not w.startswith('> ⚠️'):
    w = banner + w

with open('docs/weekly_notes.md', 'w', encoding='utf-8') as f:
    f.write(w)
print("weekly_notes.md fixed")
