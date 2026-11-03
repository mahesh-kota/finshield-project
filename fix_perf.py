with open('docs/performance_summary.md', 'r', encoding='utf-8') as f:
    t = f.read()

# Fix the metrics table
t = t.replace(
    '| Accuracy | 88.0% | 91.4% | **93.3%** |',
    '| Macro-F1 | — | — | **0.570** |'
)
t = t.replace(
    '| Weighted F1 | 0.82 | 0.92 | **0.94** |',
    '| Neutral F1 | — | — | **0.000** |'
)
t = t.replace(
    '| Suspicious F1 | 0.92 | 0.91 | **0.95** |',
    '| Suspicious F1 | — | — | **0.897** |'
)

# Fix Granger note
t = t.replace(
    '**Note:** Granger result is sensitive to dataset composition. The 174-post result (p=0.004) is the more cited finding. Larger social corpus needed for robust confirmation.',
    '**Note:** Granger result was corrected Nov 2, 2026. Re-analysis with proper stationarity checks showed no robust evidence that social signals lead market anomalies. Reported as a null result.'
)

# Fix recall
t = t.replace(
    '| Recall | 75% (3/4 cases) |',
    '| Recall | Superseded — see Sadhna case limitation (Nov 3) |'
)

# Add correction banner
banner = '> ⚠️ **CORRECTION (Nov 3, 2026):** Metrics in this document were found to be incorrect on re-measurement. MuRIL v3 actual result: macro-F1 0.570, neutral F1 0.000. SEBI backtest superseded. Granger causality corrected to null result. See FAILURES.md for details.\n\n'
if not t.startswith('> ⚠️'):
    t = banner + t

with open('docs/performance_summary.md', 'w', encoding='utf-8') as f:
    f.write(t)
print("performance_summary.md fixed")
