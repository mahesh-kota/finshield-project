with open('docs/performance_summary.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'Accuracy | 88.0%' in line:
        new_lines.append('| Macro-F1 | — | — | **0.570** |\n')
    elif 'Weighted F1 | 0.82' in line:
        new_lines.append('| Neutral F1 | — | — | **0.000** |\n')
    elif 'Suspicious F1 | 0.92' in line:
        new_lines.append('| Suspicious F1 | — | — | **0.897** |\n')
    elif 'Granger result is sensitive to dataset composition' in line:
        new_lines.append('**Note:** Granger result was corrected Nov 2, 2026. Re-analysis with proper stationarity checks showed no robust evidence that social signals lead market anomalies. Reported as a null result.\n')
    elif 'Recall | 75% (3/4 cases)' in line:
        new_lines.append('| Recall | Superseded — see Sadhna case limitation (Nov 3) |\n')
    else:
        new_lines.append(line)

with open('docs/performance_summary.md', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("performance_summary.md fixed (line-based)")
