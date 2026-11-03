with open('docs/weekly_notes.md', 'r', encoding='utf-8') as f:
    t = f.read()

t = t.replace(
    '**- Day 21: Granger causality test - confirmed social signals lead market by 1-2 days (p=0.004)**',
    '**- Day 21: Granger causality test — superseded Nov 2: no robust evidence after stationarity checks**'
)
t = t.replace(
    '**- Granger causality p=0.004 at lag 1 (after first-differencing) — strongest quantitative finding**',
    '**- Granger causality — superseded Nov 2: no robust evidence. Reported as null result.**'
)

with open('docs/weekly_notes.md', 'w', encoding='utf-8') as f:
    f.write(t)
print("weekly_notes.md fixed")
