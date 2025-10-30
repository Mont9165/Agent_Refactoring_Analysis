import pandas as pd

data = [
    ('Maintainability', 267, 46.1, 2236, 43.7, 431, 64.1),
    ('Readability', 1525, 26.3, 1330, 26.0, 195, 29.0),
    ('Logical Mismatch', 615, 10.6, 612, 12.0, 3, 0.4),
    ('Repurpose/Reuse', 247, 4.3, 240, 4.7, 7, 1.0),
    ('Testability', 240, 4.1, 230, 4.5, 10, 1.5),
    ('Dependency', 141, 2.4, 138, 2.7, 3, 0.4),
    ('Legacy Code', 119, 2.1, 110, 2.1, 9, 1.3),
    ('Slow Performance', 93, 1.6, 89, 1.7, 4, 0.6),
    ('Hard to Debug', 89, 1.5, 86, 1.7, 3, 0.4),
    ('Duplication', 53, 0.9, 46, 0.9, 7, 1.0),
]

def fmt(count, pct):
    return f"{count:,} ({pct:.1f}\\%)"

with open('outputs/research_questions/tables/table_rq3_refactoring_purposes.tex', 'w') as tex:
    tex.write('\\begin{table}[h]\n')
    tex.write('\\centering\n')
    tex.write('\\caption{Distribution of refactoring purposes across Overall, Non-SAR, and SAR commits.}\n')
    tex.write('\\label{tab:rq3-purpose-comparison}\n')
    tex.write('\\begin{tabular}{lrrr}\n')
    tex.write('\\toprule\n')
    tex.write('\\textbf{Purpose Category} & \\textbf{Overall} & \\textbf{Non-SAR} & \\textbf{SAR} \\\\n')
    tex.write('\\midrule\n')
    for label, overall, overall_pct, non_sar, non_sar_pct, sar, sar_pct in data:
        tex.write(f"{label} & {fmt(overall, overall_pct)} & {fmt(non_sar, non_sar_pct)} & {fmt(sar, sar_pct)} \\\\n")
    tex.write('\\midrule\n')
    tex.write('\\textbf{Total Commits} & \\textbf{5,789} & \\textbf{5,117} & \\textbf{672} \\\\n')
    tex.write('\\bottomrule\n')
    tex.write('\\end{tabular}\n')
    tex.write('\\end{table}\n')
