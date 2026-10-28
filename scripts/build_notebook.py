import os, nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as M, new_code_cell as C
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
nb = new_notebook()
nb.cells = [
M("# EDA - RAG-Based Document Q&A System (Project 72)\n**HCLTech Career Shaper Programme 2025-26** | Domain: LLM / Python\n\nThis notebook explores `data/dataset.csv` (5,000 chunk-level records: 500 original + 4,500 synthetic rows) and then checks the **real** retrieval experiment run on the 50-document corpus. It contains 8 visualisations (the brief asks for 3+).\n\n**How to run:** open from the project folder (`jupyter notebook`) and choose *Run All*. Figures are saved to `figures/`."),
C("""import os, warnings, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
warnings.filterwarnings('ignore'); sns.set_theme(style='whitegrid', context='notebook')
ROOT = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.path.abspath('.')
FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
df = pd.read_csv(os.path.join(ROOT, 'data', 'dataset_features.csv'))
def save(name): plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=140); plt.show()
print(df.shape)
df.head()"""),
M("## 1. Data quality checks"),
C("""print(json.dumps(json.load(open(os.path.join(ROOT,'data','cleaning_report.json'))), indent=2))
print('\\nMissing values per column:'); print(df.isna().sum()[df.isna().sum()>0] if df.isna().sum().sum() else 'none')
df[['word_count','retrieval_score','answer_quality_score','combined_score']].describe().round(3)"""),
M("**Reading:** no missing values or duplicates. The raw file had every `doc_id` mapped to 3-5 different `doc_type` values, which is impossible for one real file, so `doc_type` was aligned with the corpus manifest during cleaning (see report, section 5)."),
M("## 2. Distributions of the numeric columns (Figure 1)"),
C("""fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, c, col in zip(ax, ['word_count', 'retrieval_score', 'answer_quality_score'], ['#4C72B0', '#55A868', '#C44E52']):
    sns.histplot(df[c], bins=30, kde=True, ax=a, color=col); a.axvline(df[c].mean(), color='k', ls='--', lw=1); a.set_title(f'{c} (mean={df[c].mean():.2f})')
save('fig1_distributions.png')"""),
M("**Reading:** all three columns are almost flat (uniform): chunk length 50-499 words, retrieval score 0.50-1.00, answer quality 0.60-1.00. Real data would usually be skewed (many good scores, few bad)."),
M("## 3. Categorical balance (Figure 2)"),
C("""fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for a, c in zip(ax, ['doc_type', 'query_type', 'category']):
    order = df[c].value_counts().index; sns.countplot(data=df, x=c, order=order, ax=a, palette='viridis')
    a.set_title(f'Records per {c}'); a.tick_params(axis='x', rotation=20)
    for p in a.patches: a.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()), ha='center', va='bottom', fontsize=8)
save('fig2_categories.png')"""),
M("## 4. Retrieval score and answer quality by question type (Figure 3)"),
C("""fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
sns.boxplot(data=df, x='query_type', y='retrieval_score', ax=ax[0], palette='Set2'); ax[0].set_title('Retrieval score by query type')
sns.boxplot(data=df, x='query_type', y='answer_quality_score', ax=ax[1], palette='Set2'); ax[1].set_title('Answer quality by query type')
save('fig3_by_query_type.png')
for c in ['retrieval_score', 'answer_quality_score']:
    H, p = stats.kruskal(*[g[c] for _, g in df.groupby('query_type')]); print(f'Kruskal-Wallis {c} ~ query_type: H={H:.2f}, p={p:.3f}')"""),
M("**Reading:** medians are nearly equal for all five question types and the Kruskal-Wallis tests are not significant (p > 0.05), so in this dataset no question type is clearly harder than another."),
M("## 5. Correlation (Figure 4)"),
C("""num = ['word_count', 'retrieval_score', 'answer_quality_score']
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
sns.heatmap(df[num].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax[0]); ax[0].set_title('Correlation matrix')
hb = ax[1].hexbin(df.retrieval_score, df.answer_quality_score, gridsize=25, cmap='Blues'); plt.colorbar(hb, ax=ax[1]); ax[1].set_xlabel('retrieval_score'); ax[1].set_ylabel('answer_quality_score'); ax[1].set_title('Retrieval vs answer quality')
save('fig4_correlation.png')
print(stats.spearmanr(df.retrieval_score, df.answer_quality_score))"""),
M("**Reading:** correlations are about 0. In a real RAG system better retrieval should give better answers, so this tells us the supplied/expanded dataset is **synthetic** (independent random columns). It is useful for practising EDA, but conclusions about real system behaviour must come from the real experiment in section 9."),
M("## 6. Records that need human review (Figure 5)"),
C("""risk = df.groupby('category').hallucination_risk.mean().mul(100)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
sns.barplot(x=risk.index, y=risk.values, ax=ax[0], palette='Reds_d'); ax[0].set_ylabel('% records at risk'); ax[0].set_title('Hallucination-risk rate by corpus category')
r2 = df.groupby('doc_type').hallucination_risk.mean().mul(100).sort_values(); sns.barplot(x=r2.index, y=r2.values, ax=ax[1], palette='Oranges_d'); ax[1].set_title('...by document format'); ax[1].set_ylabel('% records at risk')
save('fig5_risk.png'); print(f'Overall risk rate: {df.hallucination_risk.mean()*100:.1f}%  (rule: retrieval_score < 0.6 or answer_quality_score < 0.65)')"""),
M("## 7. Outlier check (Figure 6)"),
C("""def iqr_out(s): q1, q3 = s.quantile([.25, .75]); i = q3 - q1; return ((s < q1-1.5*i) | (s > q3+1.5*i)).sum()
print({c: int(iqr_out(df[c])) for c in ['word_count', 'retrieval_score', 'answer_quality_score']}, '<- IQR outliers per column')
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for a, c in zip(ax, ['word_count', 'retrieval_score', 'answer_quality_score']): sns.boxplot(x=df[c], ax=a, color='#8172B2')
save('fig6_outliers.png')"""),
M("**Reading:** zero IQR outliers, consistent with bounded uniform data. Values stay inside valid ranges (scores within 0-1)."),
M("## 8. Per-document health (Figure 7)"),
C("""d = df.groupby('doc_id').agg(chunks=('chunk_id', 'count'), retrieval=('retrieval_score', 'mean'), quality=('answer_quality_score', 'mean'), combined=('combined_score', 'mean')).sort_values('combined')
fig, ax = plt.subplots(figsize=(14, 4)); ax.bar(d.index, d.combined, color='#4C72B0'); ax.set_ylim(0.7, 0.85); ax.tick_params(axis='x', rotation=90, labelsize=7)
ax.axhline(d.combined.mean(), color='r', ls='--'); ax.set_title('Mean combined score per document (red = overall mean)'); save('fig7_per_document.png')
d.head(5).round(3)"""),
M("## 9. REAL experiment: retrieval on the 50-document corpus (Figure 8)\nThese numbers come from `scripts/evaluate.py` and `scripts/ablation.py`, run on the 1,000 Q&A pairs in `data/qa_pairs.csv` (hashing embeddings, offline). Run both scripts first, and re-run with `--embedder minilm` for stronger results."),
C("""ab = pd.read_csv(os.path.join(ROOT, 'data', 'ablation_results.csv'))
ev = pd.read_csv(os.path.join(ROOT, 'data', 'eval_results.csv'))
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
hard = ab[ab.subset == 'hard_no_title']
for cs, g in hard.groupby('chunk_size'): ax[0].plot(g.k, g.doc_hit, marker='o', label=f'chunk {cs}')
ax[0].set_title('Doc hit@k (questions without the title)'); ax[0].set_xlabel('top-k'); ax[0].set_ylabel('doc hit rate'); ax[0].legend()
for cs, g in hard.groupby('chunk_size'): ax[1].plot(g.k, g.evidence_recall, marker='s', label=f'chunk {cs}')
ax[1].set_title('Evidence recall@k (answer sentence retrieved)'); ax[1].set_xlabel('top-k'); ax[1].legend()
ev['band'] = pd.cut(ev.top_score, [0, .15, .25, .35, .5, 1]); b = ev.groupby('band', observed=True).evidence_hit.agg(['mean', 'count']).reset_index()
ax[2].bar(b.band.astype(str), b['mean'], color='#55A868'); ax[2].set_title('Evidence found vs retrieval-score band'); ax[2].set_ylabel('evidence recall'); ax[2].set_xlabel('top-1 retrieval score')
for i, r in b.iterrows(): ax[2].text(i, r['mean'] + .01, f"n={r['count']}", ha='center', fontsize=8)
save('fig8_real_retrieval.png'); print(b.round(3))"""),
M("**Reading:** (a) larger chunks (800 characters) find the answer sentence more often, and going from top-k 3 to 5 adds only about 1-2 points of document hit rate; (b) when the best retrieval score is very low (0.15 or below) the evidence is found far less often (about 79%) than when it is above 0.15 (about 90-94%). The relationship is a threshold effect, not a smooth rise, and it is **absent in the synthetic dataset** - this supports the app's *minimum retrieval score* refusal rule."),
M("## 10. Summary of findings\n1. Dataset is clean (no missing values or duplicates) after aligning `doc_type` to the real file format of each document.\n2. All numeric columns are uniformly distributed and uncorrelated -> synthetic; no question type or file format is significantly harder.\n3. About 30% of records meet the *hallucination-risk* rule and would need human review - another reason to always show the source chunk.\n4. On the real corpus, retrieval works well (see section 9), and a very low retrieval score is a useful signal for deciding when to refuse to answer."),
]
nbf.write(nb, os.path.join(ROOT, "notebooks", "EDA.ipynb"))
