"""
QUESTÃO 5 — Correlação e Regressão: Gasto em Educação x IDEB
Municípios: Vitória da Conquista e Ilhéus (BA)
Período: 2014–2023 | Valores reais deflacionados pelo IPCA (base 2023)
Método: Regressão Linear Simples por MQO (scipy.stats.linregress)
"""

import pandas as pd
import numpy as np
from scipy import stats

# ── 1. LEITURA DOS DADOS ──────────────────────────────────────────────────────
all_sheets = pd.read_excel('Dados_Principais.xlsx', sheet_name=None)
m = all_sheets['M']

cols = ['ano','rec_corr','rec_corr_ipca','rec_trib','rec_trib_ipca','desp','desp_ipca',
        'saude','saude_ipca','educ','educ_ipca','rec_cap','rec_cap_ipca',
        'transf','transf_ipca','iptu','iptu_ipca','issqn','issqn_ipca',
        'fator_ipca','ideb_vtc','ideb_ba','populacao','pib','pib_ipca']

vtc = m.iloc[1:11].copy();  vtc.columns = cols   # Vitória da Conquista
ilh = m.iloc[14:24].copy(); ilh.columns = cols   # Ilhéus

# ── 2. CONVERSÃO PARA NUMÉRICO ────────────────────────────────────────────────
for df in [vtc, ilh]:
    df['educ_ipca']  = pd.to_numeric(df['educ_ipca'],  errors='coerce')
    df['saude_ipca'] = pd.to_numeric(df['saude_ipca'], errors='coerce')
    df['ideb_vtc']   = pd.to_numeric(df['ideb_vtc'],   errors='coerce')
    df['ano']        = pd.to_numeric(df['ano'],        errors='coerce').astype(int)

# ── 3. REMOVER NaN ────────────────────────────────────────────────────────────
vtc_c = vtc.dropna(subset=['educ_ipca', 'ideb_vtc'])
ilh_c = ilh.dropna(subset=['educ_ipca', 'ideb_vtc'])

# ── 4. VARIÁVEIS (gasto em R$ milhões para facilitar interpretação) ───────────
x_vtc = vtc_c['educ_ipca'] / 1e6   # Gasto educação VTC (R$ Mi reais)
y_vtc = vtc_c['ideb_vtc']           # IDEB VTC

x_ilh = ilh_c['educ_ipca'] / 1e6   # Gasto educação Ilhéus (R$ Mi reais)
y_ilh = ilh_c['ideb_vtc']           # IDEB Ilhéus

# ── 5. REGRESSÃO LINEAR SIMPLES (MQO) ─────────────────────────────────────────
# Retorna: slope (β1), intercept (β0), r, p-valor, erro padrão
sl_v, int_v, r_v, p_v, se_v = stats.linregress(x_vtc, y_vtc)
sl_i, int_i, r_i, p_i, se_i = stats.linregress(x_ilh, y_ilh)

# ── 6. RESULTADOS ─────────────────────────────────────────────────────────────
print("=" * 60)
print("VITÓRIA DA CONQUISTA — Gasto Educação x IDEB")
print("=" * 60)
print(f"β0 (intercepto) : {int_v:.4f}")
print(f"β1 (coef. gasto): {sl_v:.6f}")
print(f"Equação         : IDEB = {int_v:.4f} + {sl_v:.6f} × Gasto(R$Mi)")
print(f"r (correlação)  : {r_v:.4f}")
print(f"r² (det.)       : {r_v**2:.4f}  →  {r_v**2*100:.1f}% da variância explicada")
print(f"p-valor         : {p_v:.4f}  →  {'SIGNIFICATIVO (p < 0,05)' if p_v < 0.05 else 'NÃO significativo (p ≥ 0,05)'}")
print(f"Erro padrão β1  : {se_v:.6f}")
print(f"N observações   : {len(vtc_c)}")

print()
print("=" * 60)
print("ILHÉUS — Gasto Educação x IDEB")
print("=" * 60)
print(f"β0 (intercepto) : {int_i:.4f}")
print(f"β1 (coef. gasto): {sl_i:.6f}")
print(f"Equação         : IDEB = {int_i:.4f} + {sl_i:.6f} × Gasto(R$Mi)")
print(f"r (correlação)  : {r_i:.4f}")
print(f"r² (det.)       : {r_i**2:.4f}  →  {r_i**2*100:.1f}% da variância explicada")
print(f"p-valor         : {p_i:.4f}  →  {'SIGNIFICATIVO (p < 0,05)' if p_i < 0.05 else 'NÃO significativo (p ≥ 0,05)'}")
print(f"Erro padrão β1  : {se_i:.6f}")
print(f"N observações   : {len(ilh_c)}")

print()
print("=" * 60)
print("DADOS UTILIZADOS")
print("=" * 60)
print("\nVTC:")
print(vtc_c[['ano','educ_ipca','ideb_vtc']].rename(
    columns={'educ_ipca':'Gasto_Educ_Real','ideb_vtc':'IDEB'}).to_string(index=False))
print("\nIlhéus:")
print(ilh_c[['ano','educ_ipca','ideb_vtc']].rename(
    columns={'educ_ipca':'Gasto_Educ_Real','ideb_vtc':'IDEB'}).to_string(index=False))

# ── 7. VALORES PREVISTOS (Ŷ) ──────────────────────────────────────────────────
vtc_c = vtc_c.copy()
ilh_c = ilh_c.copy()
vtc_c['IDEB_previsto'] = int_v + sl_v * x_vtc
ilh_c['IDEB_previsto'] = int_i + sl_i * x_ilh

print()
print("=" * 60)
print("VALORES PREVISTOS vs OBSERVADOS")
print("=" * 60)
print("\nVTC:")
for _, r in vtc_c.iterrows():
    res = r['ideb_vtc'] - (int_v + sl_v * r['educ_ipca']/1e6)
    print(f"  {r['ano']}: Obs={r['ideb_vtc']} | Prev={int_v + sl_v*r['educ_ipca']/1e6:.3f} | Resíduo={res:.3f}")

print("\nIlhéus:")
for _, r in ilh_c.iterrows():
    res = r['ideb_vtc'] - (int_i + sl_i * r['educ_ipca']/1e6)
    print(f"  {r['ano']}: Obs={r['ideb_vtc']} | Prev={int_i + sl_i*r['educ_ipca']/1e6:.3f} | Resíduo={res:.3f}")
