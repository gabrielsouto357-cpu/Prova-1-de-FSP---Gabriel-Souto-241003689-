import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LEITURA E ESTRUTURAÇÃO DOS DADOS
# ─────────────────────────────────────────────
xl = pd.read_excel('/mnt/user-data/uploads/Dados_Principais.xlsx', sheet_name='Municípios', header=None)

def extract_municipality(df, name_row, data_start_row, data_end_row, city_name):
    rows = df.iloc[data_start_row:data_end_row].copy()
    rows.columns = ['Ano', 'Rec_Correntes', 'Rec_Correntes_IPCA',
                    'Rec_Tributaria', 'Rec_Tributaria_IPCA',
                    'Despesas', 'Despesas_IPCA',
                    'Saude', 'Saude_IPCA',
                    'Educacao', 'Educacao_IPCA',
                    'Rec_Capital', 'Rec_Capital_IPCA',
                    'Transf_Correntes', 'Transf_Correntes_IPCA',
                    'IPTU', 'IPTU_IPCA',
                    'ISS', 'ISS_IPCA',
                    'Fator_IPCA', 'IDEB_Mun', 'IDEB_BA',
                    'Populacao', 'PIB_Municipal', 'PIB_Municipal_IPCA']
    rows = rows[pd.to_numeric(rows['Ano'], errors='coerce').notna()].copy()
    rows['Ano'] = rows['Ano'].astype(int)
    rows['Municipio'] = city_name
    # Converter colunas numéricas
    num_cols = [c for c in rows.columns if c not in ['Ano', 'Municipio', 'Populacao', 'IDEB_Mun', 'IDEB_BA']]
    for c in num_cols:
        rows[c] = pd.to_numeric(rows[c], errors='coerce')
    return rows

vca = extract_municipality(xl, 0, 2, 12, 'Vitória da Conquista')
ilheus = extract_municipality(xl, 13, 15, 25, 'Ilhéus')

df = pd.concat([vca, ilheus], ignore_index=True)

# ─────────────────────────────────────────────
# 2. CONSTRUÇÃO DAS VARIÁVEIS DA CURVA DE LAFFER
#    Proxy de Carga Tributária = Rec. Tributária / Rec. Correntes (%)
#    Receita fiscal normalizada = Rec. Tributária / PIB Municipal (%)
# ─────────────────────────────────────────────
df['Carga_Proxy_Pct'] = (df['Rec_Tributaria_IPCA'] / df['Rec_Correntes_IPCA']) * 100
df['Receita_PIB_Pct']  = (df['Rec_Tributaria_IPCA'] / df['PIB_Municipal_IPCA']) * 100

df_clean = df.dropna(subset=['Carga_Proxy_Pct', 'Receita_PIB_Pct']).copy()

# ─────────────────────────────────────────────
# 3. REGRESSÃO QUADRÁTICA (Curva de Laffer)
# ─────────────────────────────────────────────
x = df_clean['Carga_Proxy_Pct'].values
y = df_clean['Receita_PIB_Pct'].values

coeffs = np.polyfit(x, y, 2)   # ax² + bx + c
a, b, c = coeffs
poly = np.poly1d(coeffs)

x_fit = np.linspace(x.min() - 1, x.max() + 1, 300)
y_fit = poly(x_fit)

y_pred = poly(x)
r2 = r2_score(y, y_pred)

# Ponto de máximo (vértice da parábola)
x_max = -b / (2 * a)
y_max = poly(x_max)

# ─────────────────────────────────────────────
# 4. VISUALIZAÇÃO
# ─────────────────────────────────────────────
COLORS = {
    'Vitória da Conquista': '#1a6fad',
    'Ilhéus':               '#d44f2e',
    'curve':                '#2c9e52',
    'vertex':               '#8b1a1a',
    'grid':                 '#e5e5e5',
}

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

# Curva ajustada
ax.plot(x_fit, y_fit, color=COLORS['curve'], lw=2.5,
        label=f'Regressão quadrática\n$R^2$ = {r2:.3f}', zorder=2)

# Ponto de máximo
if x.min() <= x_max <= x.max():
    ax.axvline(x_max, color=COLORS['vertex'], lw=1.4, ls='--', alpha=0.7)
    ax.scatter([x_max], [y_max], color=COLORS['vertex'], zorder=5, s=90,
               label=f'Pico ótimo: carga ≈ {x_max:.1f}%')
    ax.annotate(f'  Pico: ({x_max:.1f}%, {y_max:.2f}%)',
                xy=(x_max, y_max), xytext=(x_max + 0.5, y_max + 0.05),
                fontsize=9, color=COLORS['vertex'])

# Pontos por município e ano
markers = {'Vitória da Conquista': 'o', 'Ilhéus': 's'}
for city, grp in df_clean.groupby('Municipio'):
    sc = ax.scatter(grp['Carga_Proxy_Pct'], grp['Receita_PIB_Pct'],
                    color=COLORS[city], marker=markers[city],
                    s=70, zorder=4, label=city, edgecolors='white', linewidths=0.5)
    for _, row in grp.iterrows():
        ax.annotate(str(row['Ano']),
                    xy=(row['Carga_Proxy_Pct'], row['Receita_PIB_Pct']),
                    xytext=(4, 3), textcoords='offset points',
                    fontsize=7.5, color=COLORS[city], alpha=0.9)

# Equação da regressão
eq_sign_b = '+' if b >= 0 else '-'
eq_sign_c = '+' if c >= 0 else '-'
eq_str = (f'$y = {a:.4f}x^2 {eq_sign_b} {abs(b):.4f}x {eq_sign_c} {abs(c):.4f}$')

ax.text(0.03, 0.96, eq_str, transform=ax.transAxes,
        fontsize=10, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8))

ax.set_xlabel('Proxy de Carga Tributária\n(Receita Tributária / Receita Corrente corrigida pelo IPCA, %)',
              fontsize=11)
ax.set_ylabel('Receita Tributária / PIB Municipal\n(ambos corrigidos pelo IPCA, %)',
              fontsize=11)
ax.set_title('Curva de Laffer Empírica\nVitória da Conquista e Ilhéus (2014–2023)',
             fontsize=14, fontweight='bold', pad=14)

ax.grid(True, color=COLORS['grid'], linewidth=0.8, zorder=0)
ax.legend(fontsize=9.5, loc='upper right', framealpha=0.9)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f%%'))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/laffer_curve.png', dpi=180, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 5. TABELA DE DADOS
# ─────────────────────────────────────────────
tabela = df_clean[['Municipio','Ano','Rec_Correntes_IPCA','Rec_Tributaria_IPCA',
                   'PIB_Municipal_IPCA','Carga_Proxy_Pct','Receita_PIB_Pct']].copy()
tabela.columns = ['Município','Ano','Rec. Correntes (IPCA)','Rec. Tributária (IPCA)',
                  'PIB Municipal (IPCA)','Carga Proxy (%)','Rec. Trib./PIB (%)']
tabela = tabela.sort_values(['Município','Ano'])

print("="*70)
print("CURVA DE LAFFER EMPÍRICA — VITÓRIA DA CONQUISTA E ILHÉUS (2014–2023)")
print("="*70)
print(f"\nEquação ajustada: {eq_str.replace('$','')}")
print(f"R² = {r2:.4f}")
if x.min() <= x_max <= x.max():
    print(f"Pico ótimo (vértice): carga proxy ≈ {x_max:.2f}%  →  Rec./PIB ≈ {y_max:.4f}%")
print("\nCoeficientes: a={:.6f}, b={:.6f}, c={:.6f}".format(a, b, c))
print("\n--- Dados utilizados ---")
pd.set_option('display.float_format', '{:,.2f}'.format)
print(tabela.to_string(index=False))
print("\nGráfico salvo em: /mnt/user-data/outputs/laffer_curve.png")
