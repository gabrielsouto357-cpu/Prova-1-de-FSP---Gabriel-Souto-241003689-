import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LEITURA E ESTRUTURAÇÃO
# ─────────────────────────────────────────────
xl = pd.read_excel('/mnt/user-data/uploads/Dados_Principais.xlsx',
                   sheet_name='Municípios', header=None)

def extract_municipality(df, data_start_row, data_end_row, city_name):
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
    num_cols = [c for c in rows.columns if c not in ['Ano', 'Municipio', 'Populacao', 'IDEB_Mun', 'IDEB_BA']]
    for c in num_cols:
        rows[c] = pd.to_numeric(rows[c], errors='coerce')
    return rows

vca    = extract_municipality(xl, 2, 12, 'Vitória da Conquista')
ilheus = extract_municipality(xl, 15, 25, 'Ilhéus')
df     = pd.concat([vca, ilheus], ignore_index=True)
df     = df.dropna(subset=['Rec_Correntes_IPCA', 'PIB_Municipal_IPCA'])

# Variáveis em milhões de R$ (IPCA)
df['Y'] = df['Rec_Correntes_IPCA'] / 1e6   # Receita Corrente
df['X'] = df['PIB_Municipal_IPCA']  / 1e6  # PIB Municipal

# ─────────────────────────────────────────────
# 2. REGRESSÃO OLS: Y = β0 + β1*X + ε
# ─────────────────────────────────────────────
x = df['X'].values
y = df['Y'].values
n = len(x)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r2       = r_value**2
y_pred   = intercept + slope * x
residuals = y - y_pred
sse      = np.sum(residuals**2)
sst      = np.sum((y - y.mean())**2)
mse      = sse / (n - 2)
rmse     = np.sqrt(mse)

# Erro padrão de β0
x_mean   = x.mean()
sxx      = np.sum((x - x_mean)**2)
se_b0    = np.sqrt(mse * (1/n + x_mean**2/sxx))
se_b1    = std_err

# t-stats e p-values
t_b1     = slope     / se_b1
t_b0     = intercept / se_b0
p_b1     = 2 * stats.t.sf(abs(t_b1), df=n-2)
p_b0     = 2 * stats.t.sf(abs(t_b0), df=n-2)

# IC 95%
t_crit   = stats.t.ppf(0.975, df=n-2)
ic_b1    = (slope     - t_crit*se_b1, slope     + t_crit*se_b1)
ic_b0    = (intercept - t_crit*se_b0, intercept + t_crit*se_b0)

# F-stat
f_stat   = (r2 / 1) / ((1 - r2) / (n - 2))
p_f      = stats.f.sf(f_stat, 1, n-2)

# ─────────────────────────────────────────────
# 3. IMPRESSÃO DOS RESULTADOS
# ─────────────────────────────────────────────
print("=" * 62)
print("REGRESSÃO OLS: Receita Corrente = β0 + β1·PIB + ε")
print("Municípios: Vitória da Conquista e Ilhéus | 2014–2023")
print("Valores deflacionados pelo IPCA (R$ milhões)")
print("=" * 62)
print(f"\n{'Parâmetro':<10} {'Estimativa':>14} {'Erro Padrão':>14} {'t':>8} {'p-valor':>10} {'IC 95%':>24}")
print("-" * 82)
print(f"{'β0':<10} {intercept:>14.4f} {se_b0:>14.4f} {t_b0:>8.3f} {p_b0:>10.4f}  ({ic_b0[0]:.4f}, {ic_b0[1]:.4f})")
print(f"{'β1':<10} {slope:>14.4f} {se_b1:>14.4f} {t_b1:>8.3f} {p_b1:>10.4f}  ({ic_b1[0]:.4f}, {ic_b1[1]:.4f})")
print("-" * 82)
print(f"\nR²      = {r2:.4f}")
print(f"R       = {r_value:.4f}")
print(f"RMSE    = {rmse:.4f} (R$ milhões)")
print(f"F-stat  = {f_stat:.4f}   p-valor F = {p_f:.6f}")
print(f"N       = {n}")

# ─────────────────────────────────────────────
# 4. GRÁFICO DE DISPERSÃO
# ─────────────────────────────────────────────
COLORS = {
    'Vitória da Conquista': '#1a6fad',
    'Ilhéus':               '#d44f2e',
    'reg':                  '#2c6e2e',
    'ic':                   '#a8d5a2',
    'grid':                 '#e5e5e5',
}

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

# Banda de confiança 95%
x_plot  = np.linspace(x.min() * 0.97, x.max() * 1.03, 300)
y_plot  = intercept + slope * x_plot
se_band = np.sqrt(mse * (1/n + (x_plot - x_mean)**2 / sxx))
y_upper = y_plot + t_crit * se_band
y_lower = y_plot - t_crit * se_band

ax.fill_between(x_plot, y_lower, y_upper, color=COLORS['ic'], alpha=0.45,
                label='IC 95% da reta')
ax.plot(x_plot, y_plot, color=COLORS['reg'], lw=2.4,
        label=f'OLS: $\\hat{{y}}$ = {intercept:.2f} + {slope:.4f}·PIB\n$R^2$ = {r2:.4f}  |  p(β₁) = {p_b1:.4f}')

markers = {'Vitória da Conquista': 'o', 'Ilhéus': 's'}
for city, grp in df.groupby('Municipio'):
    ax.scatter(grp['X'], grp['Y'],
               color=COLORS[city], marker=markers[city],
               s=72, zorder=5, label=city,
               edgecolors='white', linewidths=0.6)
    for _, row in grp.iterrows():
        ax.annotate(str(row['Ano']),
                    xy=(row['X'], row['Y']),
                    xytext=(4, 3), textcoords='offset points',
                    fontsize=7.5, color=COLORS[city], alpha=0.9)

# Equação e estatísticas no canto
box_text = (f"$\\hat{{y}}$ = {intercept:.2f} + {slope:.4f} · PIB\n"
            f"$R^2$ = {r2:.4f}    N = {n}\n"
            f"p(β₁) = {p_b1:.4f}    F = {f_stat:.2f}")
ax.text(0.03, 0.97, box_text, transform=ax.transAxes,
        fontsize=9.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.85))

ax.set_xlabel('PIB Municipal — preços constantes IPCA (R$ milhões)',  fontsize=11)
ax.set_ylabel('Receita Corrente — preços constantes IPCA (R$ milhões)', fontsize=11)
ax.set_title('Regressão: Receita Corrente = β₀ + β₁ · PIB + ε\n'
             'Vitória da Conquista e Ilhéus (2014–2023)',
             fontsize=13, fontweight='bold', pad=14)

ax.grid(True, color=COLORS['grid'], linewidth=0.8, zorder=0)
ax.legend(fontsize=9.5, loc='lower right', framealpha=0.9)

fmt = mticker.FuncFormatter(lambda val, _: f'R$ {val:,.0f} M')
ax.xaxis.set_major_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/regressao_receita_pib.png', dpi=180, bbox_inches='tight')
plt.close()
print("\nGráfico salvo em: /mnt/user-data/outputs/regressao_receita_pib.png")
