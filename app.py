# -*- coding: utf-8 -*-
"""
Atlas Dashboard – Streamlit Prototype v1 (Okiar)
------------------------------------------------

Objetivo: protótipo conceitual completo do War Room do Projeto Atlas em Streamlit.
Este app mostra a visão integrada de Clientes, Concorrência e Mercado com abas:
- Visão Geral, Clientes, Concorrência, Mercado, AI

Notas:
- Dados são simulados (mock) e paramétricos para demonstração (redbank, banco fictício)
- O app foi desenhado para ser auto-contido em um único arquivo.
- Estilo e componentes focados em clareza executiva

Como rodar:
    streamlit run app.py

Dependências sugeridas:
    streamlit>=1.33.0
    pandas>=2.2.0
    numpy>=1.26.0
    plotly>=5.22.0
    python-dateutil

"""

import math
import random
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO GERAL
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Atlas – War Room (Okiar)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta e helpers visuais
PRIMARY = "#5D6CFB"
ACCENT = "#00C2A8"
WARNING = "#FFB020"
DANGER = "#EB5757"
MUTED = "#9AA0A6"
LIGHT_BG = "#F7F8FA"

CUSTOM_CSS = f"""
/* Cards e métricas */
.metric-card {{
  background: white;
  border: 1px solid #EAECF0;
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 1px 2px rgba(16,24,40,.05);
}}
.metric-label {{ color: {MUTED}; font-size: 12px; font-weight: 500; }}
.metric-value {{ font-size: 28px; font-weight: 700; }}
.metric-delta-up {{ color: {ACCENT}; font-weight: 600; }}
.metric-delta-down {{ color: {DANGER}; font-weight: 600; }}

/* Títulos e seções */
.block-title {{ font-size: 16px; font-weight: 700; margin-bottom: 6px; }}
.subtle {{ color: {MUTED}; font-size: 12px; }}
.section {{ padding: 6px 0 2px 0; }}

/* Tabela compacta */
.small-table table {{ font-size: 12px; }}

/* Badges */
.badge {{
  display: inline-block; padding: 4px 8px; font-size: 11px;
  border-radius: 12px; background: {LIGHT_BG}; border: 1px solid #E5E7EB; color: #374151
}}
.badge-green {{ background: #ECFDF5; color: #047857; border-color: #A7F3D0; }}
.badge-yellow {{ background: #FFFBEB; color: #92400E; border-color: #FDE68A; }}
.badge-red {{ background: #FEF2F2; color: #991B1B; border-color: #FCA5A5; }}

/* Alertas */
.alert {{ border-left: 4px solid {ACCENT}; background: white; padding: 10px 12px; border-radius: 8px; border:1px solid #EAECF0; }}
.alert-critico {{ border-left-color: {DANGER}; }}
.alert-alto {{ border-left-color: {WARNING}; }}

/* Expander */
details > summary {{ cursor: pointer; }}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PARÂMETROS DE SIMULAÇÃO
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Players e contexto
BANK = "RedBank"  # cliente
PLAYERS_TRAD = ["Bradesco", "Itaú", "Santander", "Banco do Brasil"]
PLAYER_DIGITAL = "Nubank"
PLAYERS = [BANK] + PLAYERS_TRAD + [PLAYER_DIGITAL]

# Canais e regiões
CANAIS = ["Agência", "App", "Site", "Correspondente", "Parceiros"]
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
UF = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"
]

# Produtos de Cartão e Investimentos
CARD_TYPES = ["Classic", "Gold", "Platinum", "Black", "PJ"]
INV_PRODUCTS = ["CDB", "LCI", "LCA", "Fundos", "Previdência"]

# Segmentos/Personas (simplificado)
PERSONAS = [
    "A1", "A2", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2"
]

# Datas (12 meses)
HOJE = date.today()
MESES = [HOJE - relativedelta(months=i) for i in range(0, 12)][::-1]
TRIS = sorted({(d.year, (d.month - 1)//3 + 1) for d in MESES})

# -----------------------------------------------------------------------------
# GERADORES DE DADOS MOCK
# -----------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def gen_market_share_series(players: List[str], months: List[date]) -> pd.DataFrame:
    """Série mensal de market share (estimado) por player."""
    data = []
    base_weights = {p: random.uniform(5, 35) for p in players}
    # Ajuste para RedBank aparecer competitivo
    base_weights[BANK] = random.uniform(18, 28)

    for m in months:
        # oscilações
        noise = {p: np.clip(np.random.normal(0, 1.2), -3, 3) for p in players}
        total = sum(base_weights[p] + noise[p] for p in players)
        for p in players:
            share = (base_weights[p] + noise[p]) / total
            data.append({"mes": m, "player": p, "market_share": max(0.02, share)})
    df = pd.DataFrame(data)
    # normaliza por mês para somar 1
    df["sum_mes"] = df.groupby("mes")["market_share"].transform("sum")
    df["market_share"] = df["market_share"] / df["sum_mes"]
    df.drop(columns=["sum_mes"], inplace=True)
    return df


def gen_brand_funnel(players: List[str], tris: List[Tuple[int, int]]) -> pd.DataFrame:
    """Funil de marca trimestral: memória (awareness), consideração, preferência."""
    rows = []
    for (y, q) in tris:
        for p in players:
            base = random.uniform(0.5, 0.9) if p != PLAYER_DIGITAL else random.uniform(0.6, 0.95)
            awareness = np.clip(base + np.random.normal(0, 0.05), 0.4, 0.98)
            consideration = np.clip(awareness - random.uniform(0.1, 0.25), 0.1, awareness - 0.05)
            preference = np.clip(consideration - random.uniform(0.05, 0.15), 0.05, consideration - 0.02)
            rows.append({
                "ano": y, "tri": q, "player": p,
                "awareness": awareness,
                "consideration": consideration,
                "preference": preference,
            })
    return pd.DataFrame(rows)


def gen_channel_region_perf(players: List[str], canais: List[str], ufs: List[str]) -> pd.DataFrame:
    """KPIs simplificados por canal e UF: visitas, conversões e taxa."""
    data = []
    for p in players:
        for uf in ufs:
            for c in canais:
                visitas = max(200, int(np.random.normal(2500, 800)))
                conv = int(visitas * np.clip(np.random.beta(2, 12), 0.01, 0.25))
                data.append({
                    "player": p, "uf": uf, "canal": c,
                    "visitas": visitas, "conversoes": conv,
                    "taxa_conv": conv / visitas
                })
    return pd.DataFrame(data)


def gen_card_competitiveness(players: List[str], types: List[str], tris: List[Tuple[int, int]]) -> pd.DataFrame:
    """Indicadores de cartões por tipo: base ativa, principalidade, NPS (simulado)."""
    rows = []
    for (y, q) in tris:
        for p in players:
            for t in types:
                base_ativa = int(max(10000, np.random.normal(120000, 30000)))
                principalidade = np.clip(np.random.normal(0.38 if p==BANK else 0.34, 0.05), 0.15, 0.65)
                nps = int(np.clip(np.random.normal(42 if p==BANK else 36, 8), -20, 80))
                rows.append({
                    "ano": y, "tri": q, "player": p, "tipo": t,
                    "base_ativa": base_ativa,
                    "principalidade": principalidade,
                    "nps": nps,
                })
    return pd.DataFrame(rows)


def gen_investments_comp(players: List[str], products: List[str], tris: List[Tuple[int, int]]) -> pd.DataFrame:
    """Competitividade em investimentos: captação líquida e ticket médio (simulado)."""
    rows = []
    for (y, q) in tris:
        for p in players:
            for prod in products:
                captacao = int(max(1e6, np.random.normal(5e6 if p==BANK else 4e6, 1.5e6)))
                ticket = float(np.clip(np.random.normal(4200 if p==BANK else 3800, 900), 1200, 12000))
                taxa_fee = float(np.clip(np.random.normal(0.012, 0.004), 0.002, 0.03))
                rows.append({
                    "ano": y, "tri": q, "player": p, "produto": prod,
                    "captacao_liquida": captacao, "ticket_medio": ticket, "taxa_fee": taxa_fee
                })
    return pd.DataFrame(rows)


def gen_habits_tracking(personas: List[str], tris: List[Tuple[int, int]]) -> pd.DataFrame:
    """Tracking trimestral de hábitos & atitudes por persona (índices 0-100)."""
    rows = []
    for (y, q) in tris:
        for per in personas:
            digital = int(np.clip(np.random.normal(68, 12), 20, 95))
            price_sens = int(np.clip(np.random.normal(58, 15), 10, 95))
            risk_av = int(np.clip(np.random.normal(62, 10), 10, 95))
            trust_bank = int(np.clip(np.random.normal(64, 12), 10, 95))
            rows.append({
                "ano": y, "tri": q, "persona": per,
                "afinidade_digital": digital,
                "sensibilidade_preco": price_sens,
                "aversao_risco": risk_av,
                "confianca_banco": trust_bank,
            })
    return pd.DataFrame(rows)


def gen_events_competitive(players: List[str], months: List[date]) -> pd.DataFrame:
    """Eventos competitivos categorizados por severidade/impacto."""
    tipos = ["Lançamento", "Preço", "Campanha", "Parceria", "M&A", "Regulatório"]
    severidades = ["Baixo", "Médio", "Alto", "Crítico"]
    rows = []
    for m in months:
        for _ in range(random.randint(2, 7)):
            p = random.choice([pp for pp in players if pp != BANK])
            t = random.choice(tipos)
            sev = random.choices(severidades, weights=[5, 6, 3, 1])[0]
            desc = f"{t} do {p} em {m.strftime('%b/%Y')}"
            score = {"Baixo": 10, "Médio": 30, "Alto": 70, "Crítico": 90}[sev]
            rows.append({
                "data": m + timedelta(days=random.randint(0, 27)),
                "player": p, "tipo": t, "severidade": sev, "score": score,
                "descricao": desc
            })
    return pd.DataFrame(rows)


def gen_sov_series(players: List[str], months: List[date]) -> pd.DataFrame:
    """Share of Voice simulado (social/search)."""
    rows = []
    for m in months:
        total = 0
        vals = {}
        for p in players:
            val = max(1, int(np.random.normal(120 if p==PLAYER_DIGITAL else 80, 30)))
            vals[p] = val
            total += val
        for p, v in vals.items():
            rows.append({"mes": m, "player": p, "sov": v/total})
    return pd.DataFrame(rows)


def gen_inventory_docs() -> pd.DataFrame:
    """Inventário de estudos/dossiês (metadados)."""
    docs = []
    bases = [
        ("Tracking Competitivo – Q2", "Concorrência", "pdf", "2025-07-22"),
        ("BRAIN – Marca – Onda Jul", "Concorrência", "pptx", "2025-08-01"),
        ("MERIDIO – Segmentação v1", "Clientes", "pdf", "2025-06-15"),
        ("Benchmark Financeiro T2", "Concorrência", "xlsx", "2025-08-05"),
        ("Radar de Tendências – Jul", "Mercado", "pdf", "2025-07-30"),
    ]
    for t, aba, ext, dt in bases:
        docs.append({"titulo": t, "aba": aba, "tipo": ext.upper(), "data": pd.to_datetime(dt)})
    return pd.DataFrame(docs)


# -----------------------------------------------------------------------------
# CRIAÇÃO DOS DATASETS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df_ms = gen_market_share_series(PLAYERS, MESES)
    df_funnel = gen_brand_funnel(PLAYERS, TRIS)
    df_ch = gen_channel_region_perf(PLAYERS, CANAIS, UF)
    df_card = gen_card_competitiveness(PLAYERS, CARD_TYPES, TRIS)
    df_inv = gen_investments_comp(PLAYERS, INV_PRODUCTS, TRIS)
    df_hab = gen_habits_tracking(PERSONAS, TRIS)
    df_evt = gen_events_competitive(PLAYERS, MESES)
    df_sov = gen_sov_series(PLAYERS, MESES)
    df_docs = gen_inventory_docs()
    return {
        "market_share": df_ms,
        "funnel": df_funnel,
        "channel": df_ch,
        "card": df_card,
        "inv": df_inv,
        "habits": df_hab,
        "events": df_evt,
        "sov": df_sov,
        "docs": df_docs,
    }

data = load_data()

# -----------------------------------------------------------------------------
# HELPERS DE UI
# -----------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: float | None = None):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-up" if delta >= 0 else "metric-delta-down"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f"<div class='{cls}'>{arrow} {delta:.1f}%</div>"
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pct(x):
    return f"{x*100:.1f}%"


def line_pct(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    fig = px.line(df, x=x, y=y, color=color, markers=True)
    fig.update_layout(
        title=title, height=320, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    if y.endswith("share") or y in ("awareness", "consideration", "preference", "sov"):
        fig.update_yaxes(tickformat=",.0%")
    return fig


def bars_pct(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", text_auto=".0%")
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def bars_num(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", text_auto=True)
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def area_pct(df: pd.DataFrame, x: str, y: str, color: str, title: str):
    fig = px.area(df, x=x, y=y, color=color)
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(tickformat=",.0%")
    return fig


def alert_box(titulo: str, texto: str, nivel: str = "normal"):
    cls = "alert"
    if nivel == "critico":
        cls += " alert-critico"
    elif nivel == "alto":
        cls += " alert-alto"
    st.markdown(f"<div class='{cls}'><b>{titulo}</b><br/>{texto}</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR – FILTROS GLOBAIS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.write("## ⚙️ Filtros Globais")
    players_sel = st.multiselect("Players", PLAYERS, default=[BANK, "Bradesco", "Itaú", PLAYER_DIGITAL])
    tri_sel = st.selectbox("Trimestre", options=[f"{y}-T{q}" for (y, q) in TRIS], index=len(TRIS)-1)
    ano_tri = tuple(map(int, tri_sel.replace("-T", " ").split()))
    mes_sel = st.slider("Janela de meses (histórico)", min_value=3, max_value=12, value=12)

    st.divider()
    st.caption("Versão: v1 • Dados simulados")

# Subconjuntos por filtros
ms = data["market_share"][data["market_share"]["player"].isin(players_sel)].copy()
ms_hist = ms[ms["mes"].isin(MESES[-mes_sel:])]

funnel = data["funnel"][data["funnel"]["player"].isin(players_sel)].copy()
cur_tri = funnel[(funnel["ano"]==ano_tri[0]) & (funnel["tri"]==ano_tri[1])]

sov = data["sov"][data["sov"]["player"].isin(players_sel) & data["sov"]["mes"].isin(MESES[-mes_sel:])]

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.title("Atlas – War Room")
st.caption("Projeto Atlas (Okiar): visão integrada de Clientes, Concorrência e Mercado")


# -----------------------------------------------------------------------------
# ABAS PRINCIPAIS
# -----------------------------------------------------------------------------
tab_overview, tab_clientes, tab_comp, tab_mercado, tab_ai = st.tabs([
    "Visão Geral", "Clientes", "Concorrência", "Mercado", "AI"
])

# -----------------------------------------------------------------------------
# VISÃO GERAL
# -----------------------------------------------------------------------------
with tab_overview:
    st.subheader("Resumo Executivo")

    col1, col2, col3, col4 = st.columns(4)
    # KPIs simulados
    ultimo_ms_rb = ms_hist[(ms_hist["player"]==BANK) & (ms_hist["mes"]==ms_hist["mes"].max())]["market_share"].mean()
    prev_ms_rb = ms_hist[(ms_hist["player"]==BANK) & (ms_hist["mes"]==ms_hist["mes"].unique()[-2])]["market_share"].mean()
    delta_ms = (ultimo_ms_rb - prev_ms_rb) / max(prev_ms_rb, 1e-9) * 100

    cur_aw = float(cur_tri[cur_tri["player"]==BANK]["awareness"].mean())
    cur_cn = float(cur_tri[cur_tri["player"]==BANK]["consideration"].mean())
    cur_pf = float(cur_tri[cur_tri["player"]==BANK]["preference"].mean())

    col1.markdown("### RedBank")
    with col1:
        metric_card("Market Share (mês)", pct(ultimo_ms_rb), delta_ms)
    with col2:
        metric_card("Awareness (tri)", pct(cur_aw))
    with col3:
        metric_card("Consideração (tri)", pct(cur_cn))
    with col4:
        metric_card("Preferência (tri)", pct(cur_pf))

    st.divider()

    c1, c2 = st.columns((1,1))
    with c1:
        st.markdown("<div class='block-title'>Market share – histórico (últimos meses)</div>", unsafe_allow_html=True)
        fig_ms = line_pct(ms_hist, x="mes", y="market_share", color="player", title="")
        st.plotly_chart(fig_ms, use_container_width=True)

    with c2:
        st.markdown("<div class='block-title'>Share of Voice – histórico</div>", unsafe_allow_html=True)
        fig_sov = area_pct(sov, x="mes", y="sov", color="player", title="")
        st.plotly_chart(fig_sov, use_container_width=True)

    st.divider()

    st.markdown("<div class='block-title'>Funil de marca (trimestre selecionado)</div>", unsafe_allow_html=True)
    ftri = cur_tri.copy()
    ftri = ftri.melt(id_vars=["ano", "tri", "player"], value_vars=["awareness", "consideration", "preference"], var_name="etapa", value_name="valor")
    fig_funil = bars_pct(ftri, x="etapa", y="valor", color="player", title="")
    st.plotly_chart(fig_funil, use_container_width=True)

    st.divider()

    st.markdown("<div class='block-title'>Alertas Recentes</div>", unsafe_allow_html=True)
    evts = data["events"].sort_values("data", ascending=False).head(6)
    for _, r in evts.iterrows():
        nivel = "critico" if r["severidade"]=="Crítico" else ("alto" if r["severidade"]=="Alto" else "normal")
        alert_box(
            f"{r['tipo']} – {r['player']} ({r['data'].strftime('%d/%m/%Y')})",
            r["descricao"], nivel=nivel
        )

    with st.expander("Inventário de estudos (últimos)"):
        st.dataframe(data["docs"].sort_values("data", ascending=False), use_container_width=True, height=180)


# -----------------------------------------------------------------------------
# CLIENTES
# -----------------------------------------------------------------------------
with tab_clientes:
    st.subheader("Clientes – Segmentos, Jornada e Performance")

    # Filtros locais
    c_a, c_b, c_c = st.columns((1,1,1))
    with c_a:
        per_sel = st.multiselect("Personas", PERSONAS, default=["A1", "B1", "C2", "D1"])
    with c_b:
        canais_sel = st.multiselect("Canais", CANAIS, default=CANAIS)
    with c_c:
        uf_sel = st.multiselect("UF", UF, default=["SP", "RJ", "MG", "PR", "RS", "BA", "PE"])  # foco nas maiores

    # Habits & attitudes – índices
    hab = data["habits"][data["habits"]["persona"].isin(per_sel)].copy()
    hab_tri = hab[(hab["ano"]==ano_tri[0]) & (hab["tri"]==ano_tri[1])]

    cc1, cc2 = st.columns((1,1))
    with cc1:
        st.markdown("<div class='block-title'>Índices por persona (tri selecionado)</div>", unsafe_allow_html=True)
        htri = hab_tri.melt(id_vars=["ano", "tri", "persona"], value_vars=["afinidade_digital", "sensibilidade_preco", "aversao_risco", "confianca_banco"], var_name="indice", value_name="valor")
        fig_h = px.bar(htri, x="valor", y="persona", color="indice", orientation="h", barmode="group", text_auto=True)
        fig_h.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_h, use_container_width=True)

    with cc2:
        st.markdown("<div class='block-title'>Evolução dos índices (personas selecionadas)</div>", unsafe_allow_html=True)
        hts = hab[hab["persona"].isin(per_sel)].copy()
        hts["periodo"] = hts.apply(lambda r: f"{int(r['ano'])}-T{int(r['tri'])}", axis=1)
        fig_hts = px.line(hts.melt(id_vars=["persona", "periodo"], value_vars=["afinidade_digital", "sensibilidade_preco", "aversao_risco", "confianca_banco"], var_name="indice", value_name="valor"),
                          x="periodo", y="valor", color="indice", line_group="persona", markers=True)
        fig_hts.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_hts, use_container_width=True)

    st.divider()

    st.markdown("<div class='block-title'>Performance por canal e UF</div>", unsafe_allow_html=True)
    ch = data["channel"][
        (data["channel"]["player"]==BANK) &
        (data["channel"]["canal"].isin(canais_sel)) &
        (data["channel"]["uf"].isin(uf_sel))
    ].copy()

    ch_ag = ch.groupby(["canal", "uf"], as_index=False).agg({"visitas":"sum", "conversoes":"sum"})
    ch_ag["taxa_conv"] = ch_ag["conversoes"] / ch_ag["visitas"]

    c1, c2 = st.columns((1,1))
    with c1:
        fig_c1 = px.treemap(ch_ag, path=["canal", "uf"], values="visitas", color="taxa_conv", color_continuous_scale="Tealgrn", title="Volume de visitas (tamanho) e taxa de conversão (cor)")
        fig_c1.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_c1, use_container_width=True)
    with c2:
        top_uf = ch_ag.sort_values("conversoes", ascending=False).head(15)
        fig_c2 = px.bar(top_uf, x="conversoes", y="uf", color="canal", orientation="h", text_auto=True, title="Top UFs por conversões")
        fig_c2.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_c2, use_container_width=True)

    with st.expander("Tabela detalhada"):
        st.dataframe(ch_ag.sort_values(["canal", "conversoes"], ascending=[True, False]), use_container_width=True, height=240)


# -----------------------------------------------------------------------------
# CONCORRÊNCIA
# -----------------------------------------------------------------------------
with tab_comp:
    st.subheader("Concorrência – Movimentos, Marca e Produtos")

    comp_a, comp_b = st.columns((1,1))
    with comp_a:
        st.markdown("<div class='block-title'>Linha do tempo de movimentos (últimos meses)</div>", unsafe_allow_html=True)
        ev = data["events"][data["events"]["player"].isin([p for p in players_sel if p != BANK])].copy()
        ev = ev[ev["data"].isin(sorted(ev["data"].unique())[-(mes_sel*2):])]  # janela
        ev["data_str"] = ev["data"].dt.strftime("%d/%m")
        fig_evt = px.scatter(ev, x="data", y="player", color="severidade", size="score", hover_data=["tipo", "descricao"], color_discrete_map={"Crítico": DANGER, "Alto": WARNING, "Médio": PRIMARY, "Baixo": MUTED})
        fig_evt.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_evt, use_container_width=True)

    with comp_b:
        st.markdown("<div class='block-title'>Comparador de funil (trimestre selecionado)</div>", unsafe_allow_html=True)
        ftri_comp = cur_tri.copy()
        ftri_comp = ftri_comp.melt(id_vars=["ano", "tri", "player"], value_vars=["awareness", "consideration", "preference"], var_name="etapa", value_name="valor")
        fig_fc = px.bar(ftri_comp, x="etapa", y="valor", color="player", barmode="group", text_auto=".0%")
        fig_fc.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_fc, use_container_width=True)

    st.divider()

    st.markdown("### Cartões – Competitividade (Card Power)")
    card = data["card"][data["card"]["player"].isin(players_sel)].copy()
    card_tri = card[(card["ano"]==ano_tri[0]) & (card["tri"]==ano_tri[1])]

    c1, c2, c3 = st.columns(3)
    with c1:
        base_rb = int(card_tri[(card_tri["player"]==BANK)]["base_ativa"].mean())
        metric_card("Base ativa (média por tipo)", f"{base_rb:,.0f}")
    with c2:
        princ_rb = float(card_tri[(card_tri["player"]==BANK)]["principalidade"].mean())
        metric_card("Principalidade (média)", pct(princ_rb))
    with c3:
        nps_rb = float(card_tri[(card_tri["player"]==BANK)]["nps"].mean())
        metric_card("NPS (média)", f"{nps_rb:.0f}")

    cc1, cc2 = st.columns((1,1))
    with cc1:
        fig_card1 = px.box(card_tri, x="player", y="principalidade", color="player", points="all", title="Distribuição de principalidade por player (tri)")
        fig_card1.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        fig_card1.update_yaxes(tickformat=",.0%")
        st.plotly_chart(fig_card1, use_container_width=True)

    with cc2:
        fig_card2 = px.bar(card_tri, x="tipo", y="nps", color="player", barmode="group", text_auto=True, title="NPS por tipo de cartão")
        fig_card2.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_card2, use_container_width=True)

    with st.expander("Tabela – cartões (tri)"):
        st.dataframe(card_tri.sort_values(["player", "tipo"]).reset_index(drop=True), use_container_width=True, height=240)

    st.divider()

    st.markdown("### Investimentos – Competitividade")
    inv = data["inv"][data["inv"]["player"].isin(players_sel)].copy()
    inv_tri = inv[(inv["ano"]==ano_tri[0]) & (inv["tri"]==ano_tri[1])]

    ic1, ic2 = st.columns((1,1))
    with ic1:
        fig_inv1 = px.bar(inv_tri, x="produto", y="captacao_liquida", color="player", barmode="group", text_auto=True, title="Captação líquida por produto")
        fig_inv1.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_inv1, use_container_width=True)

    with ic2:
        fig_inv2 = px.scatter(inv_tri, x="ticket_medio", y="taxa_fee", color="player", symbol="produto", size="captacao_liquida", hover_data=["produto"], title="Posicionamento (ticket x fee)")
        fig_inv2.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_inv2, use_container_width=True)

    with st.expander("Tabela – investimentos (tri)"):
        st.dataframe(inv_tri.sort_values(["player", "produto"]).reset_index(drop=True), use_container_width=True, height=240)


# -----------------------------------------------------------------------------
# MERCADO
# -----------------------------------------------------------------------------
with tab_mercado:
    st.subheader("Mercado – Tendências, Regulação e Publicações")

    # Radar de tendências (simulado)
    np.random.seed(SEED)
    temas = ["IA/Automação", "Open Finance", "Pix 2.0", "Crédito Responsável", "Segurança/Antifraude", "UX de Onboarding"]
    impacto = np.clip(np.random.normal(70, 15, len(temas)), 30, 95)
    maturidade = np.clip(np.random.normal(55, 20, len(temas)), 10, 95)
    df_radar = pd.DataFrame({"tema": temas, "impacto": impacto, "maturidade": maturidade})

    m1, m2 = st.columns((1,1))
    with m1:
        st.markdown("<div class='block-title'>Radar de tendências (impacto x maturidade)</div>", unsafe_allow_html=True)
        fig_rad = px.scatter(df_radar, x="maturidade", y="impacto", text="tema", size=[18]*len(temas), title="Tendências setoriais")
        fig_rad.update_traces(textposition="top center")
        fig_rad.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_rad, use_container_width=True)

    with m2:
        # Linha do tempo regulatória (simulada)
        bcb_milestones = [
            ("2025-02-15", "Consulta pública – Open Finance"),
            ("2025-03-30", "Circular – Requisitos de segurança"),
            ("2025-05-10", "Comunicado – Pix funcionalidade X"),
            ("2025-07-20", "Resolução – Diretrizes de crédito"),
        ]
        df_reg = pd.DataFrame({
            "data": pd.to_datetime([d for d, _ in bcb_milestones]),
            "evento": [e for _, e in bcb_milestones]
        })
        df_reg["y"] = 1
        fig_reg = px.scatter(df_reg, x="data", y="y", text="evento", title="Linha do tempo regulatória (exemplos)")
        fig_reg.update_traces(textposition="top center")
        fig_reg.update_layout(yaxis=dict(visible=False), height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_reg, use_container_width=True)

    st.divider()

    st.markdown("<div class='block-title'>Publicações & dossiês</div>", unsafe_allow_html=True)
    st.dataframe(data["docs"].sort_values("data", ascending=False), use_container_width=True, height=260)


# -----------------------------------------------------------------------------
# AI – Q&A SIMULADO (RAG MOCK)
# -----------------------------------------------------------------------------
with tab_ai:
    st.subheader("AI – Pergunte aos dados (simulado)")
    st.caption("Protótipo conceitual com base em dados deste app.")

    # Base textual simples (mock): sumariza alguns fatos por tri
    def build_knowledge_base() -> List[Dict[str, str]]:
        kb = []
        # Fatos simples sobre market share e funil no tri selecionado
        ms_last = ms_hist[ms_hist["mes"]==ms_hist["mes"].max()].groupby("player", as_index=False)["market_share"].mean()
        for _, r in ms_last.iterrows():
            kb.append({
                "tema": "market_share",
                "texto": f"No mês mais recente, o market share estimado de {r['player']} foi {r['market_share']:.1%}."
            })
        for _, r in cur_tri.iterrows():
            kb.append({
                "tema": "funnel",
                "texto": f"No {int(r['ano'])}-T{int(r['tri'])}, {r['player']} teve awareness {r['awareness']:.0%}, consideração {r['consideration']:.0%} e preferência {r['preference']:.0%}."
            })
        # Fatos de cartões e investimentos (tri)
        for _, r in data["card"][ (data["card"]["ano"]==ano_tri[0]) & (data["card"]["tri"]==ano_tri[1]) ].groupby("player").agg({"principalidade":"mean"}).reset_index().iterrows():
            kb.append({"tema":"cartao", "texto": f"Principalidade média de cartões do {r['player']}: {r['principalidade']:.0%}."})
        for _, r in data["inv"][ (data["inv"]["ano"]==ano_tri[0]) & (data["inv"]["tri"]==ano_tri[1]) ].groupby(["player"]).agg({"captacao_liquida":"sum"}).reset_index().iterrows():
            kb.append({"tema":"investimentos", "texto": f"Captação líquida total em investimentos no tri para {r['player']}: R$ {r['captacao_liquida']:,}."})
        # Eventos
        ev_last = data["events"].sort_values("data", ascending=False).head(10)
        for _, r in ev_last.iterrows():
            kb.append({"tema":"eventos", "texto": f"{r['data'].strftime('%d/%m')}: {r['tipo']} do {r['player']} (severidade {r['severidade']})."})
        return kb

    KB = build_knowledge_base()

    q = st.text_input("Faça uma pergunta (ex.: 'Qual o market share do RedBank?', 'Como está a preferência vs Nubank?')")
    btn = st.button("Responder")

    def simple_search(query: str, kb: List[Dict[str, str]], topk: int = 5) -> List[str]:
        """Busca semântica simples baseada em sobreposição de palavras (mock)."""
        terms = set([t.lower() for t in query.split()])
        scores = []
        for chunk in kb:
            text = chunk["texto"].lower()
            score = sum(1 for t in terms if t in text)
            scores.append((score, chunk["texto"]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scores[:topk] if _ > 0]

    if btn and q:
        resps = simple_search(q, KB)
        if not resps:
            st.info("Não encontrei uma resposta direta. Tente reformular ou consulte as abas acima.")
        else:
            for r in resps:
                st.markdown(f"- {r}")

    st.divider()
    st.markdown("### Central de Insights (inventário)")
    st.caption("Pesquise e acesse estudos/dossiês.")
    filtro_aba = st.selectbox("Filtrar por aba", options=["Todas"] + sorted(data["docs"]["aba"].unique().tolist()))
    if filtro_aba != "Todas":
        docs = data["docs"][data["docs"]["aba"]==filtro_aba]
    else:
        docs = data["docs"]
    st.dataframe(docs.sort_values("data", ascending=False), use_container_width=True, height=220)

# -----------------------------------------------------------------------------
# RODAPÉ
# -----------------------------------------------------------------------------
st.divider()
st.write(":grey[Protótipo conceitual • Okiar • Dados simulados para apresentação • Versão v1]")
