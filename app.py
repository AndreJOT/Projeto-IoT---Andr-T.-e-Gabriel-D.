import math
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="GPS Lap Analyzer",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,.07);
    border: 1px solid rgba(128,128,128,.18);
    padding: 13px 15px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🏁 GPS Lap Analyzer")
st.caption("Análise de voltas e telemetria GPS")
st.write(
    "Carregue um CSV do RaceChrono com as voltas já identificadas "
    "para analisar tempos, velocidades e desempenho."
)

arquivo = st.file_uploader("Arquivo de telemetria (.csv)", type=["csv"])

if arquivo is None:
    st.info("Carregue o CSV exportado pelo RaceChrono para começar.")
    st.stop()


# ========================================================
# LEITURA DO CSV
# ========================================================

df = None
erro_leitura = None

for encoding in ("utf-8-sig", "utf-8", "latin1"):
    for sep in (None, ",", ";", "\t"):
        try:
            arquivo.seek(0)
            tentativa = pd.read_csv(
                arquivo,
                encoding=encoding,
                sep=sep,
                engine="python" if sep is None else "c"
            )
            if len(tentativa.columns) > 1:
                df = tentativa
                break
        except Exception as e:
            erro_leitura = e
    if df is not None:
        break

if df is None:
    st.error(f"Não foi possível ler o CSV: {erro_leitura}")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]


def localizar_coluna(opcoes):
    mapa = {str(c).strip().lower(): c for c in df.columns}
    return next((mapa[n.lower()] for n in opcoes if n.lower() in mapa), None)


lat_col = localizar_coluna(["latitude_decimal", "latitude", "lat"])
lon_col = localizar_coluna(["longitude_decimal", "longitude", "lon", "lng"])
tempo_col = localizar_coluna([
    "segundos_desde_inicio",
    "timestamp_ms_bruto",
    "timestamp",
    "time",
    "tempo"
])
volta_col = localizar_coluna([
    "volta_associada",
    "lap",
    "lap_number",
    "lapnumber",
    "volta",
    "numero_volta"
])

if lat_col is None or lon_col is None:
    st.error("Não encontrei as colunas de latitude e longitude.")
    st.write("Colunas disponíveis:", list(df.columns))
    st.stop()

if volta_col is None:
    st.error(
        "Não encontrei a coluna que identifica as voltas. "
        "Este analisador espera a coluna 'volta_associada' "
        "ou uma coluna equivalente no CSV."
    )
    st.write("Colunas disponíveis:", list(df.columns))
    st.stop()

df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
df[volta_col] = pd.to_numeric(df[volta_col], errors="coerce")

if tempo_col is None:
    st.error("Não encontrei uma coluna de tempo no CSV.")
    st.stop()

tempo_valores = pd.to_numeric(df[tempo_col], errors="coerce")
if str(tempo_col).lower() == "timestamp_ms_bruto":
    df["_tempo_s"] = tempo_valores / 1000.0
else:
    df["_tempo_s"] = tempo_valores

df = df.dropna(subset=[lat_col, lon_col, "_tempo_s"]).copy()
df = df[
    df[lat_col].between(-90, 90)
    & df[lon_col].between(-180, 180)
].copy()

if df.empty:
    st.error("O CSV não contém coordenadas GPS e tempos válidos.")
    st.stop()

df = df.sort_values("_tempo_s").reset_index(drop=True)

# Apenas registros associados a uma volta são usados na análise.
df_voltas = df.dropna(subset=[volta_col]).copy()

if df_voltas.empty:
    st.error(
        f"A coluna '{volta_col}' existe, mas não possui números de volta válidos."
    )
    st.stop()

df_voltas[volta_col] = df_voltas[volta_col].astype(int)


# ========================================================
# PREPARAÇÃO DOS DADOS DE CADA VOLTA
# ========================================================

def distancia_metros(lat1, lon1, lat2, lon2):
    """Calcula a distância entre dois pontos GPS usando Haversine."""
    raio_terra = 6371000.0
    lat1, lon1, lat2, lon2 = map(
        math.radians, (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * raio_terra * math.asin(min(1.0, math.sqrt(a)))


def preparar_dados_volta(numero_volta):
    dados = df_voltas[df_voltas[volta_col] == int(numero_volta)].copy()
    dados = dados.sort_values("_tempo_s").drop_duplicates("_tempo_s")
    if len(dados) < 2:
        return None

    latitudes = np.radians(dados[lat_col].to_numpy(dtype=float))
    longitudes = np.radians(dados[lon_col].to_numpy(dtype=float))

    dlat = np.diff(latitudes)
    dlon = np.diff(longitudes)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(latitudes[:-1]) * np.cos(latitudes[1:])
        * np.sin(dlon / 2) ** 2
    )

    segmentos = (
        2 * 6371000.0
        * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )

    dados["Distância (m)"] = np.concatenate(([0.0], np.cumsum(segmentos)))
    dados["Tempo decorrido (s)"] = (
        dados["_tempo_s"] - dados["_tempo_s"].iloc[0]
    )

    dt = dados["_tempo_s"].diff().to_numpy(dtype=float)
    ds = np.concatenate(([np.nan], segmentos))
    velocidade = np.divide(
        ds * 3.6,
        dt,
        out=np.full(len(dados), np.nan),
        where=dt > 0
    )
    velocidade[velocidade > 500] = np.nan
    dados["Velocidade (km/h)"] = velocidade

    return dados


# ========================================================
# HISTÓRICO E TEMPOS DAS VOLTAS
# ========================================================

st.divider()
st.subheader("1. Histórico de voltas")

registros = []

for numero in sorted(df_voltas[volta_col].unique()):
    dados = df_voltas[df_voltas[volta_col] == numero].sort_values("_tempo_s")
    if len(dados) < 2:
        continue

    inicio = float(dados["_tempo_s"].iloc[0])
    fim = float(dados["_tempo_s"].iloc[-1])
    duracao = fim - inicio

    if duracao > 0:
        registros.append({
            "Volta": int(numero),
            "Início (s)": inicio,
            "Fim (s)": fim,
            "Tempo da volta (s)": duracao,
            "Registros GPS": len(dados)
        })

if not registros:
    st.error("Não há voltas com pelo menos dois registros GPS e tempo válido.")
    st.stop()

tabela = pd.DataFrame(registros).sort_values("Volta").reset_index(drop=True)

idx_melhor = tabela["Tempo da volta (s)"].idxmin()
numero_volta_mais_rapida = int(tabela.loc[idx_melhor, "Volta"])
tempo_mais_rapido = float(tabela.loc[idx_melhor, "Tempo da volta (s)"])

tabela["Diferença para anterior (s)"] = tabela["Tempo da volta (s)"].diff()
tabela["Diferença para mais rápida (s)"] = (
    tabela["Tempo da volta (s)"] - tempo_mais_rapido
)

tabela_exibicao = tabela.copy()
for coluna in [
    "Início (s)", "Fim (s)", "Tempo da volta (s)",
    "Diferença para anterior (s)", "Diferença para mais rápida (s)"
]:
    tabela_exibicao[coluna] = tabela_exibicao[coluna].round(3)


def destacar_melhor_volta(linha):
    if int(linha["Volta"]) == numero_volta_mais_rapida:
        return [
            "background-color:#d8f3dc;color:#173b25;font-weight:bold"
        ] * len(linha)
    return [""] * len(linha)


st.success(
    f"🏆 Melhor volta: Volta {numero_volta_mais_rapida} "
    f"· {tempo_mais_rapido:.3f} s"
)

with st.expander("Tabela completa de voltas", expanded=True):
    st.dataframe(
        tabela_exibicao.style.apply(destacar_melhor_volta, axis=1),
        use_container_width=True,
        hide_index=True
    )

with st.expander("Evolução dos tempos por volta"):
    st.line_chart(
        tabela.set_index("Volta")["Tempo da volta (s)"],
        x_label="Volta",
        y_label="Tempo (s)"
    )


# ========================================================
# ANÁLISE DA VOLTA SELECIONADA
# ========================================================

st.divider()
st.subheader("2. Análise de desempenho")

voltas_disponiveis = tabela["Volta"].astype(int).tolist()

volta_selecionada = st.selectbox(
    "Selecione a volta para analisar",
    voltas_disponiveis,
    format_func=lambda v: (
        f"Volta {v} 🏆 (melhor)"
        if v == numero_volta_mais_rapida else f"Volta {v}"
    ),
    key="seletor_volta"
)

linha_selecionada = tabela[tabela["Volta"] == volta_selecionada].iloc[0]
tempo_volta = float(linha_selecionada["Tempo da volta (s)"])
dados_atual = preparar_dados_volta(volta_selecionada)
dados_melhor = preparar_dados_volta(numero_volta_mais_rapida)

diferenca = tempo_volta - tempo_mais_rapido
valor_diff = (
    "0,000 s" if volta_selecionada == numero_volta_mais_rapida
    else f"+{diferenca:.3f} s"
)

m1, m2, m3 = st.columns([1.2, 1, 1])
m1.metric(f"Tempo da volta {volta_selecionada}", f"{tempo_volta:.3f} s")
m2.metric("Diferença para a melhor", valor_diff)
m3.metric("Melhor volta", f"{tempo_mais_rapido:.3f} s",
          f"Volta {numero_volta_mais_rapida}")


# ========================================================
# VELOCIDADE POR DISTÂNCIA
# ========================================================

st.markdown("### Velocidade por distância")
st.caption(
    "Compare a velocidade da volta selecionada com a melhor volta "
    "ao longo da distância percorrida."
)

if dados_atual is not None and dados_melhor is not None:
    atual = dados_atual.dropna(subset=["Velocidade (km/h)"]).sort_values("Distância (m)")
    melhor = dados_melhor.dropna(subset=["Velocidade (km/h)"]).sort_values("Distância (m)")

    if not atual.empty and not melhor.empty:
        dist_max = min(
            float(atual["Distância (m)"].max()),
            float(melhor["Distância (m)"].max())
        )

        if dist_max > 0:
            max_a = float(atual["Velocidade (km/h)"].max())
            max_m = float(melhor["Velocidade (km/h)"].max())

            tempo_a = float(
                dados_atual["_tempo_s"].iloc[-1] - dados_atual["_tempo_s"].iloc[0]
            )
            tempo_m = float(
                dados_melhor["_tempo_s"].iloc[-1] - dados_melhor["_tempo_s"].iloc[0]
            )

            media_a = (
                float(dados_atual["Distância (m)"].iloc[-1]) * 3.6 / tempo_a
                if tempo_a > 0 else float("nan")
            )
            media_m = (
                float(dados_melhor["Distância (m)"].iloc[-1]) * 3.6 / tempo_m
                if tempo_m > 0 else float("nan")
            )

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Máxima · selecionada", f"{max_a:.1f} km/h")
            v2.metric("Média GPS · selecionada", f"{media_a:.1f} km/h")
            v3.metric("Máxima · melhor volta", f"{max_m:.1f} km/h")
            v4.metric("Média GPS · melhor volta", f"{media_m:.1f} km/h")

            distancia = np.linspace(0, dist_max, 200)
            da = atual["Distância (m)"].to_numpy(dtype=float)
            va = atual["Velocidade (km/h)"].to_numpy(dtype=float)
            dm = melhor["Distância (m)"].to_numpy(dtype=float)
            vm = melhor["Velocidade (km/h)"].to_numpy(dtype=float)

            da_u, ia = np.unique(da, return_index=True)
            dm_u, im = np.unique(dm, return_index=True)

            chart = pd.DataFrame({
                "Distância (m)": distancia,
                f"Volta {int(volta_selecionada)}": np.interp(distancia, da_u, va[ia]),
                f"Melhor volta {numero_volta_mais_rapida}": np.interp(distancia, dm_u, vm[im])
            })

            st.line_chart(
                chart.set_index("Distância (m)"),
                x_label="Distância (m)",
                y_label="Velocidade (km/h)"
            )
        else:
            st.info("Não há distância comum suficiente para comparar as voltas.")
    else:
        st.info("Não há registros de velocidade suficientes para comparar as voltas.")
else:
    st.info("Não foi possível calcular os dados GPS de uma das voltas.")


# ========================================================
# DIFERENÇA ACUMULADA PARA A MELHOR VOLTA
# ========================================================

st.markdown("### Diferença acumulada para a melhor volta")

if volta_selecionada == numero_volta_mais_rapida:
    st.info("Esta é a melhor volta; a diferença em relação a ela é zero.")

elif dados_atual is not None and dados_melhor is not None:
    da = dados_atual["Distância (m)"].to_numpy(dtype=float)
    ta = dados_atual["Tempo decorrido (s)"].to_numpy(dtype=float)
    dm = dados_melhor["Distância (m)"].to_numpy(dtype=float)
    tm = dados_melhor["Tempo decorrido (s)"].to_numpy(dtype=float)

    oa = np.argsort(da)
    om = np.argsort(dm)
    da, ta = da[oa], ta[oa]
    dm, tm = dm[om], tm[om]

    dau, ia = np.unique(da, return_index=True)
    dmu, im = np.unique(dm, return_index=True)
    tau = ta[ia]
    tmu = tm[im]

    distancia_comum = min(float(dau[-1]), float(dmu[-1]))

    if distancia_comum > 0:
        distancia = np.linspace(0, distancia_comum, 101)
        gap = (
            np.interp(distancia, dau, tau)
            - np.interp(distancia, dmu, tmu)
        )

        gap_df = pd.DataFrame({
            "Distância (m)": distancia.round(1),
            "Diferença (s)": gap.round(3)
        })

        st.line_chart(
            gap_df.set_index("Distância (m)")["Diferença (s)"],
            x_label="Distância (m)",
            y_label="Diferença (s)"
        )
        st.caption(
            "Gap positivo = a volta selecionada está atrás da melhor. "
            "Gap negativo = está à frente."
        )
    else:
        st.info("Não foi possível comparar as distâncias destas voltas.")
else:
    st.info("Não há registros GPS suficientes para comparar esta volta com a melhor.")


# ========================================================
# MAPA DE VELOCIDADE POR TRECHO
# ========================================================

st.divider()
st.markdown("### Mapa de velocidade por trecho")
st.caption("Passe o mouse sobre um trecho para ver velocidade, distância e intervalo de tempo.")

if dados_atual is None or len(dados_atual) < 2:
    st.warning("Não há registros GPS suficientes nesta volta para desenhar o trajeto.")
else:
    dados_gps = dados_atual.reset_index(drop=True)
    trajeto_volta = [
        [float(lat), float(lon)]
        for lat, lon in zip(dados_gps[lat_col], dados_gps[lon_col])
    ]

    centro = [
        sum(p[i] for p in trajeto_volta) / len(trajeto_volta)
        for i in (0, 1)
    ]

    mapa_volta = folium.Map(
        location=centro,
        zoom_start=16,
        control_scale=True
    )

    folium.PolyLine(
        trajeto_volta,
        color="#2563eb",
        weight=3,
        opacity=0.25,
        tooltip="Trajeto da volta"
    ).add_to(mapa_volta)

    def cor_velocidade(v):
        if pd.isna(v):
            return "#808080"
        if v < 30:
            return "#0066FF"
        if v < 60:
            return "#00AA00"
        if v < 90:
            return "#CCCC00"
        if v < 120:
            return "#FF8800"
        return "#FF0000"

    for i in range(1, len(dados_gps)):
        anterior = dados_gps.iloc[i - 1]
        atual = dados_gps.iloc[i]

        lat1, lon1 = float(anterior[lat_col]), float(anterior[lon_col])
        lat2, lon2 = float(atual[lat_col]), float(atual[lon_col])
        t1, t2 = float(anterior["_tempo_s"]), float(atual["_tempo_s"])
        dt = t2 - t1

        distancia_trecho = distancia_metros(lat1, lon1, lat2, lon2)
        velocidade = (
            distancia_trecho / dt * 3.6 if dt > 0 else float("nan")
        )
        if velocidade > 500:
            velocidade = float("nan")

        cor = cor_velocidade(velocidade)
        texto = (
            f"Velocidade: {velocidade:.1f} km/h"
            if pd.notna(velocidade) else "Velocidade indisponível"
        )

        folium.PolyLine(
            [[lat1, lon1], [lat2, lon2]],
            color=cor,
            weight=6,
            opacity=0.95,
            tooltip=(
                f"Trecho {i}: {texto} | "
                f"Distância: {distancia_trecho:.1f} m | "
                f"Intervalo: {dt:.2f} s"
            )
        ).add_to(mapa_volta)

    folium.Marker(
        trajeto_volta[0],
        tooltip="Início da volta",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(mapa_volta)

    folium.Marker(
        trajeto_volta[-1],
        tooltip="Fim da volta",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(mapa_volta)

    legenda = """
    <div style="
        position:fixed; bottom:25px; left:25px; z-index:9999;
        background:#fff; border:1px solid #aaa; border-radius:8px;
        padding:9px 11px; font-size:12px; color:#222;
        box-shadow:0 1px 5px #0003; line-height:1.5;">
        <b>Velocidade (km/h)</b><br>
        <span style="color:#06f;">━━</span> &lt; 30<br>
        <span style="color:#0a0;">━━</span> 30–60<br>
        <span style="color:#cc0;">━━</span> 60–90<br>
        <span style="color:#f80;">━━</span> 90–120<br>
        <span style="color:#f00;">━━</span> ≥ 120<br>
        <span style="color:#888;">━━</span> Sem dado válido
    </div>
    """
    mapa_volta.get_root().html.add_child(folium.Element(legenda))
    mapa_volta.fit_bounds(trajeto_volta)

    st_folium(
        mapa_volta,
        height=520,
        use_container_width=True,
        returned_objects=[],
        key=f"mapa_volta_velocidade_{int(volta_selecionada)}"
    )
