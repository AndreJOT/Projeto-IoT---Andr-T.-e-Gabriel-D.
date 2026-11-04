import math
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
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
    "Carregue um CSV do RaceChrono, defina a linha "
    "de largada/chegada e compare suas voltas."
)

arquivo = st.file_uploader(
    "Arquivo de telemetria (.csv)",
    type=["csv"]
)

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
    mapa = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    return next(
        (
            mapa[n.lower()]
            for n in opcoes
            if n.lower() in mapa
        ),
        None
    )


lat_col = localizar_coluna([
    "latitude_decimal",
    "latitude",
    "lat"
])

lon_col = localizar_coluna([
    "longitude_decimal",
    "longitude",
    "lon",
    "lng"
])

tempo_col = localizar_coluna([
    "segundos_desde_inicio",
    "timestamp_ms_bruto",
    "timestamp",
    "time",
    "tempo"
])

if lat_col is None or lon_col is None:
    st.error("Não encontrei as colunas de latitude e longitude.")

    with st.expander("Ver colunas disponíveis"):
        st.write(list(df.columns))

    st.stop()


df[lat_col] = pd.to_numeric(
    df[lat_col],
    errors="coerce"
)

df[lon_col] = pd.to_numeric(
    df[lon_col],
    errors="coerce"
)

df = df.dropna(
    subset=[lat_col, lon_col]
).copy()

df = df[
    df[lat_col].between(-90, 90)
    & df[lon_col].between(-180, 180)
].reset_index(drop=True)

if df.empty:
    st.error("O CSV não contém coordenadas GPS válidas.")
    st.stop()


# ========================================================
# IDENTIFICAÇÃO DO TEMPO
# ========================================================

tempo_s = None

if tempo_col is not None:
    valores = pd.to_numeric(
        df[tempo_col],
        errors="coerce"
    )

    if valores.notna().sum() >= 2:

        if str(tempo_col).lower() == "timestamp_ms_bruto":
            tempo_s = valores / 1000.0
        else:
            tempo_s = valores

        if not tempo_s.dropna().is_monotonic_increasing:
            tempo_s = None


# ========================================================
# INFORMAÇÕES TÉCNICAS DO ARQUIVO
# ========================================================

with st.expander("Informações do arquivo"):

    a, b, c = st.columns(3)

    a.metric(
        "Registros GPS válidos",
        len(df)
    )

    b.metric(
        "Coordenadas distintas",
        len(df[[lat_col, lon_col]].drop_duplicates())
    )

    c.metric(
        "Colunas",
        len(df.columns)
    )

    st.caption(
        f"Latitude: {lat_col} · "
        f"Longitude: {lon_col} · "
        f"Tempo: {tempo_col or 'não identificado'}"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.caption("Prévia limitada aos primeiros 100 registros.")


# ========================================================
# MAPA INICIAL E LINHA DE LARGADA/CHEGADA
# ========================================================

st.divider()

st.subheader("1. Trajeto e linha de largada/chegada")

st.caption(
    "Use a ferramenta de linha no mapa, clique nos dois "
    "extremos e finalize com duplo clique."
)

trajeto = [
    [float(lat), float(lon)]
    for lat, lon in zip(df[lat_col], df[lon_col])
]

mapa = folium.Map(
    location=[
        float(df[lat_col].mean()),
        float(df[lon_col].mean())
    ],
    zoom_start=15,
    control_scale=True
)

folium.PolyLine(
    trajeto,
    color="#2563eb",
    weight=4,
    opacity=0.85,
    tooltip="Trajeto GPS"
).add_to(mapa)

folium.CircleMarker(
    trajeto[0],
    radius=5,
    color="green",
    fill=True,
    tooltip="Início do registro"
).add_to(mapa)

folium.CircleMarker(
    trajeto[-1],
    radius=5,
    color="darkred",
    fill=True,
    tooltip="Fim do registro"
).add_to(mapa)

Draw(
    export=False,
    draw_options={
        "polyline": {
            "shapeOptions": {
                "color": "#ef4444",
                "weight": 5
            },
            "allowIntersection": False
        },
        "polygon": False,
        "rectangle": False,
        "circle": False,
        "circlemarker": False,
        "marker": False,
    },
    edit_options={
        "edit": True,
        "remove": True
    }
).add_to(mapa)

resultado = st_folium(
    mapa,
    height=520,
    use_container_width=True,
    returned_objects=["all_drawings"],
    key="mapa_trajeto"
)

desenhos = (
    resultado.get("all_drawings")
    if resultado
    else []
)

if not isinstance(desenhos, list):
    desenhos = []

linhas = [
    d for d in desenhos
    if isinstance(d, dict)
    and d.get("geometry", {}).get("type") == "LineString"
    and len(
        d.get("geometry", {}).get("coordinates", [])
    ) >= 2
]

if not linhas:
    st.info(
        "Desenhe a linha de largada/chegada no mapa "
        "para calcular as voltas."
    )
    st.stop()

coords = linhas[-1]["geometry"]["coordinates"]

ponto_a = (
    float(coords[0][1]),
    float(coords[0][0])
)

ponto_b = (
    float(coords[-1][1]),
    float(coords[-1][0])
)

st.success("Linha de largada/chegada definida.")

with st.expander("Detalhes da linha de largada/chegada"):

    st.write(
        f"**Ponto A:** lat. {ponto_a[0]:.7f}, "
        f"lon. {ponto_a[1]:.7f}"
    )

    st.write(
        f"**Ponto B:** lat. {ponto_b[0]:.7f}, "
        f"lon. {ponto_b[1]:.7f}"
    )


# ========================================================
# FUNÇÕES GEOMÉTRICAS
# ========================================================

raio = 6371000.0

lat_ref = math.radians(
    float(df[lat_col].mean())
)


def xy(lat, lon):
    return (
        raio * math.radians(lon) * math.cos(lat_ref),
        raio * math.radians(lat)
    )


def orient(a, b, c):
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def cruza(p, p2, q, q2):
    eps = 1e-8

    def dentro(a, b, c):
        return (
            min(a[0], b[0]) - eps
            <= c[0]
            <= max(a[0], b[0]) + eps
            and
            min(a[1], b[1]) - eps
            <= c[1]
            <= max(a[1], b[1]) + eps
        )

    o1 = orient(p, p2, q)
    o2 = orient(p, p2, q2)
    o3 = orient(q, q2, p)
    o4 = orient(q, q2, p2)

    if (
        (
            (o1 > eps and o2 < -eps)
            or
            (o1 < -eps and o2 > eps)
        )
        and
        (
            (o3 > eps and o4 < -eps)
            or
            (o3 < -eps and o4 > eps)
        )
    ):
        return True

    return (
        (abs(o1) <= eps and dentro(p, p2, q))
        or
        (abs(o2) <= eps and dentro(p, p2, q2))
        or
        (abs(o3) <= eps and dentro(q, q2, p))
        or
        (abs(o4) <= eps and dentro(q, q2, p2))
    )


def fracao_cruzamento(p, p2, q, q2):
    rx = p2[0] - p[0]
    ry = p2[1] - p[1]

    sx = q2[0] - q[0]
    sy = q2[1] - q[1]

    den = rx * sy - ry * sx

    if abs(den) < 1e-10:
        return 0.5

    qpx = q[0] - p[0]
    qpy = q[1] - p[1]

    return max(
        0.0,
        min(
            1.0,
            (qpx * sy - qpy * sx) / den
        )
    )


# ========================================================
# DETECÇÃO DOS CRUZAMENTOS
# ========================================================

cruzamentos = []

if tempo_s is not None:

    tempos = tempo_s.tolist()

    pontos_xy = [
        xy(float(lat), float(lon))
        for lat, lon in zip(
            df[lat_col],
            df[lon_col]
        )
    ]

    qa = xy(*ponto_a)
    qb = xy(*ponto_b)

    for i in range(1, len(df)):

        if (
            pd.isna(tempos[i - 1])
            or pd.isna(tempos[i])
        ):
            continue

        p = pontos_xy[i - 1]
        p2 = pontos_xy[i]

        if cruza(p, p2, qa, qb):

            frac = fracao_cruzamento(
                p, p2, qa, qb
            )

            t = (
                float(tempos[i - 1])
                + frac
                * (
                    float(tempos[i])
                    - float(tempos[i - 1])
                )
            )

            if (
                not cruzamentos
                or t - cruzamentos[-1] > 2.0
            ):
                cruzamentos.append(t)


# ========================================================
# TABELA DE VOLTAS
# ========================================================

st.divider()

st.subheader("2. Histórico de voltas")

if tempo_s is None:
    st.warning(
        "Não identifiquei uma coluna de tempo numérica "
        "crescente. Use 'segundos_desde_inicio' "
        "ou 'timestamp_ms_bruto'."
    )
    st.stop()

if len(cruzamentos) < 2:
    st.info(
        f"Foram detectados {len(cruzamentos)} cruzamentos. "
        "São necessários pelo menos dois. "
        "Confira a linha e o trajeto GPS."
    )
    st.stop()

tabela = pd.DataFrame([
    {
        "Volta": i,
        "Início (s)": cruzamentos[i - 1],
        "Fim (s)": cruzamentos[i],
        "Tempo da volta (s)": (
            cruzamentos[i] - cruzamentos[i - 1]
        ),
    }
    for i in range(1, len(cruzamentos))
])

idx = tabela["Tempo da volta (s)"].idxmin()

numero_volta_mais_rapida = int(
    tabela.loc[idx, "Volta"]
)

tempo_mais_rapido = float(
    tabela.loc[idx, "Tempo da volta (s)"]
)

tabela["Diferença para anterior (s)"] = (
    tabela["Tempo da volta (s)"].diff()
)

tabela["Diferença para mais rápida (s)"] = (
    tabela["Tempo da volta (s)"]
    - tempo_mais_rapido
)

tabela_exibicao = tabela.copy()

for coluna in [
    "Início (s)",
    "Fim (s)",
    "Tempo da volta (s)",
    "Diferença para anterior (s)",
    "Diferença para mais rápida (s)"
]:
    tabela_exibicao[coluna] = (
        tabela_exibicao[coluna].round(3)
    )


def destacar_melhor_volta(linha):
    if int(linha["Volta"]) == numero_volta_mais_rapida:
        return [
            "background-color:#d8f3dc;"
            "color:#173b25;font-weight:bold"
        ] * len(linha)

    return [""] * len(linha)


st.success(
    f"🏆 Melhor volta: Volta {numero_volta_mais_rapida} "
    f"· {tempo_mais_rapido:.3f} s"
)

with st.expander(
    "Tabela completa de voltas",
    expanded=True
):
    st.dataframe(
        tabela_exibicao.style.apply(
            destacar_melhor_volta,
            axis=1
        ),
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

st.subheader("3. Análise de desempenho")

volta_selecionada = st.selectbox(
    "Selecione a volta para analisar",
    tabela["Volta"].tolist(),
    format_func=lambda v: (
        f"Volta {v} 🏆 (melhor)"
        if v == numero_volta_mais_rapida
        else f"Volta {v}"
    ),
    key="seletor_volta"
)

dados_volta = tabela.loc[
    tabela["Volta"] == volta_selecionada
].iloc[0]

inicio = float(dados_volta["Início (s)"])
fim = float(dados_volta["Fim (s)"])

tempo_volta = float(
    dados_volta["Tempo da volta (s)"]
)


def preparar_dados_volta(inicio_s, fim_s):

    mask = (
        tempo_s.notna()
        & (tempo_s >= inicio_s)
        & (tempo_s <= fim_s)
    )

    dados = df.loc[mask].copy()

    dados["_tempo_s"] = (
        tempo_s.loc[mask].astype(float)
    )

    dados = (
        dados.sort_values("_tempo_s")
        .drop_duplicates("_tempo_s")
    )

    if len(dados) < 2:
        return None

    latitudes = np.radians(
        dados[lat_col].to_numpy(float)
    )

    longitudes = np.radians(
        dados[lon_col].to_numpy(float)
    )

    dlat = np.diff(latitudes)
    dlon = np.diff(longitudes)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(latitudes[:-1])
        * np.cos(latitudes[1:])
        * np.sin(dlon / 2) ** 2
    )

    segmentos = (
        2 * 6371000.0
        * np.arcsin(
            np.sqrt(np.clip(a, 0, 1))
        )
    )

    dados["Distância (m)"] = np.concatenate(
        ([0.0], np.cumsum(segmentos))
    )

    dados["Tempo decorrido (s)"] = (
        dados["_tempo_s"]
        - dados["_tempo_s"].iloc[0]
    )

    dt = dados["_tempo_s"].diff().to_numpy(float)

    ds = np.concatenate(
        ([np.nan], segmentos)
    )

    vel = np.divide(
        ds * 3.6,
        dt,
        out=np.full(len(dados), np.nan),
        where=dt > 0
    )

    vel[vel > 500] = np.nan

    dados["Velocidade (km/h)"] = vel

    return dados


dados_atual = preparar_dados_volta(
    inicio,
    fim
)

linha_melhor = tabela.loc[
    tabela["Volta"] == numero_volta_mais_rapida
].iloc[0]

dados_melhor = preparar_dados_volta(
    float(linha_melhor["Início (s)"]),
    float(linha_melhor["Fim (s)"])
)


# ========================================================
# INDICADORES PRINCIPAIS
# ========================================================

diferenca = tempo_volta - tempo_mais_rapido

valor_diff = (
    "0,000 s"
    if int(volta_selecionada) == numero_volta_mais_rapida
    else f"+{diferenca:.3f} s"
)

m1, m2, m3 = st.columns([1.2, 1, 1])

m1.metric(
    f"Tempo da volta {volta_selecionada}",
    f"{tempo_volta:.3f} s"
)

m2.metric(
    "Diferença para a melhor",
    valor_diff
)

m3.metric(
    "Melhor volta",
    f"{tempo_mais_rapido:.3f} s",
    f"Volta {numero_volta_mais_rapida}"
)


# ========================================================
# VELOCIDADE POR DISTÂNCIA
# ========================================================

st.markdown("### Velocidade por distância")

st.caption(
    "Compare a velocidade da volta selecionada com "
    "a melhor volta ao longo do circuito."
)

if dados_atual is not None and dados_melhor is not None:

    atual = (
        dados_atual
        .dropna(subset=["Velocidade (km/h)"])
        .sort_values("Distância (m)")
    )

    melhor = (
        dados_melhor
        .dropna(subset=["Velocidade (km/h)"])
        .sort_values("Distância (m)")
    )

    if not atual.empty and not melhor.empty:

        dist_max = min(
            float(atual["Distância (m)"].max()),
            float(melhor["Distância (m)"].max())
        )

        if dist_max > 0:

            max_a = float(
                atual["Velocidade (km/h)"].max()
            )

            max_m = float(
                melhor["Velocidade (km/h)"].max()
            )

            tempo_a = float(
                dados_atual["_tempo_s"].iloc[-1]
                - dados_atual["_tempo_s"].iloc[0]
            )

            tempo_m = float(
                dados_melhor["_tempo_s"].iloc[-1]
                - dados_melhor["_tempo_s"].iloc[0]
            )

            media_a = (
                float(dados_atual["Distância (m)"].iloc[-1])
                * 3.6 / tempo_a
                if tempo_a > 0
                else float("nan")
            )

            media_m = (
                float(dados_melhor["Distância (m)"].iloc[-1])
                * 3.6 / tempo_m
                if tempo_m > 0
                else float("nan")
            )

            v1, v2, v3, v4 = st.columns(4)

            v1.metric(
                "Máxima · selecionada",
                f"{max_a:.1f} km/h"
            )

            v2.metric(
                "Média GPS · selecionada",
                f"{media_a:.1f} km/h"
            )

            v3.metric(
                "Máxima · melhor volta",
                f"{max_m:.1f} km/h"
            )

            v4.metric(
                "Média GPS · melhor volta",
                f"{media_m:.1f} km/h"
            )

            dist = np.linspace(
                0,
                dist_max,
                200
            )

            da = atual["Distância (m)"].to_numpy(float)
            va = atual["Velocidade (km/h)"].to_numpy(float)

            dm = melhor["Distância (m)"].to_numpy(float)
            vm = melhor["Velocidade (km/h)"].to_numpy(float)

            da_u, ia = np.unique(
                da,
                return_index=True
            )

            dm_u, im = np.unique(
                dm,
                return_index=True
            )

            chart = pd.DataFrame({
                "Distância (m)": dist,
                f"Volta {int(volta_selecionada)}": (
                    np.interp(dist, da_u, va[ia])
                ),
                f"Melhor volta {numero_volta_mais_rapida}": (
                    np.interp(dist, dm_u, vm[im])
                )
            })

            st.line_chart(
                chart.set_index("Distância (m)"),
                x_label="Distância (m)",
                y_label="Velocidade (km/h)"
            )

        else:
            st.info(
                "Não há distância comum suficiente "
                "para comparar as voltas."
            )

    else:
        st.info(
            "Não há registros de velocidade suficientes "
            "para comparar as voltas."
        )

else:
    st.info(
        "Não foi possível calcular os dados GPS "
        "de uma das voltas."
    )


# ========================================================
# Diferença PARA A MELHOR VOLTA
# ========================================================

st.markdown("### Diferença acumulada para a melhor volta")


if int(volta_selecionada) == numero_volta_mais_rapida:

    st.info(
        "Esta é a melhor volta; a diferença em relação a ela é zero."
    )

elif dados_atual is not None and dados_melhor is not None:

    da = dados_atual["Distância (m)"].to_numpy(float)

    ta = dados_atual[
        "Tempo decorrido (s)"
    ].to_numpy(float)

    dm = dados_melhor["Distância (m)"].to_numpy(float)

    tm = dados_melhor[
        "Tempo decorrido (s)"
    ].to_numpy(float)

    oa = np.argsort(da)
    om = np.argsort(dm)

    da, ta = da[oa], ta[oa]
    dm, tm = dm[om], tm[om]

    dau, ia = np.unique(
        da,
        return_index=True
    )

    dmu, im = np.unique(
        dm,
        return_index=True
    )

    tau = ta[ia]
    tmu = tm[im]

    comum = min(
        float(dau[-1]),
        float(dmu[-1])
    )

    if comum > 0:

        dist = np.linspace(
            0,
            comum,
            101
        )

        gap = (
            np.interp(dist, dau, tau)
            - np.interp(dist, dmu, tmu)
        )

        gap_df = pd.DataFrame({
            "Distância (m)": dist.round(1),
            "Diferença (s)": gap.round(3)
        })

        st.line_chart(
            gap_df.set_index("Distância (m)")["Diferença (s)"],
            x_label="Distância (m)",
            y_label="Diferença (s)"
        )

    else:
        st.info(
            "Não foi possível comparar as distâncias destas voltas."
        )

else:
    st.info(
        "Não há registros GPS suficientes para comparar "
        "esta volta com a melhor."
    )


# ========================================================
# MAPA DE VELOCIDADE POR TRECHO
# ========================================================

st.divider()

st.markdown("### Mapa de velocidade por trecho")

st.caption(
    "Passe o mouse sobre um trecho para ver a velocidade "
    "estimada, a distância e o intervalo de tempo."
)

mask = (
    tempo_s.notna()
    & (tempo_s >= inicio)
    & (tempo_s <= fim)
)

dados_gps = df.loc[mask].copy()

dados_gps["_tempo_s"] = tempo_s.loc[
    dados_gps.index
]

dados_gps = (
    dados_gps.sort_values("_tempo_s")
    .reset_index(drop=True)
)

trajeto_volta = [
    [float(lat), float(lon)]
    for lat, lon in zip(
        dados_gps[lat_col],
        dados_gps[lon_col]
    )
]


def distancia_metros(lat1, lon1, lat2, lon2):

    r = 6371000.0

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        (lat1, lon1, lat2, lon2)
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * r * math.asin(
        min(1.0, math.sqrt(a))
    )


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


if len(trajeto_volta) < 2:

    st.warning(
        "Não há registros GPS suficientes dentro desta volta "
        "para desenhar o trajeto."
    )

else:

    centro = [
        sum(p[i] for p in trajeto_volta)
        / len(trajeto_volta)
        for i in (0, 1)
    ]

    mapa_volta = folium.Map(
        location=centro,
        zoom_start=16,
        control_scale=True
    )

    # Trajeto completo da sessão como referência

    folium.PolyLine(
        trajeto,
        color="gray",
        weight=3,
        opacity=0.22,
        tooltip="Trajeto completo da sessão"
    ).add_to(mapa_volta)

    velocidades = []

    for i in range(1, len(dados_gps)):

        ant = dados_gps.iloc[i - 1]
        cur = dados_gps.iloc[i]

        lat1 = float(ant[lat_col])
        lon1 = float(ant[lon_col])

        lat2 = float(cur[lat_col])
        lon2 = float(cur[lon_col])

        t1 = float(ant["_tempo_s"])
        t2 = float(cur["_tempo_s"])

        dt = t2 - t1

        d = distancia_metros(
            lat1,
            lon1,
            lat2,
            lon2
        )

        if dt <= 0:
            v = float("nan")
        else:
            v = d / dt * 3.6

        if v > 500:
            v = float("nan")

        velocidades.append(v)

        cor = cor_velocidade(v)

        if pd.isna(v):
            texto = "Velocidade indisponível"
        else:
            texto = f"Velocidade: {v:.1f} km/h"

        folium.PolyLine(
            [[lat1, lon1], [lat2, lon2]],
            color=cor,
            weight=6,
            opacity=0.95,
            tooltip=(
                f"Trecho {i}: {texto}"
                f" | Distância: {d:.1f} m"
                f" | Intervalo: {dt:.2f} s"
            )
        ).add_to(mapa_volta)

    # Marcador de início

    folium.Marker(
        trajeto_volta[0],
        tooltip="Início aproximado da volta",
        icon=folium.Icon(
            color="green",
            icon="play"
        )
    ).add_to(mapa_volta)

    # Marcador de fim

    folium.Marker(
        trajeto_volta[-1],
        tooltip="Fim aproximado da volta",
        icon=folium.Icon(
            color="red",
            icon="stop"
        )
    ).add_to(mapa_volta)

    # Linha de largada/chegada

    folium.PolyLine(
        [ponto_a, ponto_b],
        color="#2563eb",
        weight=4,
        opacity=0.9,
        tooltip="Linha de largada/chegada"
    ).add_to(mapa_volta)

    # Legenda de velocidade

    legenda = """
    <div style="
        position:fixed;
        bottom:25px;
        left:25px;
        z-index:9999;
        background:#fff;
        border:1px solid #aaa;
        border-radius:8px;
        padding:9px 11px;
        font-size:12px;
        color:#222;
        box-shadow:0 1px 5px #0003;
        line-height:1.5;
    ">
        <b>Velocidade (km/h)</b><br>
        <span style="color:#06f;">━━</span> &lt; 30<br>
        <span style="color:#0a0;">━━</span> 30–60<br>
        <span style="color:#cc0;">━━</span> 60–90<br>
        <span style="color:#f80;">━━</span> 90–120<br>
        <span style="color:#f00;">━━</span> ≥ 120<br>
        <span style="color:#888;">━━</span> Sem dado válido
    </div>
    """

    mapa_volta.get_root().html.add_child(
        folium.Element(legenda)
    )

    mapa_volta.fit_bounds(trajeto_volta)

    st_folium(
        mapa_volta,
        height=520,
        use_container_width=True,
        returned_objects=[],
        key=f"mapa_volta_velocidade_{int(volta_selecionada)}"
    )