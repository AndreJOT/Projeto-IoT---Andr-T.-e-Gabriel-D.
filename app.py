
import json
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_csv(file):
    for encoding in ('utf-8-sig', 'latin1'):
        for sep in (None, ',', ';', '\t'):
            try:
                file.seek(0)
                d = pd.read_csv(file, encoding=encoding, sep=sep, engine='python')
                if len(d.columns) > 1:
                    d.columns = d.columns.str.strip().str.lower()
                    return d
            except (ValueError, UnicodeError, pd.errors.ParserError):
                pass
    raise ValueError('CSV não reconhecido.')


def prepare(d, laps=None, meta=None):
    d = d.copy()
    meta = meta or {}

    aliases = {
        'lat': ['latitude_decimal', 'latitude', 'lat'],
        'lon': ['longitude_decimal', 'longitude', 'lon', 'lng'],
        'lap': ['volta_associada', 'lap', 'lap_number', 'lapnumber', 'volta', 'numero_volta'],
        'speed': ['velocidade_kmh', 'speed_kmh', 'speed (km/h)']
    }

    for target, names in aliases.items():
        col = next((n for n in names if n in d), None)
        if col:
            d[target] = pd.to_numeric(d[col], errors='coerce')

    if not {'lat', 'lon', 'lap'} <= set(d):
        raise ValueError('CSV precisa de latitude, longitude e número de volta.')

    col = next((n for n in [
        'timestamp_ms', 'timestamp_ms_bruto', 'segundos_desde_inicio',
        'timestamp', 'time', 'tempo'
    ] if n in d), None)

    if col is None:
        raise ValueError('Coluna de tempo não encontrada.')

    d['t'] = pd.to_numeric(d[col], errors='coerce') / (
        1000 if col.endswith('_ms') or col == 'timestamp_ms_bruto' else 1
    )

    if col not in ('timestamp_ms', 'timestamp_ms_bruto') and laps is not None:
        if col == 'segundos_desde_inicio' and 'startedTimestampMs' in meta:
            d['t'] += meta['startedTimestampMs'] / 1000
        else:
            raise ValueError(
                'Para usar laps.json, é necessário timestamp_ms ou '
                'segundos_desde_inicio com metadata.json.'
            )

    d = d.dropna(subset=['lat', 'lon', 't']).sort_values('t').drop_duplicates('t')
    d = d[d.lat.between(-90, 90) & d.lon.between(-180, 180)].copy()

    if 'fix_quality' in d:
        d = d[pd.to_numeric(d.fix_quality, errors='coerce') > 0].copy()

    if d.empty:
        raise ValueError('Sem amostras GPS válidas.')

    official = laps is not None

    # RaceMetrics: cada volta concluída é confirmada pela volta seguinte.
    # A última volta em andamento nunca entra no ranking.
    if laps is None and {'ultima_volta_s', 'tempo_volta_s'} <= set(d):
        laps = []

        groups = {
            int(n): g
            for n, g in d.dropna(subset=['lap']).groupby('lap')
            if n >= 0 and n == int(n)
        }

        for n, g in sorted(groups.items()):
            if n <= 0 or n + 1 not in groups:
                continue

            following = groups[n + 1]
            durations = pd.to_numeric(
                following['ultima_volta_s'], errors='coerce'
            ).dropna()
            durations = durations[durations > 0]

            if durations.empty:
                continue

            duration = float(durations.mode().iloc[0])

            starts = (
                g.t - pd.to_numeric(g['tempo_volta_s'], errors='coerce')
            ).dropna()

            if starts.empty:
                continue

            a = float(starts.median())
            start_ms = round(a * 1000)
            duration_ms = round(duration * 1000)

            laps.append(dict(
                number=n,
                startTimestampMs=start_ms,
                endTimestampMs=start_ms + duration_ms,
                durationMs=duration_ms,
                valid=True
            ))

        official = True

    records = []
    traces = {}

    if official:
        if not isinstance(laps, list):
            raise ValueError('laps.json deve conter uma lista de voltas.')

        for r in laps:
            if r.get('valid') is not True:
                continue

            n = int(r['number'])
            a = r['startTimestampMs'] / 1000
            b = r['endTimestampMs'] / 1000
            duration = r['durationMs'] / 1000

            if n <= 0 or b <= a or duration <= 0:
                continue

            if abs((b - a) - duration) > 0.005:
                raise ValueError(f'Tempo inconsistente na volta {n}.')

            records.append((n, a, b, duration))

    else:
        for n, g in d[(d.lap > 0) & (d.lap < d.lap.max())].groupby('lap'):
            if len(g) >= 2:
                records.append((
                    int(n), g.t.min(), g.t.max(), g.t.max() - g.t.min()
                ))

    for n, a, b, duration in records:
        if official:
            g = d[(d.t >= a) & (d.t <= b)].copy()

            # Interpolar somente limites cercados por amostras,
            # sem extrapolar.
            for boundary in (a, b):
                if not (d.t == boundary).any() and d.t.min() < boundary < d.t.max():
                    k = np.searchsorted(d.t.to_numpy(), boundary)
                    left = d.iloc[k - 1]
                    right = d.iloc[k]

                    if right.t - left.t > 5:
                        continue

                    row = {'t': boundary}

                    for c in [
                        'lat', 'lon', 'speed', 'roll_deg',
                        'pitch_deg', 'yaw_deg', 'altitude_m'
                    ]:
                        if c in d:
                            x, y = pd.to_numeric(
                                pd.Series([left[c], right[c]]),
                                errors='coerce'
                            )
                            row[c] = x + (y - x) * (
                                boundary - left.t
                            ) / (right.t - left.t)

                    g = pd.concat(
                        [g, pd.DataFrame([row])],
                        ignore_index=True
                    )
        else:
            g = d[d.lap == n].copy()

        g = g.sort_values('t').drop_duplicates('t')

        if len(g) < 2:
            continue

        lat = np.radians(g.lat.to_numpy())
        lon = np.radians(g.lon.to_numpy())

        h = (
            np.sin(np.diff(lat) / 2) ** 2
            + np.cos(lat[:-1]) * np.cos(lat[1:])
            * np.sin(np.diff(lon) / 2) ** 2
        )

        ds = 2 * 6371000 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

        g['distance'] = np.r_[0, np.cumsum(ds)]
        g['elapsed'] = g.t - a

        if 'speed' not in g:
            g['speed'] = np.r_[np.nan, ds / np.diff(g.t) * 3.6]

        g['speed'] = g.speed.where(g.speed.between(0, 500))

        traces[n] = g

    table = pd.DataFrame(
        records,
        columns=['Volta', 'inicio', 'fim', 'Tempo (s)']
    )

    if table.empty:
        raise ValueError('Nenhuma volta disponível.')

    table = table.sort_values('Volta').reset_index(drop=True)

    table['Máxima (km/h)'] = [
        traces[n].speed.max() if n in traces else np.nan
        for n in table.Volta
    ]

    table['Média amostral (km/h)'] = [
        traces[n].speed.mean() if n in traces else np.nan
        for n in table.Volta
    ]

    table['Delta (s)'] = table['Tempo (s)'] - table['Tempo (s)'].min()

    return d, table, traces, official


def main():
    st.set_page_config(
        page_title='RaceMetrics | Lap Analyzer',
        page_icon='🏁',
        layout='wide'
    )

    st.title('🏁 RaceMetrics | Lap Analyzer')
    st.caption('Telemetria real • comparação de voltas • análise de desempenho')

    c1, c2, c3 = st.columns(3)

    csv = c1.file_uploader('telemetry.csv', type='csv')
    lf = c2.file_uploader('laps.json (opcional)', type='json')
    mf = c3.file_uploader('metadata.json (opcional)', type='json')

    if csv is None:
        st.info('Carregue os arquivos exportados pelo RaceMetrics. CSV legado também é aceito.')
        return

    try:
        raw = read_csv(csv)
        meta = json.load(mf) if mf else {}
        laps = json.load(lf) if lf else None

        d, table, traces, official = prepare(raw, laps, meta)

    except (ValueError, KeyError, TypeError) as e:
        st.error(str(e))
        return

    best = int(table.loc[table['Tempo (s)'].idxmin(), 'Volta'])
    bt = table['Tempo (s)'].min()

    if not official:
        st.warning(
            'CSV genérico: tempos estimados. Volta 0 e última volta '
            'excluídas por precaução; sem confirmação de conclusão.'
        )
    else:
        st.success(
            'Tempos oficiais importados de laps.json; somente voltas marcadas como válidas.'
            if lf else
            'Tempos recuperados de ultima_volta_s do CSV RaceMetrics; '
            'somente voltas com conclusão confirmada.'
        )

        if meta.get('lapCount') is not None and meta['lapCount'] != len(table):
            st.warning('Quantidade de voltas diverge dos metadados.')

        if meta.get('bestLapTimeMs') is not None and abs(
            meta['bestLapTimeMs'] / 1000 - bt
        ) > 0.001:
            st.warning('Melhor tempo diverge dos metadados.')

    st.caption(
        f"Pista: {meta.get('track', 'Não informada')} | "
        f"Sessão: {meta.get('id', 'CSV')} | "
        f"Velocidade: {'registrada pelo dispositivo' if 'speed' in d else 'estimada pelas coordenadas GPS'}"
    )

    a, b, c, e = st.columns(4)

    a.metric('Melhor volta', f'{bt:.3f} s', f'V{best}', delta_color='off')
    b.metric('Voltas', len(table))
    c.metric('Amostras válidas', len(d))

    e.metric(
        'Velocidade máxima',
        f'{d.speed.max():.2f} km/h'
        if 'speed' in d else
        f"{table['Máxima (km/h)'].max():.2f} km/h"
    )

    # Histórico de voltas
    st.subheader('Histórico de voltas')

    st.dataframe(
        table.drop(columns=['inicio', 'fim']).round(3),
        hide_index=True,
        use_container_width=True
    )

    # Gráfico de tempo por volta em linhas
    st.plotly_chart(
        go.Figure(
            go.Scatter(
                x=table.Volta.astype(str),
                y=table['Tempo (s)'],
                mode='lines+markers',
                line=dict(color='#00d084', width=3),
                marker=dict(
                    size=9,
                    color=[
                        '#00d084' if n == best else '#64748b'
                        for n in table.Volta
                    ]
                )
            )
        ).update_layout(
            xaxis_title='Volta',
            yaxis_title='Tempo (s)',
            height=320
        ),
        use_container_width=True
    )

    choices = [
        int(n) for n in table.Volta
        if n in traces
    ]

    if best not in choices:
        st.warning('Melhor volta sem telemetria suficiente.')
        return

    other = [n for n in choices if n != best]

    st.subheader('Comparação de telemetria — estilo F1')

    selected = st.selectbox(
        'Comparar melhor volta com',
        other or [best],
        format_func=lambda n: f'Volta {n}'
    )

    mode = st.radio(
        'Eixo horizontal',
        ['Distância (m)', 'Progresso da volta (%)'],
        horizontal=True
    )

    ref = traces[best]
    comp = traces[selected]

    def axis(g):
        x = g.distance.to_numpy()

        return (
            x if mode == 'Distância (m)'
            else x / x[-1] * 100 if x[-1] > 0
            else x
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[.72, .28],
        vertical_spacing=.08
    )

    for n, g, color in [
        (best, ref, '#00d084'),
        (selected, comp, '#ff664d')
    ]:
        if n == best and color == '#ff664d':
            continue

        duration = table.loc[
            table.Volta == n, 'Tempo (s)'
        ].iloc[0]

        fig.add_trace(
            go.Scatter(
                x=axis(g),
                y=g.speed,
                name=f'V{n} · {duration:.3f} s',
                mode='lines+markers',
                marker_size=3,
                line=dict(color=color, width=2.5)
            ),
            row=1,
            col=1
        )

    x1, x2 = axis(ref), axis(comp)

    u1, i1 = np.unique(x1, return_index=True)
    u2, i2 = np.unique(x2, return_index=True)

    if min(u1[-1], u2[-1]) > 0:
        x = np.linspace(0, min(u1[-1], u2[-1]), 200)

        delta = (
            np.interp(x, u2, comp.elapsed.to_numpy()[i2])
            - np.interp(x, u1, ref.elapsed.to_numpy()[i1])
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=delta,
                name='Delta acumulado',
                line=dict(color='#60a5fa')
            ),
            row=2,
            col=1
        )

    fig.add_hline(y=0, row=2, col=1, line_dash='dot')

    fig.update_yaxes(title_text='km/h', row=1, col=1)
    fig.update_yaxes(title_text='Delta (s)', row=2, col=1)
    fig.update_xaxes(title_text=mode, row=2, col=1)

    fig.update_layout(
        height=600,
        hovermode='x unified',
        legend=dict(orientation='h', y=1.08)
    )

    st.plotly_chart(fig, use_container_width=True)

    gap = (
        table.loc[table.Volta == selected, 'Tempo (s)'].iloc[0]
        - bt
    )

    st.metric(
        'Diferença oficial de tempo' if official else 'Diferença estimada',
        f'+{gap:.3f} s'
    )

    st.caption(
        'Delta positivo: volta comparada atrás da referência. '
        'Distância acumulada por GPS é aproximada; curvas podem não '
        'representar exatamente o mesmo ponto físico. Limites '
        'interpolados quando há amostras adjacentes; não há extrapolação. '
        'Progresso normalizado também é aproximado.'
    )

    # Mapa da volta selecionada
    st.subheader('Mapa da volta selecionada')

    g = comp
    points = g[['lat', 'lon']].values.tolist()

    m = folium.Map(
        location=points[0],
        zoom_start=16,
        control_scale=True
    )

    for i in range(1, len(g)):
        v = g.speed.iloc[i]

        color = (
            '#888888' if pd.isna(v)
            else '#2563eb' if v < 30
            else '#16a34a' if v < 60
            else '#eab308' if v < 90
            else '#ef4444'
        )

        folium.PolyLine(
            points[i - 1:i + 1],
            color=color,
            weight=5,
            tooltip=f'{v:.1f} km/h' if pd.notna(v) else 'Sem velocidade'
        ).add_to(m)

    track = meta.get('trackSnapshot', {})

    if all(k in track for k in ('latA', 'lonA', 'latB', 'lonB')):
        folium.PolyLine(
            [
                [track['latA'], track['lonA']],
                [track['latB'], track['lonB']]
            ],
            color='black',
            weight=5,
            tooltip='Linha de chegada'
        ).add_to(m)

    m.fit_bounds(points)

    st_folium(
        m,
        height=450,
        use_container_width=True,
        returned_objects=[]
    )

    st.caption(
        'Azul: <30 km/h | Verde: 30–60 | Amarelo: 60–90 | '
        'Vermelho: ≥90. Mapa-base depende de internet.'
    )

    # Sensores e qualidade GPS
    st.subheader('Sensores e qualidade GPS')

    sensores = {
        'roll_deg': {
            'nome': 'Roll - Inclinação lateral',
            'unidade': 'Graus (°)',
            'explicacao': (
                'Mede a inclinação lateral do veículo. Durante as curvas, '
                'pode variar devido à transferência de peso e à inclinação '
                'do veículo. Mudanças bruscas podem indicar curvas mais '
                'agressivas ou irregularidades na pista.'
            )
        },
        'pitch_deg': {
            'nome': 'Pitch - Inclinação longitudinal',
            'unidade': 'Graus (°)',
            'explicacao': (
                'Mede a inclinação do veículo para frente ou para trás. '
                'Pode variar durante acelerações, frenagens, subidas, '
                'descidas e irregularidades do terreno.'
            )
        },
        'yaw_deg': {
            'nome': 'Yaw - Orientação e rotação',
            'unidade': 'Graus (°)',
            'explicacao': (
                'Representa a orientação ou rotação do veículo em torno '
                'do eixo vertical. Varia durante curvas e mudanças de '
                'direção. Oscilações podem indicar mudanças rápidas '
                'de trajetória.'
            )
        },
        'altitude_m': {
            'nome': 'Altitude',
            'unidade': 'Metros (m)',
            'explicacao': (
                'Indica a altitude estimada pelo GPS. Pode mostrar '
                'subidas e descidas da pista, mas também sofre influência '
                'da precisão do sinal GPS.'
            )
        },
        'hdop': {
            'nome': 'HDOP - Precisão horizontal do GPS',
            'unidade': 'Índice HDOP',
            'explicacao': (
                'Indica a qualidade da geometria dos satélites para '
                'determinar a posição horizontal. Quanto menor o HDOP, '
                'melhor tende a ser a precisão. Valores altos indicam '
                'maior possibilidade de erro na localização.'
            )
        },
        'satelites': {
            'nome': 'Satélites GPS',
            'unidade': 'Quantidade',
            'explicacao': (
                'Mostra a quantidade de satélites utilizados ou recebidos '
                'pelo GPS. Em geral, mais satélites ajudam na localização, '
                'mas a quantidade sozinha não garante precisão.'
            )
        }
    }

    for k, info in sensores.items():
        if k in d.columns:
            st.markdown(f"**{info['nome']}**")

            valores = pd.to_numeric(d[k], errors='coerce')

            grafico = go.Figure(
                go.Scatter(
                    x=d.t - d.t.min(),
                    y=valores,
                    mode='lines',
                    name=info['nome'],
                    line=dict(width=2)
                )
            )

            grafico.update_layout(
                height=330,
                xaxis_title='Tempo da sessão (s)',
                yaxis_title=info['unidade'],
                margin=dict(t=25, b=45)
            )

            st.plotly_chart(
                grafico,
                use_container_width=True
            )

            st.caption(info['explicacao'])
            st.divider()


if __name__ == '__main__':
    main()
