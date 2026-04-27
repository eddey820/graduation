"""
=========================================================
 졸업논문 분석 코드 - Step 5: 환승 네트워크 분석 (RQ3)
 목적:
   - 서울 지하철 환승 네트워크 그래프 구축
   - 중심성 분석 (Degree, PageRank, Betweenness, Closeness)
   - 중심성 × 혼잡도 결합 → "핵심 병목역" 식별
   - 군집별 네트워크 위치 차이 분석
=========================================================
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 0. 한글 폰트 설정
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {"font.family": "Pretendard"})

PROJECT_ROOT = Path("/Users/donghyunkim/Desktop/김동현/2026년 1학기/졸업논문/thesis")
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = PROJECT_ROOT / "output" / "figures"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
CONGESTION_FILE = DATA_DIR / "congestion" / "서울교통공사_지하철혼잡도정보_20260331.csv"


# ---------------------------------------------------------
# 1. 역명 표준화 (Step 2와 동일)
# ---------------------------------------------------------
def standardize_station(name):
    if pd.isna(name):
        return name
    s = str(name).strip()
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'[ES]$', '', s)
    return s.strip()


# ---------------------------------------------------------
# 2. 데이터 로드
# ---------------------------------------------------------
print("█" * 70)
print("█  Step 5: 환승 네트워크 분석")
print("█" * 70)

# 혼잡도 원본에서 역번호 정보 활용
df_raw = pd.read_csv(CONGESTION_FILE, encoding="cp949", low_memory=False)
df_raw['역명_표준'] = df_raw['출발역'].apply(standardize_station)

# 군집 정보
profile = pd.read_csv(PROCESSED_DIR / "혼잡도_프로파일_평일_군집포함.csv")
station_cluster = (profile
    .sort_values('전체_평균', ascending=False)
    .drop_duplicates('역명_표준', keep='first'))
station_cluster_dict = dict(zip(station_cluster['역명_표준'], station_cluster['cluster']))
station_congestion_dict = dict(zip(station_cluster['역명_표준'], station_cluster['전체_평균']))


# ---------------------------------------------------------
# 3. 네트워크 그래프 구축
#    - 노드: 역
#    - 엣지: 같은 호선 내 인접한 역 + 환승역(여러 호선이 같은 역에 있음)
# ---------------------------------------------------------
print("\n" + "─" * 70)
print("  네트워크 그래프 구축")
print("─" * 70)

G = nx.Graph()  # 무방향 그래프

# (1) 호선별 인접 역 연결
for line in ['1호선', '2호선', '3호선', '4호선', '5호선', '6호선', '7호선', '8호선']:
    line_df = (df_raw[df_raw['호선'] == line]
               .drop_duplicates(['역번호', '역명_표준'])
               .sort_values('역번호'))
    # 9000번대(지선) 제거 — 메인 노선만 우선
    line_main = line_df[line_df['역번호'] < 9000].copy()
    stations_in_line = line_main['역명_표준'].tolist()

    # 노드 추가 (호선 정보 포함)
    for st in stations_in_line:
        if not G.has_node(st):
            G.add_node(st, lines=set([line]))
        else:
            G.nodes[st]['lines'].add(line)

    # 인접 엣지 추가
    for i in range(len(stations_in_line) - 1):
        a, b = stations_in_line[i], stations_in_line[i + 1]
        G.add_edge(a, b, line=line, edge_type='line')

    # 2호선 순환선 — 마지막↔첫 역 연결
    if line == '2호선' and len(stations_in_line) > 2:
        G.add_edge(stations_in_line[0], stations_in_line[-1],
                   line='2호선', edge_type='loop')

# (2) 환승 엣지 — 같은 역에 호선이 2개 이상 → 자기 자신이므로 별도 엣지 불필요
#    (이미 노드에 lines 속성으로 환승 표시됨)
# 단, 분기점 처리: 성수지선, 마천지선
print(f"\n  ▶ 노드(역) 수: {G.number_of_nodes()}")
print(f"  ▶ 엣지(연결) 수: {G.number_of_edges()}")
print(f"  ▶ 연결 컴포넌트: {nx.number_connected_components(G)}")

# 가장 큰 컴포넌트만 사용 (격리된 노드 제거)
largest_cc = max(nx.connected_components(G), key=len)
G_main = G.subgraph(largest_cc).copy()
print(f"  ▶ 최대 컴포넌트 크기: {G_main.number_of_nodes()}역")

# 환승역 (lines 속성에서 호선 수 ≥ 2)
transfer_stations = [n for n, d in G_main.nodes(data=True)
                     if len(d.get('lines', set())) >= 2]
print(f"  ▶ 환승역 수: {len(transfer_stations)}")


# ---------------------------------------------------------
# 4. 중심성 분석 (4가지)
# ---------------------------------------------------------
print("\n" + "─" * 70)
print("  중심성 지표 계산")
print("─" * 70)

# Degree: 인접 역 수 (환승역은 큼)
degree_cent = nx.degree_centrality(G_main)

# Betweenness: 최단경로 통과 빈도 — "병목"의 직접 측정
print("  Betweenness 계산 중... (수십 초 소요)")
betweenness_cent = nx.betweenness_centrality(G_main, normalized=True)

# Closeness: 모든 역까지의 평균 최단거리의 역수 — 접근성
closeness_cent = nx.closeness_centrality(G_main)

# PageRank: 영향력 기반 중심성
pagerank = nx.pagerank(G_main, alpha=0.85)

print("  ✓ 4가지 중심성 계산 완료")


# ---------------------------------------------------------
# 5. 중심성 + 혼잡도 + 군집 통합 테이블
# ---------------------------------------------------------
centrality_df = pd.DataFrame({
    '역명': list(G_main.nodes()),
    'degree': [degree_cent[n] for n in G_main.nodes()],
    'betweenness': [betweenness_cent[n] for n in G_main.nodes()],
    'closeness': [closeness_cent[n] for n in G_main.nodes()],
    'pagerank': [pagerank[n] for n in G_main.nodes()],
    '환승역': [len(G_main.nodes[n].get('lines', set())) >= 2
                for n in G_main.nodes()],
    '호선수': [len(G_main.nodes[n].get('lines', set()))
                for n in G_main.nodes()],
    '혼잡도': [station_congestion_dict.get(n, np.nan) for n in G_main.nodes()],
    '군집': [station_cluster_dict.get(n, -1) for n in G_main.nodes()]
})

# 혼잡도 데이터가 없는 역(매우 한산) 제거
centrality_df = centrality_df[centrality_df['혼잡도'].notna()].reset_index(drop=True)
print(f"\n  분석 대상 역: {len(centrality_df)}개")


# ---------------------------------------------------------
# 6. 핵심 병목역 식별 — 복합 지표(Bottleneck Index)
#    BI = z(betweenness) + z(혼잡도)
#    구조적 중요도와 실제 혼잡이 모두 높은 역
# ---------------------------------------------------------
def zscore(x):
    return (x - x.mean()) / x.std()

centrality_df['z_betweenness'] = zscore(centrality_df['betweenness'])
centrality_df['z_pagerank'] = zscore(centrality_df['pagerank'])
centrality_df['z_혼잡도'] = zscore(centrality_df['혼잡도'])

# Bottleneck Index = 매개중심성 + 혼잡도
centrality_df['BI_매개혼잡'] = centrality_df['z_betweenness'] + centrality_df['z_혼잡도']
# Critical Hub Index = PageRank + 혼잡도
centrality_df['CHI_영향력혼잡'] = centrality_df['z_pagerank'] + centrality_df['z_혼잡도']

print("\n" + "─" * 70)
print("  핵심 병목역 Top 15 (Bottleneck Index = z매개중심성 + z혼잡도)")
print("─" * 70)
top_bn = centrality_df.nlargest(15, 'BI_매개혼잡')[
    ['역명', '환승역', '호선수', 'betweenness', '혼잡도', 'BI_매개혼잡', '군집']
]
top_bn.columns = ['역명', '환승역', '호선수', '매개중심성', '혼잡도', 'BI', '군집']
print(top_bn.to_string(index=False))

centrality_df.to_csv(TABLE_DIR / "table06_centrality_full.csv",
                     index=False, encoding='utf-8-sig')
top_bn.to_csv(TABLE_DIR / "table07_top_bottleneck.csv",
              index=False, encoding='utf-8-sig')


# ---------------------------------------------------------
# 7. 환승역 vs 비환승역 중심성 비교
# ---------------------------------------------------------
print("\n" + "─" * 70)
print("  환승역 vs 비환승역 비교")
print("─" * 70)
compare = (centrality_df
    .groupby('환승역')
    .agg(역수=('역명', 'count'),
         평균_매개중심성=('betweenness', 'mean'),
         평균_PageRank=('pagerank', 'mean'),
         평균_혼잡도=('혼잡도', 'mean'),
         평균_BI=('BI_매개혼잡', 'mean'))
    .round(4))
print(compare.to_string())


# ---------------------------------------------------------
# 8. 군집별 중심성 분포
# ---------------------------------------------------------
print("\n" + "─" * 70)
print("  군집별 중심성 평균 (어떤 군집이 네트워크상 더 중심적인가?)")
print("─" * 70)

CLUSTER_NAMES = {
    0: "전일 분산형 · 도심·관광 (중혼잡)",
    1: "출근 집중형 · 주거지 (고혼잡)",
    2: "출근 집중형 · 주거지 (중혼잡)",
    3: "전일 분산형 · 도심·관광 (고혼잡)"
}
CLUSTER_SHORT = {0: "도심분산-중", 1: "주거첨두-고",
                 2: "주거첨두-중", 3: "도심분산-고"}
CLUSTER_COLORS = {0: '#3498DB', 1: '#E74C3C', 2: '#27AE60', 3: '#F39C12'}

cluster_central = (centrality_df[centrality_df['군집'] >= 0]
    .groupby('군집')
    .agg(역수=('역명', 'count'),
         평균_매개중심성=('betweenness', 'mean'),
         평균_PageRank=('pagerank', 'mean'),
         평균_혼잡도=('혼잡도', 'mean'),
         평균_BI=('BI_매개혼잡', 'mean'),
         환승역_비율=('환승역', lambda x: x.mean() * 100))
    .round(4))
cluster_central['군집명'] = [CLUSTER_NAMES[c] for c in cluster_central.index]
print(cluster_central.to_string())
cluster_central.to_csv(TABLE_DIR / "table08_centrality_by_cluster.csv",
                      encoding='utf-8-sig')


# ---------------------------------------------------------
# 9. 그림 8: 중심성 vs 혼잡도 산점도
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# (1) Betweenness × 혼잡도
ax = axes[0]
for c in [0, 1, 2, 3]:
    sub = centrality_df[centrality_df['군집'] == c]
    ax.scatter(sub['betweenness'], sub['혼잡도'],
               c=CLUSTER_COLORS[c], alpha=0.7, s=70,
               edgecolors='white', linewidths=0.8,
               label=f"C{c}: {CLUSTER_SHORT[c]}")

# 상위 병목역 라벨링
top_label = centrality_df.nlargest(8, 'BI_매개혼잡')
for _, row in top_label.iterrows():
    ax.annotate(row['역명'],
                (row['betweenness'], row['혼잡도']),
                fontsize=9, fontweight='bold',
                xytext=(7, 7), textcoords='offset points')

ax.set_xlabel('매개중심성 (Betweenness Centrality)', fontsize=11)
ax.set_ylabel('평균 혼잡도 (%)', fontsize=11)
ax.set_title('매개중심성 vs 혼잡도 — 핵심 병목역 식별',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax.grid(alpha=0.3)

# (2) PageRank × 혼잡도
ax = axes[1]
for c in [0, 1, 2, 3]:
    sub = centrality_df[centrality_df['군집'] == c]
    ax.scatter(sub['pagerank'], sub['혼잡도'],
               c=CLUSTER_COLORS[c], alpha=0.7, s=70,
               edgecolors='white', linewidths=0.8,
               label=f"C{c}: {CLUSTER_SHORT[c]}")

top_label2 = centrality_df.nlargest(8, 'CHI_영향력혼잡')
for _, row in top_label2.iterrows():
    ax.annotate(row['역명'],
                (row['pagerank'], row['혼잡도']),
                fontsize=9, fontweight='bold',
                xytext=(7, 7), textcoords='offset points')

ax.set_xlabel('PageRank', fontsize=11)
ax.set_ylabel('평균 혼잡도 (%)', fontsize=11)
ax.set_title('PageRank vs 혼잡도 — 핵심 영향력 역 식별',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig08_centrality_vs_congestion.png", bbox_inches='tight')
plt.close()
print(f"\n  ✓ 저장: fig08_centrality_vs_congestion.png")


# ---------------------------------------------------------
# 10. 그림 9: 군집별 중심성 분포 (boxplot)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = [('betweenness', '매개중심성'), ('pagerank', 'PageRank'),
           ('혼잡도', '혼잡도(%)')]

for ax, (col, label) in zip(axes, metrics):
    data_to_plot = []
    labels = []
    colors = []
    for c in [0, 1, 2, 3]:
        sub = centrality_df[centrality_df['군집'] == c][col].dropna()
        data_to_plot.append(sub.values)
        labels.append(f"C{c}\n{CLUSTER_SHORT[c]}")
        colors.append(CLUSTER_COLORS[c])

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    widths=0.6, showfliers=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_ylabel(label, fontsize=10)
    ax.grid(alpha=0.3, axis='y')

plt.suptitle('군집별 네트워크 중심성 및 혼잡도 분포',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig09_centrality_by_cluster.png", bbox_inches='tight')
plt.close()
print(f"  ✓ 저장: fig09_centrality_by_cluster.png")


# ---------------------------------------------------------
# 11. 그림 10: 네트워크 시각화
#     - kamada-kawai 레이아웃 (선형성 보존)
#     - 노드 크기 = 혼잡도
#     - 노드 색 = 군집
#     - Top 병목역 라벨 + 강조 표시
# ---------------------------------------------------------
print("\n  네트워크 레이아웃 계산 중...(kamada-kawai)")
pos = nx.kamada_kawai_layout(G_main)

fig, ax = plt.subplots(figsize=(16, 13))

# 엣지 (호선별 색상으로 그리면 더 직관적)
LINE_EDGE_COLORS = {
    '1호선': '#0052A4', '2호선': '#00A84D', '3호선': '#EF7C1C',
    '4호선': '#00A5DE', '5호선': '#996CAC', '6호선': '#CD7C2F',
    '7호선': '#747F00', '8호선': '#E6186C'
}
for u, v, data in G_main.edges(data=True):
    line = data.get('line', None)
    color = LINE_EDGE_COLORS.get(line, 'gray')
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=color, alpha=0.5, linewidth=1.5, zorder=1)

# 노드
node_list = list(G_main.nodes())
node_colors, node_sizes, node_alphas = [], [], []
for n in node_list:
    cluster = station_cluster_dict.get(n, -1)
    if cluster in CLUSTER_COLORS:
        node_colors.append(CLUSTER_COLORS[cluster])
        node_alphas.append(0.85)
    else:
        node_colors.append('#BDC3C7')
        node_alphas.append(0.4)
    cong = station_congestion_dict.get(n, 0)
    node_sizes.append(80 + cong * 25)

# 일반 노드
for i, n in enumerate(node_list):
    ax.scatter(pos[n][0], pos[n][1],
               c=[node_colors[i]], s=node_sizes[i],
               alpha=node_alphas[i],
               edgecolors='white', linewidths=1.0, zorder=2)

# Top 병목역은 검은 테두리로 강조
top15_names = centrality_df.nlargest(15, 'BI_매개혼잡')['역명'].tolist()
for n in top15_names:
    if n in pos:
        cong = station_congestion_dict.get(n, 0)
        size = 80 + cong * 25
        ax.scatter(pos[n][0], pos[n][1],
                   facecolors='none', edgecolors='black',
                   s=size + 200, linewidths=2.5, zorder=3)

# 라벨 — Top 병목역만
for n in top15_names:
    if n in pos:
        ax.annotate(n, pos[n], fontsize=11, fontweight='bold',
                    xytext=(8, 8), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', edgecolor='black',
                              alpha=0.85),
                    zorder=4)

# 범례 - 군집 + 호선
from matplotlib.lines import Line2D
cluster_legend = [
    Line2D([0], [0], marker='o', color='w', markersize=14,
           markerfacecolor=CLUSTER_COLORS[c],
           markeredgecolor='white', markeredgewidth=1.5,
           label=f"C{c}: {CLUSTER_SHORT[c]}")
    for c in [0, 1, 2, 3]
]
top_legend = [Line2D([0], [0], marker='o', color='w', markersize=14,
                     markerfacecolor='none', markeredgecolor='black',
                     markeredgewidth=2.5, label='핵심 병목역 Top 15')]
line_legend = [Line2D([0], [0], color=LINE_EDGE_COLORS[l],
                      linewidth=2.5, label=l)
               for l in ['1호선', '2호선', '3호선', '4호선',
                         '5호선', '6호선', '7호선', '8호선']]

leg1 = ax.legend(handles=cluster_legend + top_legend,
                 loc='upper left', fontsize=10, framealpha=0.95,
                 title='역사 군집', title_fontsize=11)
ax.add_artist(leg1)
ax.legend(handles=line_legend, loc='upper right', fontsize=9,
          framealpha=0.95, title='호선', title_fontsize=11, ncol=2)

ax.set_title('서울 지하철 환승 네트워크 지도\n'
             '(노드 크기 = 평균 혼잡도, 색상 = 군집 유형, 검은 테두리 = 핵심 병목역)',
             fontsize=14, fontweight='bold', pad=15)
ax.axis('off')
plt.tight_layout()
plt.savefig(FIG_DIR / "fig10_network_map.png", bbox_inches='tight')
plt.close()
print(f"  ✓ 저장: fig10_network_map.png")


# ---------------------------------------------------------
# 12. 완료 요약
# ---------------------------------------------------------
print("\n\n" + "█" * 70)
print("█  Step 5 완료")
print("█" * 70)
print(f"""
  네트워크 통계:
    - 노드 수: {G_main.number_of_nodes()}
    - 엣지 수: {G_main.number_of_edges()}
    - 환승역 수: {len(transfer_stations)}
    - 평균 차수: {2 * G_main.number_of_edges() / G_main.number_of_nodes():.2f}
    - 지름(diameter): {nx.diameter(G_main)}
    - 평균 최단경로: {nx.average_shortest_path_length(G_main):.2f}

  생성된 산출물:
    [그림]
      - fig08_centrality_vs_congestion.png  (병목역 식별 산점도)
      - fig09_centrality_by_cluster.png     (군집별 중심성 분포)
      - fig10_network_map.png               (네트워크 지도)
    [표]
      - table06_centrality_full.csv         (전체 역 중심성)
      - table07_top_bottleneck.csv          (Top 15 병목역)
      - table08_centrality_by_cluster.csv   (군집별 중심성 평균)
""")
