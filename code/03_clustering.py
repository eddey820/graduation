"""
=========================================================
 졸업논문 분석 코드 - Step 3: 시간대별 혼잡 패턴 군집화 (RQ1)
 목적:
   - 평일 30분 단위 혼잡도 프로파일을 K-means로 군집화
   - 최적 K 결정 (Elbow + Silhouette)
   - 각 군집의 시공간 패턴 해석
   - 군집별 대표 역, 호선 분포, 평균 패턴 시각화
=========================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 0. 한글 폰트 설정
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {"font.family": "Pretendard"})

# ---------------------------------------------------------
# 1. 경로
# ---------------------------------------------------------
PROJECT_ROOT = Path("/Users/donghyunkim/Desktop/김동현/2026년 1학기/졸업논문/thesis")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "output" / "figures"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"

# ---------------------------------------------------------
# 2. 데이터 로드
# ---------------------------------------------------------
print("█" * 70)
print("█  Step 3: 혼잡 패턴 군집화 분석")
print("█" * 70)

profile = pd.read_csv(PROCESSED_DIR / "혼잡도_프로파일_평일.csv")
time_cols = [c for c in profile.columns
             if '시' in str(c) and '분' in str(c)
             and not any(kw in c for kw in ['첨두', '시간', '평균', '집중', '최대'])]
print(f"\n  로드: {len(profile)}개 (호선-역 조합), 시간 컬럼 {len(time_cols)}개")

# 매우 한산한 역(평균 5% 미만) 제외 — 패턴 분석 의미 없음
profile_active = profile[profile['전체_평균'] > 5].reset_index(drop=True)
print(f"  활성 역(평균>5%): {len(profile_active)}개")


# ---------------------------------------------------------
# 3. 입력 행렬 생성: 패턴 모양만 비교 (max 정규화)
# ---------------------------------------------------------
X_raw = profile_active[time_cols].values
X_norm = X_raw / (X_raw.max(axis=1, keepdims=True) + 1e-9)
print(f"  입력 행렬 shape: {X_norm.shape}")


# ---------------------------------------------------------
# 4. 최적 K 탐색: Elbow + Silhouette
# ---------------------------------------------------------
print("\n" + "─" * 70)
print("  최적 클러스터 수 탐색")
print("─" * 70)

K_range = range(2, 9)
inertias, silhouettes = [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_norm)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_norm, labels)
    silhouettes.append(sil)
    print(f"  K={k}: inertia={km.inertia_:>7.1f}, silhouette={sil:.4f}")

# 그림 1: Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(list(K_range), inertias, 'o-', color='#2C3E50', linewidth=2, markersize=8)
axes[0].set_xlabel('클러스터 수 (K)', fontsize=11)
axes[0].set_ylabel('Inertia (군집 내 총 분산)', fontsize=11)
axes[0].set_title('Elbow Method', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].plot(list(K_range), silhouettes, 'o-', color='#E74C3C', linewidth=2, markersize=8)
axes[1].set_xlabel('클러스터 수 (K)', fontsize=11)
axes[1].set_ylabel('Silhouette Score', fontsize=11)
axes[1].set_title('Silhouette Score', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig01_optimal_k.png", bbox_inches='tight')
plt.close()
print(f"  ✓ 저장: fig01_optimal_k.png")


# ---------------------------------------------------------
# 5. 본 분석: K=4로 군집화 (해석 용이성 + Elbow 근방)
# ---------------------------------------------------------
print("\n" + "─" * 70)
print("  최종 K=4 군집화")
print("─" * 70)

K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=20)
profile_active['cluster'] = km.fit_predict(X_norm)

# 군집별 대표 패턴 (centroid)
centroids = km.cluster_centers_
print(f"\n  군집별 역 수:")
print(profile_active['cluster'].value_counts().sort_index().to_string())


# ---------------------------------------------------------
# 6. 군집 라벨링 (해석 가능한 이름 부여)
# ---------------------------------------------------------
def label_cluster(centroid, time_cols):
    """centroid 패턴을 보고 자동 라벨링 (정량 지표 반환)"""
    morning_idx = [i for i, c in enumerate(time_cols) if c.startswith(('7시', '8시', '9시'))]
    evening_idx = [i for i, c in enumerate(time_cols) if c.startswith(('18시', '19시', '20시'))]
    midday_idx = [i for i, c in enumerate(time_cols) if c.startswith(('12시', '13시', '14시'))]

    morning = centroid[morning_idx].mean()
    evening = centroid[evening_idx].mean()
    midday = centroid[midday_idx].mean()
    peak_hour_idx = centroid.argmax()
    peak_time = time_cols[peak_hour_idx]
    # 첨두 집중도: 첨두 평균 / 비첨두 평균
    nonpeak_mean = (centroid.sum() - centroid[morning_idx].sum() - centroid[evening_idx].sum()) / (
        len(centroid) - len(morning_idx) - len(evening_idx))
    peak_ratio = ((morning + evening) / 2) / (nonpeak_mean + 1e-9)
    return {
        'morning': morning, 'evening': evening, 'midday': midday,
        'peak_time': peak_time, 'peak_ratio': peak_ratio,
        'asymmetry': morning - evening  # +면 아침 우세, -면 저녁 우세
    }

# 각 군집의 특성 추출 후 상대적 라벨링
cluster_metrics = {c: label_cluster(centroids[c], time_cols) for c in range(K)}

# 군집별 실제 혼잡도 평균 (라벨에 사용)
cluster_mean_congestion = {
    c: profile_active.loc[profile_active['cluster'] == c, '전체_평균'].mean()
    for c in range(K)
}

# 군집 간 상대 비교로 라벨링
peak_ratios = [cluster_metrics[c]['peak_ratio'] for c in range(K)]
asymmetries = [cluster_metrics[c]['asymmetry'] for c in range(K)]
peak_ratio_median = np.median(peak_ratios)
congestion_median = np.median(list(cluster_mean_congestion.values()))

cluster_info = {}
for c in range(K):
    m = cluster_metrics[c]
    high_peak = m['peak_ratio'] > peak_ratio_median
    morning_strong = m['asymmetry'] > 0.05
    evening_strong = m['asymmetry'] < -0.05
    high_congestion = cluster_mean_congestion[c] > congestion_median

    # 패턴 형태 라벨
    if high_peak and morning_strong:
        shape = "출근 집중형"
        interpretation = "주거지·베드타운"
    elif high_peak and evening_strong:
        shape = "퇴근 집중형"
        interpretation = "업무지구"
    elif high_peak:
        shape = "출퇴근 양극형"
        interpretation = "환승·중간기점"
    elif morning_strong:
        shape = "아침 우세 분산형"
        interpretation = "복합 주거"
    elif evening_strong:
        shape = "저녁 우세 분산형"
        interpretation = "상업·복합"
    else:
        shape = "전일 분산형"
        interpretation = "도심·관광지"

    # 혼잡도 수준 한정자
    level = "고혼잡" if high_congestion else "중혼잡"
    name = f"{shape} · {interpretation} ({level})"

    cluster_info[c] = {
        'name': name,
        'short_name': shape,
        'size': (profile_active['cluster'] == c).sum(),
        'mean_congestion': cluster_mean_congestion[c],
        **m
    }
    print(f"\n  [Cluster {c}] '{name}' ({cluster_info[c]['size']}개 역)")
    print(f"    평균 혼잡도: {cluster_mean_congestion[c]:.2f}%")
    print(f"    아침={m['morning']:.2f}, 낮={m['midday']:.2f}, 저녁={m['evening']:.2f}")
    print(f"    첨두집중도={m['peak_ratio']:.2f}, 비대칭성(아침-저녁)={m['asymmetry']:+.3f}")
    print(f"    최대 혼잡 시간대: {m['peak_time']}")


# ---------------------------------------------------------
# 7. 그림 2: 군집별 평균 시간 패턴
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12']

for c in range(K):
    ax = axes[c // 2, c % 2]
    # 해당 군집의 모든 역 (회색)
    cluster_data = X_norm[profile_active['cluster'] == c]
    for row in cluster_data:
        ax.plot(range(len(time_cols)), row, color='lightgray', alpha=0.3, linewidth=0.5)
    # 군집 평균 (centroid)
    ax.plot(range(len(time_cols)), centroids[c], color=colors[c],
            linewidth=2.8, label=f'군집 평균')
    ax.fill_between(range(len(time_cols)), centroids[c], alpha=0.15, color=colors[c])

    ax.set_title(f"Cluster {c}: {cluster_info[c]['short_name']}\n({cluster_info[c]['size']}역, 평균 {cluster_info[c]['mean_congestion']:.1f}%)",
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('정규화 혼잡도 (0~1)', fontsize=10)
    # x축 - 매 4시간 표시
    tick_idx = [i for i, t in enumerate(time_cols) if t.endswith('00분') and int(t[:-4].rstrip('시')) % 3 == 0]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([time_cols[i].replace('시00분', '시') for i in tick_idx], rotation=0)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

axes[1, 0].set_xlabel('시간대', fontsize=11)
axes[1, 1].set_xlabel('시간대', fontsize=11)
plt.suptitle('서울 지하철 역사별 평일 혼잡도 시간 패턴 군집',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig02_cluster_patterns.png", bbox_inches='tight')
plt.close()
print(f"\n  ✓ 저장: fig02_cluster_patterns.png")


# ---------------------------------------------------------
# 8. 그림 3: PCA 2D 산점도
# ---------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

fig, ax = plt.subplots(figsize=(9, 7))
for c in range(K):
    mask = profile_active['cluster'] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[c], s=60, alpha=0.7,
               edgecolors='white', linewidth=1,
               label=f"C{c}: {cluster_info[c]['short_name']} ({cluster_info[c]['size']}역)")

# 강남, 잠실 등 주요역 라벨
key_stations = ['강남', '잠실', '석촌', '서울역', '홍대입구', '교대', '사당', '종로3가']
for station in key_stations:
    rows = profile_active[profile_active['역명_표준'] == station]
    if len(rows) > 0:
        idx = rows.index[0]
        ax.annotate(station, (X_pca[idx, 0], X_pca[idx, 1]),
                    fontsize=9, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% 설명)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% 설명)', fontsize=11)
ax.set_title('역사별 혼잡 패턴의 PCA 투영 (2D)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9, framealpha=0.95)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig03_pca_scatter.png", bbox_inches='tight')
plt.close()
print(f"  ✓ 저장: fig03_pca_scatter.png")


# ---------------------------------------------------------
# 9. 그림 4: 군집별 호선 분포 히트맵
# ---------------------------------------------------------
crosstab = pd.crosstab(profile_active['호선'], profile_active['cluster'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd',
            cbar_kws={'label': '역 수'}, ax=ax, linewidths=0.5)
ax.set_xlabel('군집', fontsize=11)
ax.set_ylabel('호선', fontsize=11)
ax.set_title('호선별 × 군집별 역 분포', fontsize=13, fontweight='bold')
ax.set_xticklabels([f"C{c}\n{cluster_info[c]['short_name']}" for c in range(K)], rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig04_line_cluster_heatmap.png", bbox_inches='tight')
plt.close()
print(f"  ✓ 저장: fig04_line_cluster_heatmap.png")


# ---------------------------------------------------------
# 10. 군집별 대표 역 표 저장
# ---------------------------------------------------------
print("\n" + "─" * 70)
print("  군집별 대표 역 (혼잡도 Top 10)")
print("─" * 70)

representative_table = []
for c in sorted(profile_active['cluster'].unique()):
    sub = profile_active[profile_active['cluster'] == c].nlargest(10, '전체_평균')
    sub_print = sub[['호선', '역명_표준', '전체_평균', '아침첨두_평균',
                     '저녁첨두_평균', '최대혼잡시간']].copy()
    sub_print['군집'] = f"C{c}: {cluster_info[c]['name']}"
    representative_table.append(sub_print)
    print(f"\n  ▶ Cluster {c}: {cluster_info[c]['name']}")
    print(sub_print[['호선', '역명_표준', '전체_평균', '최대혼잡시간']].to_string(index=False))

result_df = pd.concat(representative_table, ignore_index=True)
result_df.to_csv(TABLE_DIR / "table01_cluster_representatives.csv",
                 index=False, encoding='utf-8-sig')


# ---------------------------------------------------------
# 11. 전체 군집 결과 저장 (네트워크 분석에 재사용)
# ---------------------------------------------------------
profile_active.to_csv(PROCESSED_DIR / "혼잡도_프로파일_평일_군집포함.csv",
                     index=False, encoding='utf-8-sig')

# 군집 요약 통계
cluster_summary = (profile_active
    .groupby('cluster')
    .agg(역수=('역명_표준', 'count'),
         평균혼잡도=('전체_평균', 'mean'),
         최대혼잡도=('최대혼잡도', 'max'),
         아침첨두_평균=('아침첨두_평균', 'mean'),
         저녁첨두_평균=('저녁첨두_평균', 'mean'),
         첨두_집중도_평균=('첨두_집중도', 'mean'))
    .round(2))
cluster_summary['군집명'] = [cluster_info[c]['name'] for c in cluster_summary.index]
cluster_summary.to_csv(TABLE_DIR / "table02_cluster_summary.csv", encoding='utf-8-sig')

print("\n" + "─" * 70)
print("  군집 요약 통계")
print("─" * 70)
print(cluster_summary.to_string())


# ---------------------------------------------------------
# 12. 완료
# ---------------------------------------------------------
print("\n\n" + "█" * 70)
print("█  Step 3 완료")
print("█" * 70)
print(f"""
  생성된 산출물:
    [그림]
      - fig01_optimal_k.png          (최적 K 탐색)
      - fig02_cluster_patterns.png   (군집별 시간 패턴)
      - fig03_pca_scatter.png        (PCA 2D 산점도)
      - fig04_line_cluster_heatmap.png (호선×군집 분포)
    [표]
      - table01_cluster_representatives.csv
      - table02_cluster_summary.csv
    [데이터]
      - 혼잡도_프로파일_평일_군집포함.csv

  → 다음 단계(코로나 분석) 진행
""")
