"""
=========================================================
 졸업논문 분석 코드 - Step 4: 코로나19 충격 및 회복 분석 (RQ2)
 목적:
   - 11년치 월별 데이터로 코로나 전·중·후 이용 패턴 변화 추적
   - 군집별 충격 크기와 회복 속도 정량 비교
   - 단절 시계열 회귀(Interrupted Time Series, ITS)로 인과 추정
   - 회복 지수(Recovery Index) 계산
=========================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# statsmodels 옵션 (없으면 numpy로 대체)
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    from scipy import stats
    print("ⓘ statsmodels가 없으니 numpy 기반 OLS 사용")

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
# 1. 데이터 로드
# ---------------------------------------------------------
print("█" * 70)
print("█  Step 4: 코로나19 충격 및 회복 분석")
print("█" * 70)

ts = pd.read_csv(PROCESSED_DIR / "출근승차_시계열행렬.csv", index_col=0)
ts.columns = pd.to_datetime(ts.columns)
print(f"\n  시계열 행렬: {ts.shape}")
print(f"  기간: {ts.columns.min().strftime('%Y-%m')} ~ {ts.columns.max().strftime('%Y-%m')}")

clusters = pd.read_csv(PROCESSED_DIR / "혼잡도_프로파일_평일_군집포함.csv")
# 환승역의 경우 가장 혼잡한 호선의 군집을 대표로 사용
station_cluster = (clusters
    .sort_values('전체_평균', ascending=False)
    .drop_duplicates('역명_표준', keep='first'))
station_cluster_dict = dict(zip(station_cluster['역명_표준'], station_cluster['cluster']))


# ---------------------------------------------------------
# 2. 군집명 매핑 (Step 3 결과 재사용)
# ---------------------------------------------------------
CLUSTER_NAMES = {
    0: "전일 분산형 · 도심·관광 (중혼잡)",
    1: "출근 집중형 · 주거지 (고혼잡)",
    2: "출근 집중형 · 주거지 (중혼잡)",
    3: "전일 분산형 · 도심·관광 (고혼잡)"
}
CLUSTER_SHORT = {
    0: "도심분산-중", 1: "주거첨두-고",
    2: "주거첨두-중", 3: "도심분산-고"
}
CLUSTER_COLORS = {0: '#3498DB', 1: '#E74C3C', 2: '#27AE60', 3: '#F39C12'}


# ---------------------------------------------------------
# 3. 군집별 월별 시계열 합산
# ---------------------------------------------------------
common = set(ts.index) & set(station_cluster_dict.keys())
ts_clustered = ts.loc[list(common)].copy()
ts_clustered['cluster'] = ts_clustered.index.map(station_cluster_dict)

# 월별 군집별 합산
monthly_by_cluster = ts_clustered.groupby('cluster').sum()  # 4 × 135

# 전체 시계열도
monthly_total = ts.sum(axis=0)

print(f"\n  분석 대상 역: {len(common)}개")
print(f"  군집별 시계열 행렬: {monthly_by_cluster.shape}")


# ---------------------------------------------------------
# 4. 주요 시점 정의
# ---------------------------------------------------------
COVID_START = pd.Timestamp('2020-02-01')  # 1차 대유행
RECOVERY_TARGET = pd.Timestamp('2019-12-01')  # 코로나 직전 기준 (회복 100%의 기준)

# 4.1 충격 크기 측정 (2020년 평균 / 2019년 평균)
# 컬럼이 Timestamp이므로 마스크 사용
cols = monthly_by_cluster.columns
pre_covid_mask = (cols >= pd.Timestamp('2019-01-01')) & (cols <= pd.Timestamp('2019-12-01'))
during_mask = (cols >= pd.Timestamp('2020-04-01')) & (cols <= pd.Timestamp('2021-12-01'))
post_mask = (cols >= pd.Timestamp('2025-04-01')) & (cols <= pd.Timestamp('2026-03-01'))

pre_covid = monthly_by_cluster.loc[:, pre_covid_mask].mean(axis=1)
during_covid = monthly_by_cluster.loc[:, during_mask].mean(axis=1)
post_covid = monthly_by_cluster.loc[:, post_mask].mean(axis=1)

shock_pct = (during_covid - pre_covid) / pre_covid * 100
recovery_pct = (post_covid - pre_covid) / pre_covid * 100

print("\n" + "─" * 70)
print("  군집별 코로나 충격 및 최근 회복")
print("─" * 70)
shock_table = pd.DataFrame({
    '군집명': [CLUSTER_NAMES[c] for c in monthly_by_cluster.index],
    '코로나 직전(2019)': pre_covid.values.astype(int),
    '코로나 기간(2020-21)': during_covid.values.astype(int),
    '최근(2025-26)': post_covid.values.astype(int),
    '충격(%)': shock_pct.round(2).values,
    '회복(%)': recovery_pct.round(2).values
})
shock_table.index = [f"C{c}" for c in monthly_by_cluster.index]
print(shock_table.to_string())
shock_table.to_csv(TABLE_DIR / "table03_covid_shock.csv", encoding='utf-8-sig')


# ---------------------------------------------------------
# 5. 그림 5: 군집별 시계열 (정규화 vs 절대값)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# (1) 절대값
ax = axes[0]
for c in [0, 1, 2, 3]:
    series = monthly_by_cluster.loc[c] / 1e6  # 백만 단위
    ax.plot(series.index, series.values,
            color=CLUSTER_COLORS[c], linewidth=2, alpha=0.85,
            label=f"C{c}: {CLUSTER_SHORT[c]}")

# 코로나 시점 표시
ax.axvline(COVID_START, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax.text(COVID_START, ax.get_ylim()[1] * 0.95, '  코로나 시작', color='red', fontsize=10)

ax.set_xlabel('연도-월', fontsize=11)
ax.set_ylabel('출근시간대 승차 인원 (백만 명/월)', fontsize=11)
ax.set_title('군집별 출근시간대 이용량 시계열 (2015.01~2026.03)',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3)

# (2) 정규화 (2019년 12월 = 100)
ax = axes[1]
baseline = monthly_by_cluster.loc[:, RECOVERY_TARGET]
for c in [0, 1, 2, 3]:
    series = monthly_by_cluster.loc[c] / baseline[c] * 100
    # 12개월 이동평균(추세 강조)
    smoothed = series.rolling(window=3, center=True).mean()
    ax.plot(series.index, smoothed.values,
            color=CLUSTER_COLORS[c], linewidth=2.5, alpha=0.9,
            label=f"C{c}: {CLUSTER_SHORT[c]}")

ax.axvline(COVID_START, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axhline(100, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.text(COVID_START, 105, '  코로나 시작', color='red', fontsize=10)

ax.set_xlabel('연도-월', fontsize=11)
ax.set_ylabel('상대 이용량 (%, 2019.12 = 100)', fontsize=11)
ax.set_title('정규화된 군집별 이용량 변화 (3개월 이동평균)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig05_timeseries_by_cluster.png", bbox_inches='tight')
plt.close()
print(f"\n  ✓ 저장: fig05_timeseries_by_cluster.png")


# ---------------------------------------------------------
# 6. ITS (Interrupted Time Series) 회귀 분석
#    Y_t = β0 + β1·T + β2·D + β3·T_after + ε
#      T:        시간(1~135)
#      D:        코로나 더미 (코로나 후 1, 그 외 0)
#      T_after:  코로나 후 경과 시간
#    β2: 즉각 효과 (level shift)
#    β3: 추세 변화 (slope shift)
# ---------------------------------------------------------
print("\n" + "─" * 70)
print("  ITS 회귀 분석: 군집별 코로나 효과")
print("─" * 70)

months_idx = monthly_by_cluster.columns
T = np.arange(1, len(months_idx) + 1)
covid_idx = np.searchsorted(months_idx, COVID_START)
D = (np.arange(len(months_idx)) >= covid_idx).astype(int)
T_after = np.where(D == 1, np.arange(len(months_idx)) - covid_idx + 1, 0)

its_results = []


def fit_its(Y, T, D, T_after):
    """ITS 회귀 — statsmodels 또는 numpy로 실행"""
    if HAS_STATSMODELS:
        X = sm.add_constant(np.column_stack([T, D, T_after]))
        model = sm.OLS(Y, X).fit()
        return {
            'params': model.params,
            'pvalues': model.pvalues,
            'rsquared': model.rsquared,
            'predict': model.predict(X)
        }
    else:
        X = np.column_stack([np.ones(len(T)), T, D, T_after])
        n, k = X.shape
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ Y
        y_pred = X @ beta
        resid = Y - y_pred
        sigma2 = (resid @ resid) / (n - k)
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        t_stat = beta / se
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))
        r2 = 1 - (resid @ resid) / ((Y - Y.mean()) @ (Y - Y.mean()))
        return {
            'params': beta,
            'pvalues': p_vals,
            'rsquared': r2,
            'predict': y_pred
        }


for c in [0, 1, 2, 3]:
    Y = monthly_by_cluster.loc[c].values / 1e6
    res = fit_its(Y, T, D, T_after)

    its_results.append({
        '군집': f"C{c}",
        '군집명': CLUSTER_NAMES[c],
        'β0_절편': round(res['params'][0], 3),
        'β1_시간': round(res['params'][1], 4),
        'β2_즉각효과': round(res['params'][2], 3),
        'β3_추세변화': round(res['params'][3], 4),
        'β2_p값': f"{res['pvalues'][2]:.4f}",
        'β3_p값': f"{res['pvalues'][3]:.4f}",
        'R²': round(res['rsquared'], 3)
    })
    print(f"\n  [{c}] {CLUSTER_NAMES[c]}")
    print(f"    즉각 충격(β2): {res['params'][2]:+.3f}백만명 (p={res['pvalues'][2]:.4f})")
    print(f"    추세 변화(β3): {res['params'][3]:+.4f}백만명/월 (p={res['pvalues'][3]:.4f})")
    print(f"    R²: {res['rsquared']:.3f}")

its_df = pd.DataFrame(its_results)
its_df.to_csv(TABLE_DIR / "table04_its_regression.csv",
              index=False, encoding='utf-8-sig')


# ---------------------------------------------------------
# 7. 그림 6: ITS 회귀선 시각화 (한 군집씩)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

for c in [0, 1, 2, 3]:
    ax = axes[c // 2, c % 2]
    Y = monthly_by_cluster.loc[c].values / 1e6
    ax.scatter(months_idx, Y, alpha=0.5, s=15,
               color=CLUSTER_COLORS[c], label='실측치')

    res = fit_its(Y, T, D, T_after)
    Y_pred = res['predict']

    # 코로나 이전/이후 분리해서 그리기
    ax.plot(months_idx[D == 0], Y_pred[D == 0],
            color='black', linewidth=2, label='회귀선')
    ax.plot(months_idx[D == 1], Y_pred[D == 1],
            color='black', linewidth=2)

    ax.axvline(COVID_START, color='red', linestyle='--', alpha=0.4)
    ax.set_title(f"C{c}: {CLUSTER_NAMES[c]}\n"
                 f"즉각 충격={res['params'][2]:+.2f}M (p={res['pvalues'][2]:.3f}), "
                 f"추세={res['params'][3]:+.4f}M/월",
                 fontsize=10, fontweight='bold')
    ax.set_ylabel('이용량 (백만명/월)', fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

axes[1, 0].set_xlabel('연도-월', fontsize=10)
axes[1, 1].set_xlabel('연도-월', fontsize=10)
plt.suptitle('군집별 단절 시계열 회귀 (ITS) 분석',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig06_its_regression.png", bbox_inches='tight')
plt.close()
print(f"\n  ✓ 저장: fig06_its_regression.png")


# ---------------------------------------------------------
# 8. 회복 속도 비교: 회복 곡선 곡률
# ---------------------------------------------------------
print("\n" + "─" * 70)
print("  회복 속도 분석")
print("─" * 70)

# 각 군집이 코로나 직전 수준의 95%를 회복한 시점
recovery_table = []
for c in [0, 1, 2, 3]:
    series = monthly_by_cluster.loc[c]
    baseline_val = series.loc[RECOVERY_TARGET]

    # 코로나 이후 95% 도달 시점
    after_covid = series[series.index >= COVID_START]
    threshold_95 = baseline_val * 0.95
    threshold_100 = baseline_val * 1.00
    recovery_95 = after_covid[after_covid >= threshold_95]
    recovery_100 = after_covid[after_covid >= threshold_100]

    rec_95_month = recovery_95.index[0] if len(recovery_95) > 0 else None
    rec_100_month = recovery_100.index[0] if len(recovery_100) > 0 else None
    rec_95_lag = ((rec_95_month - COVID_START).days // 30) if rec_95_month else None
    rec_100_lag = ((rec_100_month - COVID_START).days // 30) if rec_100_month else None

    # 최저점
    min_idx = after_covid.idxmin()
    min_pct = (after_covid.min() - baseline_val) / baseline_val * 100

    recovery_table.append({
        '군집': f"C{c}",
        '군집명': CLUSTER_SHORT[c],
        '최저점_시점': min_idx.strftime('%Y-%m') if pd.notna(min_idx) else '-',
        '최대낙폭(%)': round(min_pct, 2),
        '95%회복_시점': rec_95_month.strftime('%Y-%m') if rec_95_month else '미회복',
        '95%회복_지연(개월)': rec_95_lag if rec_95_lag is not None else None,
        '100%회복_시점': rec_100_month.strftime('%Y-%m') if rec_100_month else '미회복',
        '100%회복_지연(개월)': rec_100_lag if rec_100_lag is not None else None
    })

recovery_df = pd.DataFrame(recovery_table)
print(recovery_df.to_string(index=False))
recovery_df.to_csv(TABLE_DIR / "table05_recovery_speed.csv",
                   index=False, encoding='utf-8-sig')


# ---------------------------------------------------------
# 9. 그림 7: 충격-회복 산점도
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

for c in [0, 1, 2, 3]:
    ax.scatter(shock_pct[c], recovery_pct[c],
               s=400, c=CLUSTER_COLORS[c], alpha=0.85,
               edgecolors='white', linewidth=2,
               label=f"C{c}: {CLUSTER_SHORT[c]}")
    ax.annotate(f"C{c}", (shock_pct[c], recovery_pct[c]),
                fontsize=12, fontweight='bold', ha='center', va='center',
                color='white')

ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('코로나 기간 충격 크기 (%, 2019 대비)', fontsize=11)
ax.set_ylabel('현재 회복 정도 (%, 2019 대비)', fontsize=11)
ax.set_title('군집별 충격 크기 vs 회복 정도', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3)

# 사분면 라벨
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(xlim[0] * 0.95, ylim[1] * 0.95, '큰 충격 + 완전회복', fontsize=9,
        color='gray', ha='left', va='top', style='italic')
ax.text(xlim[1] * 0.95, ylim[1] * 0.95, '작은 충격 + 완전회복', fontsize=9,
        color='gray', ha='right', va='top', style='italic')

plt.tight_layout()
plt.savefig(FIG_DIR / "fig07_shock_recovery.png", bbox_inches='tight')
plt.close()
print(f"\n  ✓ 저장: fig07_shock_recovery.png")


# ---------------------------------------------------------
# 10. 완료
# ---------------------------------------------------------
print("\n\n" + "█" * 70)
print("█  Step 4 완료")
print("█" * 70)
print(f"""
  생성된 산출물:
    [그림]
      - fig05_timeseries_by_cluster.png  (군집별 시계열, 정규화)
      - fig06_its_regression.png          (ITS 회귀 분석)
      - fig07_shock_recovery.png          (충격-회복 산점도)
    [표]
      - table03_covid_shock.csv           (충격 크기)
      - table04_its_regression.csv        (ITS 회귀 결과)
      - table05_recovery_speed.csv        (회복 속도)

  → 다음 단계(환승 네트워크 분석) 진행
""")
