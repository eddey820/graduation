"""
=========================================================
 졸업논문 분석 코드 - Step 1: 데이터 탐색
 - 사용자 환경 경로 반영
 - cp949 인코딩 자동 사용
 - 데이터 특성에 맞춘 분석 추가
=========================================================
"""

import os
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# 0. 경로 설정 (사용자 환경 반영)
# ---------------------------------------------------------
PROJECT_ROOT = Path("/Users/donghyunkim/Desktop/김동현/2026년 1학기/졸업논문/thesis")
DATA_DIR = PROJECT_ROOT / "data"
BOARDING_FILE = DATA_DIR / "boarding" / "서울시 지하철 호선별 역별 시간대별 승하차 인원 정보.csv"
CONGESTION_FILE = DATA_DIR / "congestion" / "서울교통공사_지하철혼잡도정보_20260331.csv"

# 출력 폴더 생성
(PROJECT_ROOT / "output" / "tables").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "output" / "figures").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"프로젝트 루트: {PROJECT_ROOT}")
print(f"승하차 데이터: {BOARDING_FILE.name}")
print(f"혼잡도 데이터: {CONGESTION_FILE.name}")
print("=" * 70)


# ---------------------------------------------------------
# 1. 승하차 데이터 탐색
# ---------------------------------------------------------
print("\n\n" + "█" * 70)
print("█  데이터 A: 시간대별 승하차 인원")
print("█" * 70)

df_a = pd.read_csv(BOARDING_FILE, encoding="cp949", low_memory=False)
print(f"\n  행 수: {len(df_a):,} / 열 수: {df_a.shape[1]}")
print(f"  파일 크기: {BOARDING_FILE.stat().st_size / 1024 / 1024:.2f} MB")

print(f"\n  ▶ 컬럼 (총 {len(df_a.columns)}개):")
for i, c in enumerate(df_a.columns):
    print(f"     [{i:2d}] {c}")

print(f"\n  ▶ 사용월(년월) 범위: {df_a['사용월'].min()} ~ {df_a['사용월'].max()}")
print(f"     고유 월 개수: {df_a['사용월'].nunique()}")

print(f"\n  ▶ 호선 분포:")
print(df_a['호선명'].value_counts().to_string())

print(f"\n  ▶ 고유 역 수: {df_a['지하철역'].nunique()}")

# 결측치
na = df_a.isna().sum()
print(f"\n  ▶ 결측치: {'없음' if (na == 0).all() else na[na > 0].to_string()}")

# 처음 3행 (시간대 컬럼 일부만)
print(f"\n  ▶ 샘플 (처음 3행, 출근시간대만):")
sample_cols = ['사용월', '호선명', '지하철역',
               '07시-08시 승차인원', '07시-08시 하차인원',
               '08시-09시 승차인원', '08시-09시 하차인원']
print(df_a[sample_cols].head(3).to_string())


# ---------------------------------------------------------
# 2. 혼잡도 데이터 탐색
# ---------------------------------------------------------
print("\n\n" + "█" * 70)
print("█  데이터 B: 지하철 혼잡도 정보 (2026년 1분기)")
print("█" * 70)

df_b = pd.read_csv(CONGESTION_FILE, encoding="cp949", low_memory=False)
print(f"\n  행 수: {len(df_b):,} / 열 수: {df_b.shape[1]}")
print(f"  파일 크기: {CONGESTION_FILE.stat().st_size / 1024 / 1024:.2f} MB")

print(f"\n  ▶ 컬럼 (총 {len(df_b.columns)}개):")
for i, c in enumerate(df_b.columns):
    print(f"     [{i:2d}] {c}")

# 시간 컬럼 식별
time_cols = [c for c in df_b.columns if '시' in str(c) and '분' in str(c)]
non_time_cols = [c for c in df_b.columns if c not in time_cols]
print(f"\n  ▶ 메타 컬럼: {non_time_cols}")
print(f"  ▶ 시간대 컬럼 ({len(time_cols)}개, 30분 단위):")
print(f"     {time_cols[0]} ~ {time_cols[-1]}")

print(f"\n  ▶ 호선 분포:")
print(df_b['호선'].value_counts().to_string())

print(f"\n  ▶ 요일구분 분포:")
print(df_b['요일구분'].value_counts().to_string())

print(f"\n  ▶ 상하구분 분포:")
print(df_b['상하구분'].value_counts().to_string())

print(f"\n  ▶ 고유 역 수: {df_b['출발역'].nunique()}")

# 혼잡도 통계
print(f"\n  ▶ 혼잡도 값 통계:")
all_vals = df_b[time_cols].values.flatten()
print(f"     최소: {all_vals.min():.1f}%, 최대: {all_vals.max():.1f}%")
print(f"     평균: {all_vals.mean():.1f}%, 중간값: {pd.Series(all_vals).median():.1f}%")
print(f"     ※ 정원 대비 비율(%), 좌석만 다 차면 34%")

# 결측치
na = df_b.isna().sum()
print(f"\n  ▶ 결측치: {'없음' if (na == 0).all() else na[na > 0].to_string()}")


# ---------------------------------------------------------
# 3. 두 데이터셋 매칭 가능성 점검
# ---------------------------------------------------------
print("\n\n" + "█" * 70)
print("█  데이터 매칭 분석")
print("█" * 70)

# 두 데이터셋 모두에 등장하는 역 찾기 (혼잡도는 1~8호선만 있음)
seoul_lines = ['1호선','2호선','3호선','4호선','5호선','6호선','7호선','8호선']
boarding_stations = set(df_a[df_a['호선명'].isin(seoul_lines)]['지하철역'].unique())
congestion_stations = set(df_b['출발역'].unique())

common = boarding_stations & congestion_stations
only_b = boarding_stations - congestion_stations
only_c = congestion_stations - boarding_stations

print(f"\n  승하차 데이터의 1~8호선 역: {len(boarding_stations)}개")
print(f"  혼잡도 데이터 역: {len(congestion_stations)}개")
print(f"  교집합 (분석 대상): {len(common)}개")
print(f"  승하차에만 있음: {len(only_b)}개 (예: {sorted(list(only_b))[:5]})")
print(f"  혼잡도에만 있음: {len(only_c)}개")
if only_c:
    print(f"     예시: {sorted(list(only_c))[:10]}")


# ---------------------------------------------------------
# 4. 2026년 3월 (혼잡도 시기와 동일한 시점) 승하차 정보
# ---------------------------------------------------------
print("\n\n" + "█" * 70)
print("█  2026년 3월 승하차 데이터 (혼잡도와 동일 시점)")
print("█" * 70)

df_a_202603 = df_a[df_a['사용월'] == 202603]
print(f"\n  2026년 3월 행 수: {len(df_a_202603)}")
print(f"  호선 분포:")
print(df_a_202603['호선명'].value_counts().to_string())


# ---------------------------------------------------------
# 5. 요약 및 다음 단계
# ---------------------------------------------------------
print("\n\n" + "█" * 70)
print("█  탐색 완료 — 핵심 발견사항")
print("█" * 70)
print(f"""
  [데이터 A] 승하차 인원
    - 2015년 1월 ~ 2026년 3월, 11년치 월별 데이터 (총 {df_a['사용월'].nunique()}개월)
    - 600개 역, 1시간 단위 24구간
    - 시계열 분석 가능 (월별 추세, 코로나 영향 등)

  [데이터 B] 혼잡도
    - 2026년 1분기, 30분 단위 39구간
    - 245개 역 (서울교통공사 1~8호선만)
    - 평일/토/일 × 상선/하선 별 평균 혼잡도

  [매칭] 두 데이터 공통 역: {len(common)}개 → 핵심 분석 대상

  → 다음 단계: 전처리 + 군집화 + 환승 네트워크 구축
""")
