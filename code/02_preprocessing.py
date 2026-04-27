"""
=========================================================
 졸업논문 분석 코드 - Step 2: 데이터 전처리
 목적:
   1) 중복 행 제거 (2026년 3월)
   2) 역명 표준화 (괄호 부가정보 통일, 분기점 통합)
   3) 호선 그룹화 (서울교통공사 1~8호선만 분석)
   4) 환승역 처리 (호선별 합산 옵션)
   5) 분석용 행렬 구축 (역 × 시간대)
   6) 정제 데이터 저장
=========================================================
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# 0. 경로 설정
# ---------------------------------------------------------
PROJECT_ROOT = Path("/Users/donghyunkim/Desktop/김동현/2026년 1학기/졸업논문/thesis")
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BOARDING_FILE = DATA_DIR / "boarding" / "서울시 지하철 호선별 역별 시간대별 승하차 인원 정보.csv"
CONGESTION_FILE = DATA_DIR / "congestion" / "서울교통공사_지하철혼잡도정보_20260331.csv"

# 분석 대상 호선 (서울교통공사 운영 + 혼잡도 데이터 보유)
TARGET_LINES = ['1호선', '2호선', '3호선', '4호선',
                '5호선', '6호선', '7호선', '8호선']


# ---------------------------------------------------------
# 1. 역명 표준화 함수
# ---------------------------------------------------------
def standardize_station(name: str) -> str:
    """
    역명을 표준 형태로 정규화
    - 괄호와 그 안 내용 제거: '강변(동서울터미널)' → '강변'
    - 분기점 표기 정리: '성수E'→'성수', '응암S'→'응암'
    - 공백 제거
    """
    if pd.isna(name):
        return name
    s = str(name).strip()
    # 괄호 부가정보 제거
    s = re.sub(r'\([^)]*\)', '', s)
    # 끝의 E/S 분기점 표기 제거 (성수E, 응암S)
    s = re.sub(r'[ES]$', '', s)
    s = s.strip()
    return s


# ---------------------------------------------------------
# 2. 승하차 데이터 전처리
# ---------------------------------------------------------
print("█" * 70)
print("█  데이터 A: 승하차 인원 전처리")
print("█" * 70)

df_a = pd.read_csv(BOARDING_FILE, encoding="cp949", low_memory=False)
print(f"  원본: {len(df_a):,}행")

# (1) 정확 중복 제거
before = len(df_a)
df_a = df_a.drop_duplicates(subset=['사용월', '호선명', '지하철역'], keep='first')
print(f"  중복 제거 후: {len(df_a):,}행 (제거: {before - len(df_a):,}행)")

# (2) 분석 대상 호선 필터링
df_a = df_a[df_a['호선명'].isin(TARGET_LINES)].copy()
print(f"  서울교통공사 1~8호선만: {len(df_a):,}행")

# (3) 역명 표준화
df_a['역명_표준'] = df_a['지하철역'].apply(standardize_station)
print(f"  표준화된 고유 역 수: {df_a['역명_표준'].nunique()}")

# (4) 시간대 컬럼 정리
boarding_cols = [c for c in df_a.columns if '승차인원' in c]
alighting_cols = [c for c in df_a.columns if '하차인원' in c]
print(f"  시간대 컬럼: 승차 {len(boarding_cols)}개, 하차 {len(alighting_cols)}개")

# (5) 사용월을 datetime으로
df_a['년월'] = pd.to_datetime(df_a['사용월'].astype(str), format='%Y%m')

print(f"\n  최종 컬럼: {df_a.shape[1]}개")
print(f"  분석 대상 월 수: {df_a['년월'].nunique()}")


# ---------------------------------------------------------
# 3. 혼잡도 데이터 전처리
# ---------------------------------------------------------
print("\n" + "█" * 70)
print("█  데이터 B: 혼잡도 전처리")
print("█" * 70)

df_b = pd.read_csv(CONGESTION_FILE, encoding="cp949", low_memory=False)
print(f"  원본: {len(df_b):,}행")

# 시간 컬럼
time_cols_30min = [c for c in df_b.columns if '시' in str(c) and '분' in str(c)]
print(f"  30분 단위 시간대: {len(time_cols_30min)}개 ({time_cols_30min[0]}~{time_cols_30min[-1]})")

# 역명 표준화
df_b['역명_표준'] = df_b['출발역'].apply(standardize_station)
print(f"  표준화된 고유 역 수: {df_b['역명_표준'].nunique()}")

# 호선 표기는 동일 (1호선, 2호선…) 유지


# ---------------------------------------------------------
# 4. 핵심: 역 단위 혼잡도 프로파일 생성
#    - 같은 역에서 상선/하선/내선/외선이 따로 있어
#      → 평균을 취해 역 대표 혼잡도 프로파일 만듦
# ---------------------------------------------------------
print("\n" + "█" * 70)
print("█  혼잡도 프로파일 행렬 구축 (RQ1 군집화 입력)")
print("█" * 70)

# 평일 데이터로 역 × 시간 행렬 생성 (방향 평균)
df_b_weekday = df_b[df_b['요일구분'] == '평일'].copy()
profile_weekday = (df_b_weekday
    .groupby(['호선', '역명_표준'])[time_cols_30min]
    .mean()
    .reset_index())
print(f"  평일 프로파일: {len(profile_weekday)}개 (호선-역 조합)")

# 주말도 함께
df_b_weekend = df_b[df_b['요일구분'].isin(['토요일', '일요일'])].copy()
profile_weekend = (df_b_weekend
    .groupby(['호선', '역명_표준'])[time_cols_30min]
    .mean()
    .reset_index())
print(f"  주말 프로파일: {len(profile_weekend)}개")

# 첨두/비첨두 비율 같은 파생 지표
def add_features(profile_df):
    df = profile_df.copy()
    morning_peak = ['7시30분', '8시00분', '8시30분', '9시00분']
    evening_peak = ['18시00분', '18시30분', '19시00분', '19시30분']
    midday = ['12시00분', '12시30분', '13시00분', '13시30분']
    df['아침첨두_평균'] = df[morning_peak].mean(axis=1)
    df['저녁첨두_평균'] = df[evening_peak].mean(axis=1)
    df['낮시간_평균'] = df[midday].mean(axis=1)
    df['전체_평균'] = df[time_cols_30min].mean(axis=1)
    df['최대혼잡도'] = df[time_cols_30min].max(axis=1)
    df['최대혼잡시간'] = df[time_cols_30min].idxmax(axis=1)
    # 첨두 집중도: (아침+저녁) / 낮시간 비율
    df['첨두_집중도'] = (df['아침첨두_평균'] + df['저녁첨두_평균']) / (df['낮시간_평균'] + 1e-6)
    return df

profile_weekday = add_features(profile_weekday)
profile_weekend = add_features(profile_weekend)


# ---------------------------------------------------------
# 5. 승하차 데이터에서 시계열 매트릭스 만들기
#    - 역 × 월 매트릭스 (출근시간대 승차 합산)
# ---------------------------------------------------------
print("\n" + "█" * 70)
print("█  승하차 시계열 매트릭스 구축 (RQ2 코로나 영향 분석용)")
print("█" * 70)

# 출근 시간대(7-9시) 승차 인원 합산
morning_boarding_cols = ['07시-08시 승차인원', '08시-09시 승차인원']
evening_alighting_cols = ['18시-19시 하차인원', '19시-20시 하차인원']

df_a['출근_승차'] = df_a[morning_boarding_cols].sum(axis=1)
df_a['퇴근_하차'] = df_a[evening_alighting_cols].sum(axis=1)
df_a['일평균_승차'] = df_a[boarding_cols].sum(axis=1)
df_a['일평균_하차'] = df_a[alighting_cols].sum(axis=1)

# 환승역은 호선별로 합산하여 한 행으로 통합
df_a_agg = (df_a
    .groupby(['년월', '역명_표준'])
    .agg({
        '출근_승차': 'sum',
        '퇴근_하차': 'sum',
        '일평균_승차': 'sum',
        '일평균_하차': 'sum',
        '호선명': lambda x: ','.join(sorted(set(x)))
    })
    .reset_index()
    .rename(columns={'호선명': '호선목록'}))
print(f"  통합 후: {len(df_a_agg):,}행 (역 × 월)")

# 역 × 월 행렬 (출근 승차)
ts_matrix = df_a_agg.pivot(index='역명_표준', columns='년월', values='출근_승차').fillna(0)
print(f"  시계열 행렬: {ts_matrix.shape[0]}역 × {ts_matrix.shape[1]}개월")


# ---------------------------------------------------------
# 6. 두 데이터셋 매칭 — 분석 대상 역 확정
# ---------------------------------------------------------
print("\n" + "█" * 70)
print("█  최종 분석 대상 역 확정")
print("█" * 70)

stations_a = set(df_a_agg['역명_표준'].unique())
stations_b = set(profile_weekday['역명_표준'].unique())
common_stations = stations_a & stations_b
print(f"  승하차(1~8호선): {len(stations_a)}개")
print(f"  혼잡도: {len(stations_b)}개")
print(f"  공통: {len(common_stations)}개  ← 이 역들이 핵심 분석 대상")


# ---------------------------------------------------------
# 7. 결과 저장
# ---------------------------------------------------------
print("\n" + "█" * 70)
print("█  전처리 결과 저장")
print("█" * 70)

profile_weekday.to_csv(PROCESSED_DIR / "혼잡도_프로파일_평일.csv",
                      index=False, encoding='utf-8-sig')
profile_weekend.to_csv(PROCESSED_DIR / "혼잡도_프로파일_주말.csv",
                      index=False, encoding='utf-8-sig')
df_a_agg.to_csv(PROCESSED_DIR / "승하차_월별_통합.csv",
                index=False, encoding='utf-8-sig')
ts_matrix.to_csv(PROCESSED_DIR / "출근승차_시계열행렬.csv",
                 encoding='utf-8-sig')

# 공통역 목록도 저장
pd.Series(sorted(common_stations), name='역명_표준').to_csv(
    PROCESSED_DIR / "공통역_목록.csv", index=False, encoding='utf-8-sig')

print(f"  ✓ 저장 위치: {PROCESSED_DIR}")
print(f"     - 혼잡도_프로파일_평일.csv  ({len(profile_weekday)}행)")
print(f"     - 혼잡도_프로파일_주말.csv  ({len(profile_weekend)}행)")
print(f"     - 승하차_월별_통합.csv      ({len(df_a_agg)}행)")
print(f"     - 출근승차_시계열행렬.csv   ({ts_matrix.shape})")
print(f"     - 공통역_목록.csv           ({len(common_stations)}역)")


# ---------------------------------------------------------
# 8. 검증 요약
# ---------------------------------------------------------
print("\n" + "█" * 70)
print("█  전처리 결과 검증")
print("█" * 70)

# 평일 혼잡도 Top 10 (전체 평균 기준)
print("\n  ▶ 평일 평균 혼잡도 Top 10 역:")
top10 = profile_weekday.nlargest(10, '전체_평균')[
    ['호선', '역명_표준', '전체_평균', '최대혼잡도', '최대혼잡시간']]
print(top10.to_string(index=False))

# 첨두 집중형 vs 분산형 역
print("\n  ▶ 첨두 집중도 Top 5 (출퇴근 집중형):")
top5_peak = profile_weekday.nlargest(5, '첨두_집중도')[
    ['호선', '역명_표준', '첨두_집중도', '아침첨두_평균', '낮시간_평균']]
print(top5_peak.to_string(index=False))

print("\n  ▶ 첨두 집중도 Bottom 5 (분산형):")
bot5_peak = profile_weekday.nsmallest(5, '첨두_집중도')[
    ['호선', '역명_표준', '첨두_집중도', '아침첨두_평균', '낮시간_평균']]
print(bot5_peak.to_string(index=False))

# 시계열 검증 — 코로나 영향
print("\n  ▶ 코로나 영향 검증 (전체 역 출근 승차 합):")
monthly_total = ts_matrix.sum(axis=0)
key_months = [pd.Timestamp('2019-12-01'), pd.Timestamp('2020-03-01'),
              pd.Timestamp('2020-08-01'), pd.Timestamp('2022-03-01'),
              pd.Timestamp('2024-03-01'), pd.Timestamp('2026-03-01')]
for m in key_months:
    if m in monthly_total.index:
        print(f"     {m.strftime('%Y-%m')}: {monthly_total[m]:>15,.0f}명")

print("\n" + "█" * 70)
print("█  전처리 완료. 다음 단계: 군집화 + 네트워크 분석")
print("█" * 70)
