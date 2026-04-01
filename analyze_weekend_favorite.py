"""
주말/평일별 페이버릿/언더독 승률 분석
- 미국 동부 시간대(ET) 기준
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def analyze_favorite_by_day_type():
    """주말/평일별 페이버릿 승률 분석"""
    
    # 마스터 파일 로드
    master_file = Path(__file__).parent / "data" / "matched" / "nba_odds_results_matched_master.json"
    
    with open(master_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"총 경기 수: {len(df)}")
    
    # ET 시간에서 요일 추출
    def get_weekday_et(commence_time_et):
        try:
            # "2025-11-11 19:30:00 EST" 형식
            dt_str = commence_time_et.replace(' EST', '').replace(' EDT', '')
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
            return dt.weekday()  # 0=월, 6=일
        except:
            return None
    
    df['weekday_et'] = df['commence_time_et'].apply(get_weekday_et)
    df = df.dropna(subset=['weekday_et'])
    
    # 주말/평일 구분 (토=5, 일=6)
    df['is_weekend'] = df['weekday_et'].isin([5, 6])
    df['day_type'] = df['is_weekend'].map({True: '주말 (토,일)', False: '평일 (월~금)'})
    
    # 페이버릿 구분 (낮은 배당 = 페이버릿)
    df['home_is_favorite'] = df['home_odds'] < df['away_odds']
    
    # 페이버릿이 이겼는지
    df['favorite_won'] = (
        (df['home_is_favorite'] & (df['winner'] == 'home')) |
        (~df['home_is_favorite'] & (df['winner'] == 'away'))
    )
    
    # 언더독이 이겼는지
    df['underdog_won'] = ~df['favorite_won']
    
    print("\n" + "="*60)
    print("[ANALYSIS] Weekend vs Weekday Favorite Win Rate (ET)")
    print("="*60)
    
    # 주말/평일별 분석
    result = df.groupby('day_type').agg({
        'favorite_won': ['sum', 'count', 'mean'],
        'underdog_won': 'mean'
    }).round(4)
    
    result.columns = ['페이버릿 승', '총 경기', '페이버릿 승률', '언더독 승률']
    result['페이버릿 승률'] = (result['페이버릿 승률'] * 100).round(2).astype(str) + '%'
    result['언더독 승률'] = (result['언더독 승률'] * 100).round(2).astype(str) + '%'
    result['페이버릿 승'] = result['페이버릿 승'].astype(int)
    result['총 경기'] = result['총 경기'].astype(int)
    
    print("\n", result)
    
    # 요일별 상세 분석
    print("\n" + "="*60)
    print("[DETAIL] By Day of Week")
    print("="*60)
    
    day_names = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
    df['day_name'] = df['weekday_et'].map(day_names)
    
    day_result = df.groupby(['weekday_et', 'day_name']).agg({
        'favorite_won': ['count', 'mean'],
        'underdog_won': 'mean'
    }).round(4)
    
    day_result.columns = ['경기수', '페이버릿 승률', '언더독 승률']
    day_result = day_result.reset_index()
    day_result = day_result.sort_values('weekday_et')
    day_result['페이버릿 승률'] = (day_result['페이버릿 승률'] * 100).round(2).astype(str) + '%'
    day_result['언더독 승률'] = (day_result['언더독 승률'] * 100).round(2).astype(str) + '%'
    
    print("\n요일 | 경기수 | 페이버릿 승률 | 언더독 승률")
    print("-" * 45)
    for _, row in day_result.iterrows():
        print(f" {row['day_name']}  |  {row['경기수']:>4}  |    {row['페이버릿 승률']:>7}   |   {row['언더독 승률']:>7}")
    
    # 배당 구간별 분석 (추가)
    print("\n" + "="*60)
    print("[ODDS] Favorite Win Rate by Odds Range")
    print("="*60)
    
    # 페이버릿 배당 추출
    df['favorite_odds'] = df.apply(
        lambda x: x['home_odds'] if x['home_is_favorite'] else x['away_odds'],
        axis=1
    )
    
    # 구간화
    bins = [-float('inf'), -400, -250, -150, -100]
    labels = ['압도적(-400이하)', '강함(-400~-250)', '보통(-250~-150)', '약함(-150~-100)']
    df['odds_bucket'] = pd.cut(df['favorite_odds'], bins=bins, labels=labels)
    
    odds_result = df.groupby('odds_bucket', observed=True).agg({
        'favorite_won': ['count', 'mean']
    }).round(4)
    odds_result.columns = ['경기수', '페이버릿 승률']
    odds_result['페이버릿 승률'] = (odds_result['페이버릿 승률'] * 100).round(2).astype(str) + '%'
    
    print("\n", odds_result)
    
    print("\n" + "="*60)
    print("[DONE] Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    analyze_favorite_by_day_type()
