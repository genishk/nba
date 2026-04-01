# src/odds_fetcher_totals.py
# NBA Total Over/Under 배당 데이터 수집

import requests
import pandas as pd
from pathlib import Path
import json
import os
from datetime import datetime, timezone, timedelta
import time

# 시간대 정의
ET = timezone(timedelta(hours=-5))


class TotalsOddsFetcher:
    """NBA Total Over/Under 배당 수집기"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.sport = "basketball_nba"
        
    def fetch_totals_odds(self) -> pd.DataFrame:
        """모든 경기의 Total Over/Under 배당 가져오기"""
        
        # 1. 경기 목록 가져오기
        events_url = f"{self.base_url}/{self.sport}/events"
        events_params = {'apiKey': self.api_key}
        
        try:
            print("\n=== NBA Totals 배당 수집 시작 ===")
            
            events_response = requests.get(events_url, params=events_params)
            events_response.raise_for_status()
            events_data = events_response.json()
            
            print(f"📋 {len(events_data)}개 경기 발견")
            
            if not events_data:
                print("❌ 예정된 경기가 없습니다.")
                return None
            
            # 2. 각 경기별 totals 배당 수집
            all_totals = []
            
            for event in events_data:
                event_id = event['id']
                
                # ET 시간 계산
                utc_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
                et_time = utc_time.astimezone(ET)
                
                print(f"  📡 {event['home_team']} vs {event['away_team']} ({et_time.strftime('%m/%d %H:%M ET')})")
                
                # totals 배당 요청
                odds_url = f"{self.base_url}/{self.sport}/events/{event_id}/odds"
                odds_params = {
                    'apiKey': self.api_key,
                    'regions': 'us',
                    'markets': 'totals',
                    'oddsFormat': 'american',
                    'bookmakers': 'draftkings'
                }
                
                odds_response = requests.get(odds_url, params=odds_params)
                odds_response.raise_for_status()
                odds_data = odds_response.json()
                
                # 파싱
                parsed = self._parse_totals(odds_data)
                if parsed:
                    all_totals.append(parsed)
                
                time.sleep(0.5)  # Rate limit
            
            if not all_totals:
                print("❌ 수집된 배당 데이터가 없습니다.")
                return None
            
            # 3. DataFrame 생성 및 저장
            df = pd.DataFrame(all_totals)
            
            print(f"\n✅ {len(df)}개 경기 totals 배당 수집 완료")
            print(f"   Line 범위: {df['total_line'].min()} ~ {df['total_line'].max()}")
            
            # 저장
            output_path = self._save_data(df)
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API 오류: {e}")
            return None
    
    def _parse_totals(self, odds_data: dict) -> dict:
        """totals 응답 파싱"""
        try:
            # UTC → ET 변환
            commence_utc = odds_data.get('commence_time')
            if commence_utc:
                utc_time = datetime.fromisoformat(commence_utc.replace('Z', '+00:00'))
                et_time = utc_time.astimezone(ET)
                commence_et = et_time.strftime('%Y-%m-%d %H:%M ET')
                date_et = et_time.strftime('%Y-%m-%d')
            else:
                commence_et = None
                date_et = None
            
            result = {
                'game_id': odds_data.get('id'),
                'home_team': odds_data.get('home_team'),
                'away_team': odds_data.get('away_team'),
                'commence_time_utc': commence_utc,
                'commence_time_et': commence_et,
                'date_et': date_et,
            }
            
            # 북메이커 데이터 파싱
            bookmakers = odds_data.get('bookmakers', [])
            if not bookmakers:
                return None
            
            bm = bookmakers[0]
            result['bookmaker'] = bm['key']
            
            for market in bm.get('markets', []):
                if market['key'] == 'totals':
                    for outcome in market.get('outcomes', []):
                        if outcome['name'] == 'Over':
                            result['total_line'] = outcome.get('point')
                            result['over_odds'] = outcome.get('price')
                        elif outcome['name'] == 'Under':
                            result['under_odds'] = outcome.get('price')
            
            return result
            
        except Exception as e:
            print(f"  ⚠️ 파싱 오류: {e}")
            return None
    
    def _save_data(self, df: pd.DataFrame) -> Path:
        """데이터 저장"""
        odds_dir = Path(__file__).parent.parent / 'data' / 'odds'
        odds_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = odds_dir / f'processed_nba_totals_{timestamp}.json'
        
        df.to_json(output_path, orient='records', indent=2)
        print(f"\n💾 저장: {output_path}")
        
        return output_path


def main():
    """실행"""
    api_key = "3e76069f78f461b5348b7dfdf1ff5535"
    
    fetcher = TotalsOddsFetcher(api_key)
    df = fetcher.fetch_totals_odds()
    
    if df is not None:
        print("\n=== 수집 완료 ===")
        print(df[['home_team', 'away_team', 'total_line', 'over_odds', 'under_odds']].to_string(index=False))


if __name__ == "__main__":
    main()

