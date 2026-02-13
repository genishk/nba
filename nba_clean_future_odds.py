#!/usr/bin/env python3
"""
NBA Historical Odds Cleaner
- 마스터 파일에서 미래 날짜의 배당률 제거
- 증분 알고리즘 정상화
"""

import json
import pytz
from pathlib import Path
from datetime import datetime


def clean_future_odds():
    """마스터 파일에서 미래 경기 제거"""
    
    # 파일 경로
    project_root = Path(__file__).parent
    master_file = project_root / "data" / "historical_odds" / "nba_historical_odds_fanduel_master.json"
    
    if not master_file.exists():
        print("❌ Master file not found")
        return
    
    # 동부시간 기준 오늘 날짜
    eastern_tz = pytz.timezone('US/Eastern')
    today_et = datetime.now(eastern_tz).date()
    
    print(f"📅 Today (ET): {today_et}")
    print(f"📂 Loading: {master_file}")
    
    # 데이터 로드
    with open(master_file, 'r', encoding='utf-8') as f:
        odds_data = json.load(f)
    
    print(f"📊 Total records before cleaning: {len(odds_data)}")
    
    # 미래 경기 필터링
    cleaned_data = []
    future_games = []
    
    for odds in odds_data:
        game_date_str = odds.get('date')
        if game_date_str:
            game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date()
            
            if game_date <= today_et:
                # 과거 또는 오늘 경기만 유지
                cleaned_data.append(odds)
            else:
                # 미래 경기 기록
                future_games.append({
                    'date': game_date_str,
                    'home': odds.get('home_team'),
                    'away': odds.get('away_team')
                })
    
    # 결과 출력
    print(f"\n✅ Records after cleaning: {len(cleaned_data)}")
    print(f"🗑️  Future games removed: {len(future_games)}")
    
    if future_games:
        print("\n📋 Removed future games:")
        # 날짜별로 그룹화
        from collections import defaultdict
        by_date = defaultdict(list)
        for game in future_games:
            by_date[game['date']].append(f"{game['home']} vs {game['away']}")
        
        for date in sorted(by_date.keys()):
            print(f"\n  {date}:")
            for matchup in by_date[date]:
                print(f"    - {matchup}")
    
    # 백업 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = master_file.parent / f"nba_historical_odds_fanduel_master_backup_{timestamp}.json"
    
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(odds_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Backup created: {backup_file.name}")
    
    # 정리된 데이터 저장
    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Master file updated: {master_file}")
    
    # 최신 날짜 확인
    if cleaned_data:
        latest_date = max(item['date'] for item in cleaned_data)
        print(f"\n📅 Latest date in cleaned file: {latest_date}")
    
    print("\n✅ Cleaning completed!")


if __name__ == "__main__":
    clean_future_odds()

