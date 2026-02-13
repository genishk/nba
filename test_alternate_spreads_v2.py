#!/usr/bin/env python3
"""
Test Alternate Spreads using the new /events/{eventId}/odds endpoint
"""

import requests
import json
import time

API_KEY = "c284c82e218e82d4dd976a07e0a7b403"
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

def test_alternate_spreads_current():
    """현재 경기의 alternate spreads 테스트"""
    
    print("=" * 70)
    print("🔍 Step 1: Get current games list")
    print("=" * 70)
    
    # Step 1: 경기 목록 가져오기
    url = f"{BASE_URL}/sports/{SPORT}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'h2h',
        'bookmakers': 'fanduel'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        games = response.json()
        
        if not games:
            print("❌ No games found")
            return
        
        print(f"✅ Found {len(games)} games")
        
        # 첫 번째 경기 선택
        game = games[0]
        game_id = game['id']
        home_team = game['home_team']
        away_team = game['away_team']
        
        print(f"\n📊 Testing with game:")
        print(f"   ID: {game_id}")
        print(f"   Match: {home_team} vs {away_team}")
        
        # Step 2: 해당 경기의 alternate spreads 가져오기
        print("\n" + "=" * 70)
        print("🔍 Step 2: Get alternate spreads for this game")
        print("=" * 70)
        
        event_url = f"{BASE_URL}/sports/{SPORT}/events/{game_id}/odds"
        event_params = {
            'apiKey': API_KEY,
            'regions': 'us',
            'markets': 'alternate_spreads',
            'oddsFormat': 'american',
            'bookmakers': 'fanduel'
        }
        
        time.sleep(1)  # Rate limiting
        
        event_response = requests.get(event_url, params=event_params, timeout=10)
        event_response.raise_for_status()
        
        event_data = event_response.json()
        
        print(f"\n✅ Successfully fetched alternate spreads!")
        print(f"\nAPI requests remaining: {event_response.headers.get('x-requests-remaining', 'N/A')}")
        
        # 데이터 분석
        print("\n" + "=" * 70)
        print("📊 Alternate Spreads Data")
        print("=" * 70)
        
        for bookmaker in event_data.get('bookmakers', []):
            if bookmaker['key'] == 'fanduel':
                print(f"\n🎯 Bookmaker: {bookmaker['title']}")
                
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'alternate_spreads':
                        outcomes = market.get('outcomes', [])
                        print(f"\n📈 Total alternate spread options: {len(outcomes)}")
                        
                        # 홈팀 스프레드만 필터링
                        home_spreads = [o for o in outcomes if o['name'] == home_team]
                        away_spreads = [o for o in outcomes if o['name'] == away_team]
                        
                        print(f"\n🏠 {home_team} spreads:")
                        for spread in sorted(home_spreads, key=lambda x: x['point']):
                            print(f"   {spread['point']:+6.1f} @ {spread['price']:+4d}")
                        
                        print(f"\n✈️  {away_team} spreads:")
                        for spread in sorted(away_spreads, key=lambda x: x['point']):
                            print(f"   {spread['point']:+6.1f} @ {spread['price']:+4d}")
                        
                        # 우리가 원하는 구간 확인
                        print("\n" + "=" * 70)
                        print("🎯 Target spreads (-2.5 to -12.5):")
                        print("=" * 70)
                        
                        target_spreads = [-2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5, -10.5, -11.5, -12.5]
                        
                        for team_name, spreads in [(home_team, home_spreads), (away_team, away_spreads)]:
                            found_targets = []
                            for spread in spreads:
                                if spread['point'] in target_spreads:
                                    found_targets.append((spread['point'], spread['price']))
                            
                            if found_targets:
                                print(f"\n{team_name}:")
                                for point, price in sorted(found_targets):
                                    print(f"   {point:+6.1f} @ {price:+4d}")
        
        # 전체 응답 저장
        with open('alternate_spreads_response.json', 'w', encoding='utf-8') as f:
            json.dump(event_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 70)
        print("✅ Full response saved to: alternate_spreads_response.json")
        print("=" * 70)
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def test_historical_alternate_spreads():
    """과거 경기의 alternate spreads 테스트 (작동 여부 확인)"""
    
    print("\n\n" + "=" * 70)
    print("🔍 Testing Historical Alternate Spreads")
    print("=" * 70)
    
    # 최근 날짜로 테스트
    test_date = "2025-11-20T12:00:00Z"
    
    print(f"\nTrying to get historical alternate spreads for {test_date}")
    
    # Historical API로 경기 목록
    url = f"{BASE_URL}/historical/sports/{SPORT}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'h2h',
        'date': test_date,
        'bookmakers': 'fanduel'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        games = data if isinstance(data, list) else data.get('data', [])
        
        if not games:
            print("❌ No historical games found")
            return
        
        print(f"✅ Found {len(games)} historical games")
        
        # 첫 번째 경기로 테스트
        game = games[0]
        game_id = game['id']
        
        print(f"\nTesting with game ID: {game_id}")
        print(f"Match: {game['home_team']} vs {game['away_team']}")
        
        # Historical event endpoint 시도
        event_url = f"{BASE_URL}/historical/sports/{SPORT}/events/{game_id}/odds"
        event_params = {
            'apiKey': API_KEY,
            'regions': 'us',
            'markets': 'alternate_spreads',
            'date': test_date,
            'oddsFormat': 'american',
            'bookmakers': 'fanduel'
        }
        
        time.sleep(1)
        
        event_response = requests.get(event_url, params=event_params, timeout=30)
        event_response.raise_for_status()
        
        print("\n✅ Historical alternate spreads endpoint works!")
        print(f"API requests remaining: {event_response.headers.get('x-requests-remaining', 'N/A')}")
        
        event_data = event_response.json()
        
        # 간단히 확인
        for bookmaker in event_data.get('bookmakers', []):
            if bookmaker['key'] == 'fanduel':
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'alternate_spreads':
                        print(f"Found {len(market.get('outcomes', []))} alternate spread options")
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Historical alternate spreads NOT supported")
        print(f"Status code: {e.response.status_code}")
        print(f"This means we can only get alternate spreads for CURRENT/UPCOMING games")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Test 1: 현재 경기의 alternate spreads
    test_alternate_spreads_current()
    
    # Test 2: 과거 경기의 alternate spreads (작동 여부 확인)
    test_historical_alternate_spreads()

