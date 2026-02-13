#!/usr/bin/env python3
"""
Test script to check if The-Odds-API provides alternate spreads
"""

import requests
import json

API_KEY = "c284c82e218e82d4dd976a07e0a7b403"
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

def test_current_odds():
    """현재 경기의 모든 마켓 확인"""
    url = f"{BASE_URL}/sports/{SPORT}/odds"
    
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'spreads,alternate_spreads',  # 둘 다 시도
        'oddsFormat': 'american',
        'bookmakers': 'fanduel'
    }
    
    print("=" * 70)
    print("🔍 Testing Alternate Spreads Support")
    print("=" * 70)
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print("❌ No games found")
            return
        
        # 첫 번째 경기 상세 분석
        game = data[0]
        print(f"\n📊 Game: {game['home_team']} vs {game['away_team']}")
        print(f"Game ID: {game['id']}")
        print(f"Commence: {game['commence_time']}")
        
        # 북메이커 확인
        for bookmaker in game.get('bookmakers', []):
            if bookmaker['key'] == 'fanduel':
                print(f"\n🎯 Bookmaker: {bookmaker['title']}")
                
                # 모든 마켓 출력
                for market in bookmaker.get('markets', []):
                    print(f"\n  📈 Market: {market['key']}")
                    print(f"     Outcomes: {len(market.get('outcomes', []))}")
                    
                    # Spread outcomes 상세 출력
                    for outcome in market.get('outcomes', []):
                        team = outcome.get('name', 'Unknown')
                        point = outcome.get('point', 'N/A')
                        price = outcome.get('price', 'N/A')
                        print(f"       - {team}: {point:+.1f} @ {price:+d}" if isinstance(point, (int, float)) else f"       - {team}: {point} @ {price}")
        
        # 전체 응답 저장
        with open('test_alternate_spreads_response.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 70)
        print("✅ Full response saved to: test_alternate_spreads_response.json")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

def test_available_markets():
    """사용 가능한 모든 마켓 확인"""
    url = f"{BASE_URL}/sports/{SPORT}/odds"
    
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'oddsFormat': 'american',
        'bookmakers': 'fanduel'
    }
    
    print("\n" + "=" * 70)
    print("🔍 Checking Available Markets (without market filter)")
    print("=" * 70)
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data:
            game = data[0]
            for bookmaker in game.get('bookmakers', []):
                if bookmaker['key'] == 'fanduel':
                    markets = [m['key'] for m in bookmaker.get('markets', [])]
                    print(f"\n📊 Available markets from FanDuel:")
                    for market in markets:
                        print(f"   - {market}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_current_odds()
    test_available_markets()

