#!/usr/bin/env python3
"""
Debug script to check alternate spreads response
"""

import requests
import json

API_KEY = "c284c82e218e82d4dd976a07e0a7b403"
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Step 1: Get game list for 2025-11-22
print("Step 1: Getting game list...")
url = f"{BASE_URL}/historical/sports/{SPORT}/odds"
params = {
    'apiKey': API_KEY,
    'regions': 'us',
    'markets': 'h2h',
    'date': '2025-11-22T12:00:00Z',
    'bookmakers': 'fanduel'
}

response = requests.get(url, params=params, timeout=30)
games = response.json()

if isinstance(games, dict) and 'data' in games:
    games = games['data']

print(f"Found {len(games)} games")

if games:
    game = games[0]
    game_id = game['id']
    print(f"\nTesting with: {game['home_team']} vs {game['away_team']}")
    print(f"Game ID: {game_id}")
    
    # Step 2: Get alternate spreads
    print("\nStep 2: Getting alternate spreads...")
    event_url = f"{BASE_URL}/historical/sports/{SPORT}/events/{game_id}/odds"
    event_params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'alternate_spreads',
        'oddsFormat': 'american',
        'date': '2025-11-22T12:00:00Z',
        'bookmakers': 'fanduel'
    }
    
    event_response = requests.get(event_url, params=event_params, timeout=30)
    print(f"Status code: {event_response.status_code}")
    
    event_data = event_response.json()
    
    # Save full response
    with open('debug_alternate_spreads_full.json', 'w') as f:
        json.dump(event_data, f, indent=2)
    
    print(f"\nFull response saved to: debug_alternate_spreads_full.json")
    
    # Check structure
    print(f"\nResponse keys: {event_data.keys()}")
    print(f"Bookmakers: {len(event_data.get('bookmakers', []))}")
    
    for bookmaker in event_data.get('bookmakers', []):
        print(f"\nBookmaker: {bookmaker.get('key')}")
        print(f"Markets: {[m['key'] for m in bookmaker.get('markets', [])]}")
        
        for market in bookmaker.get('markets', []):
            print(f"\n  Market: {market['key']}")
            print(f"  Outcomes: {len(market.get('outcomes', []))}")
            
            if market['key'] == 'alternate_spreads':
                outcomes = market.get('outcomes', [])
                print(f"\n  Sample outcomes (first 5):")
                for outcome in outcomes[:5]:
                    print(f"    {outcome.get('name')}: {outcome.get('point')} @ {outcome.get('price')}")

