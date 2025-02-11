import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import itertools

class BettingOptimizer:
    def __init__(self):
        self.predictions_dir = Path(__file__).parent.parent / "predictions"
        self.analysis_dir = Path(__file__).parent.parent / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        self.max_parlay_size = 3  # 최대 3팀 파라레이
        
    def load_latest_predictions(self) -> pd.DataFrame:
        """최신 앙상블 예측 결과 로드"""
        pred_files = list(self.predictions_dir.glob("ensemble_predictions_*.json"))
        if not pred_files:
            raise FileNotFoundError("예측 결과 파일을 찾을 수 없습니다.")
        
        latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, 'r') as f:
            predictions = pd.DataFrame(json.load(f))
        return predictions
    
    def calculate_ev(self, probability: float, odds: float) -> float:
        """Expected Value 계산"""
        return (probability * (odds - 1)) - (1 - probability)
    
    def calculate_parlay_odds(self, individual_odds: List[float]) -> float:
        """파라레이 배당률 계산"""
        return np.prod(individual_odds)
    
    def calculate_parlay_probability(self, individual_probs: List[float]) -> float:
        """파라레이 승리 확률 계산"""
        return np.prod(individual_probs)
    
    def generate_parlays(self, single_bets: pd.DataFrame) -> List[Dict]:
        """가능한 파라레이 조합 생성"""
        # 승률 조건 완화
        confident_bets = single_bets[
            (single_bets['win_probability'] >= 0.5) &  # 50% 이상으로 수정
            (single_bets['expected_value'] > 0)
        ]
        parlays = []
        
        # 2팀과 3팀 파라레이 조합 생성
        for size in range(2, self.max_parlay_size + 1):
            for combo in itertools.combinations(confident_bets.itertuples(), size):
                # 파라레이 배당률과 확률 계산
                odds = self.calculate_parlay_odds([bet.betting_odds for bet in combo])
                prob = self.calculate_parlay_probability([bet.win_probability for bet in combo])
                
                # EV 계산
                ev = self.calculate_ev(prob, odds)
                
                # Kelly Criterion 계산 (파라레이는 더 보수적으로)
                if ev > 0:
                    kelly = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
                    kelly = max(0, min(0.1, kelly))  # 파라레이는 더 낮은 상한선
                else:
                    kelly = 0
                
                parlays.append({
                    'type': f'{size}_team_parlay',
                    'matches': [bet.match for bet in combo],
                    'teams': [bet.team for bet in combo],
                    'win_probability': prob,
                    'odds': odds,
                    'expected_value': ev,
                    'kelly_fraction': kelly,
                    'individual_probabilities': [bet.win_probability for bet in combo],
                    'individual_odds': [bet.betting_odds for bet in combo]
                })
        
        return parlays
    
    def optimize_portfolio(self, single_bets: pd.DataFrame, parlays: List[Dict], 
                         bankroll: float = 1000) -> Dict:
        portfolio = {
            'singles': [],
            'parlays': [],
            'total_investment': 0,
            'expected_profit': 0,
            'max_loss': 0
        }
        
        # 자금 배분 (80% 사용)
        available_bankroll = bankroll * 0.8
        singles_allocation = available_bankroll * 0.5  # 50%로 수정
        parlays_allocation = available_bankroll * 0.3  # 30%로 수정
        
        # 단일 베팅 선택 및 할당
        current_investment = 0
        good_singles = single_bets[
            (single_bets['win_probability'] >= 0.5) &  # 50% 이상으로 수정
            (single_bets['expected_value'] > 0)
        ].sort_values('expected_value', ascending=False)
        
        # 단일 베팅 처리
        for _, bet in good_singles.iterrows():
            bet_amount = min(
                singles_allocation * bet['kelly_fraction'],
                singles_allocation - current_investment
            )
            if bet_amount < 1:
                continue
            
            true_expected_profit = self.calculate_true_expected_profit(
                bet_amount=bet_amount,
                odds=bet['betting_odds'],
                probability=bet['win_probability']
            )
            
            portfolio['singles'].append({
                'match': bet['match'],
                'team': bet['team'],
                'probability': bet['win_probability'],
                'odds': bet['betting_odds'],
                'amount': bet_amount,
                'potential_profit': bet_amount * (bet['betting_odds'] - 1),
                'expected_profit': true_expected_profit,
                'risk_amount': bet_amount
            })
            current_investment += bet_amount
        
        # 2팀 파라레이 선택 (중복 없이, 최대 5개)
        good_parlays_2team = []
        used_matches_2team = set()
        sorted_2team = sorted(
            [p for p in parlays if p['expected_value'] > 0 and p['type'] == '2_team_parlay'],
            key=lambda x: x['expected_value'],
            reverse=True
        )
        
        for parlay in sorted_2team:
            matches = set(parlay['matches'])
            if not matches & used_matches_2team:  # 겹치는 경기가 없는 경우
                good_parlays_2team.append(parlay)
                used_matches_2team.update(matches)
                if len(good_parlays_2team) >= 5:  # 최대 5개로 수정
                    break
        
        # 3팀 파라레이 선택 (1팀까지 중복 허용, 최대 5개)
        good_parlays_3team = []
        used_matches_count = {}  # 각 경기의 사용 횟수 추적
        sorted_3team = sorted(
            [p for p in parlays if p['expected_value'] > 0 and p['type'] == '3_team_parlay'],
            key=lambda x: x['expected_value'],
            reverse=True
        )
        
        for parlay in sorted_3team:
            # 현재 파라레이와 기존 선택된 파라레이들 간의 중복 경기 수 확인
            overlap_count = 0
            for existing_parlay in good_parlays_3team:
                common_matches = set(parlay['matches']) & set(existing_parlay['matches'])
                overlap_count = max(overlap_count, len(common_matches))
            
            # 중복 경기가 1개 이하인 경우에만 추가
            if overlap_count <= 1:
                good_parlays_3team.append(parlay)
                if len(good_parlays_3team) >= 5:  # 최대 5개
                    break
        
        good_parlays = good_parlays_2team + good_parlays_3team
        
        # 파라레이 자금 배분
        parlay_investment = 0
        if good_parlays:
            per_parlay_allocation = parlays_allocation / len(good_parlays)
            
            for parlay in good_parlays:
                bet_amount = min(
                    per_parlay_allocation,
                    parlays_allocation - parlay_investment
                )
                
                if bet_amount < 1:
                    continue
                
                # 프로모션이 적용된 배당률 계산 수정
                base_odds = parlay['odds']
                boosted_odds = base_odds
                if len(parlay['matches']) == 2:
                    # 2팀 파라레이: 순수익(odds-1)에 10% 부스트
                    boosted_odds = 1 + ((base_odds - 1) * 1.10)
                elif len(parlay['matches']) == 3:
                    # 3팀 파라레이: 순수익(odds-1)에 20% 부스트
                    boosted_odds = 1 + ((base_odds - 1) * 1.20)
                
                # 실제 기대 수익 계산
                true_expected_profit = self.calculate_true_expected_profit(
                    bet_amount=bet_amount,
                    odds=parlay['odds'],
                    probability=parlay['win_probability'],
                    is_parlay=True,
                    parlay_size=len(parlay['matches'])
                )
                
                portfolio['parlays'].append({
                    'type': parlay['type'],
                    'matches': parlay['matches'],
                    'teams': parlay['teams'],
                    'probability': parlay['win_probability'],
                    'odds': parlay['odds'],
                    'boosted_odds': boosted_odds,
                    'amount': bet_amount,
                    'potential_profit': bet_amount * (boosted_odds - 1),
                    'expected_profit': true_expected_profit,
                    'risk_amount': bet_amount
                })
                parlay_investment += bet_amount
        
        # 포트폴리오 전체 통계
        portfolio['total_investment'] = current_investment + parlay_investment
        portfolio['max_loss'] = portfolio['total_investment']
        portfolio['expected_profit'] = sum(bet['expected_profit'] for bet in portfolio['singles']) + \
                                     sum(bet['expected_profit'] for bet in portfolio['parlays'])
        
        return portfolio
    
    def analyze_and_save(self, odds_data: Dict[str, str], bankroll: float = 1000) -> Dict:
        # 배당률 데이터 포맷 변환
        formatted_odds = self.format_odds_data(odds_data)
        
        # 기존 분석 로직 실행
        predictions = self.load_latest_predictions()
        single_bets = self.optimize_bets(predictions, formatted_odds, bankroll)
        
        # 파라레이 생성
        parlays = self.generate_parlays(single_bets)
        
        # 포트폴리오 최적화
        portfolio = self.optimize_portfolio(single_bets, parlays, bankroll)
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.analysis_dir / f"betting_analysis_{timestamp}.json"
        
        # 결과 출력
        print("\n=== 베팅 포트폴리오 추천 ===")
        print(f"총 투자금액: ${portfolio['total_investment']:.2f}")
        print(f"최대 손실 가능액: -${portfolio['max_loss']:.2f}")
        print(f"실제 기대 수익: ${portfolio['expected_profit']:.2f} ({portfolio['expected_profit']/portfolio['total_investment']*100:.1f}%)")
        
        if portfolio['singles']:
            print("\n=== 단일 베팅 추천 ===")
            for bet in portfolio['singles']:
                print(f"\n{bet['match']}")
                print(f"베팅: {bet['team']}")
                print(f"배당률: {bet['odds']:.2f}")
                print(f"승률: {bet['probability']:.1%}")
                print(f"베팅액: ${bet['amount']:.2f}")
                print(f"성공 시 수익: ${bet['potential_profit']:.2f}")
                print(f"실패 시 손실: -${bet['risk_amount']:.2f}")
                print(f"실제 기대 수익: ${bet['expected_profit']:.2f} ({bet['expected_profit']/bet['amount']*100:.1f}%)")
        
        if portfolio['parlays']:
            # 2팀 파라레이 출력
            two_team_parlays = [p for p in portfolio['parlays'] if p['type'] == '2_team_parlay']
            if two_team_parlays:  # 2팀 파라레이가 있는 경우에만 출력
                print("\n=== 2팀 파라레이 베팅 추천 ===")
                print(f"(총 {len(two_team_parlays)}개 조합 추천)")  # 가능한 조합 수 표시
                for parlay in two_team_parlays:
                    print(f"\n{parlay['type'].upper()}")
                    for match, team in zip(parlay['matches'], parlay['teams']):
                        print(f"- {match}: {team}")
                    print(f"기본 배당률: {parlay['odds']:.2f}")
                    print(f"프로모션 적용 배당률: {parlay['boosted_odds']:.2f}")
                    print(f"승률: {parlay['probability']:.1%}")
                    print(f"베팅액: ${parlay['amount']:.2f}")
                    print(f"성공 시 수익: ${parlay['potential_profit']:.2f}")
                    print(f"실패 시 손실: -${parlay['risk_amount']:.2f}")
                    print(f"실제 기대 수익: ${parlay['expected_profit']:.2f} ({parlay['expected_profit']/parlay['amount']*100:.1f}%)")
            
            # 3팀 파라레이 출력
            three_team_parlays = [p for p in portfolio['parlays'] if p['type'] == '3_team_parlay'][:3]
            if three_team_parlays:  # 3팀 파라레이가 있는 경우에만 출력
                print("\n=== 3팀 파라레이 베팅 추천 ===")
                print(f"(총 {len(three_team_parlays)}개 조합 추천)")  # 가능한 조합 수 표시
                for parlay in three_team_parlays:
                    print(f"\n{parlay['type'].upper()}")
                    for match, team in zip(parlay['matches'], parlay['teams']):
                        print(f"- {match}: {team}")
                    print(f"기본 배당률: {parlay['odds']:.2f}")
                    print(f"프로모션 적용 배당률: {parlay['boosted_odds']:.2f}")
                    print(f"승률: {parlay['probability']:.1%}")
                    print(f"베팅액: ${parlay['amount']:.2f}")
                    print(f"성공 시 수익: ${parlay['potential_profit']:.2f}")
                    print(f"실패 시 손실: -${parlay['risk_amount']:.2f}")
                    print(f"실제 기대 수익: ${parlay['expected_profit']:.2f} ({parlay['expected_profit']/parlay['amount']*100:.1f}%)")
        
        # JSON으로 저장
        with open(output_path, 'w') as f:
            json.dump(portfolio, f, indent=2)
        
        print(f"\n분석 결과 저장 완료: {output_path}")
        
        # 포트폴리오 반환 추가
        return portfolio  # 이 부분이 추가됨

    def format_odds_data(self, odds_data: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """팀별 배당률을 경기별 배당률로 변환"""
        # 예측 데이터에서 경기 일정 가져오기
        predictions = self.load_latest_predictions()
        formatted_odds = {}
        
        for _, game in predictions.iterrows():
            home_team = game['home_team_name']
            away_team = game['away_team_name']
            match_id = f"{home_team} vs {away_team}"
            
            # 해당 경기 팀들의 배당이 있는 경우에만 추가
            if home_team in odds_data and away_team in odds_data:
                formatted_odds[match_id] = {
                    "home": odds_data[home_team],
                    "away": odds_data[away_team]
                }
        
        return formatted_odds

    def convert_american_to_decimal(self, american_odds: str) -> float:
        """미국식 배당률을 소수점 배당률로 변환"""
        odds = int(american_odds)
        if odds > 0:
            return 1 + (odds / 100)
        else:
            return 1 + (100 / abs(odds))

    def optimize_bets(self, predictions: pd.DataFrame, odds_data: Dict[str, Dict[str, str]], 
                     bankroll: float = 1000) -> pd.DataFrame:
        results = []
        
        for _, game in predictions.iterrows():
            game_id = f"{game['home_team_name']} vs {game['away_team_name']}"
            if game_id not in odds_data:
                continue
            
            # 양팀 배당 가져오기
            home_odds = self.convert_american_to_decimal(odds_data[game_id]['home'])
            away_odds = self.convert_american_to_decimal(odds_data[game_id]['away'])
            
            # 홈팀 베팅 분석
            home_prob = game['home_win_probability']
            home_ev = self.calculate_ev(home_prob, home_odds)
            if home_ev > 0:
                home_kelly = (home_prob * (home_odds - 1) - (1 - home_prob)) / (home_odds - 1)
                home_kelly = max(0, min(0.2, home_kelly))
            else:
                home_kelly = 0
            
            # 원정팀 베팅 분석
            away_prob = 1 - game['home_win_probability']
            away_ev = self.calculate_ev(away_prob, away_odds)
            if away_ev > 0:
                away_kelly = (away_prob * (away_odds - 1) - (1 - away_prob)) / (away_odds - 1)
                away_kelly = max(0, min(0.2, away_kelly))
            else:
                away_kelly = 0
            
            # 홈팀 베팅 추가
            results.append({
                'date': game['date'],
                'match': game_id,
                'team': game['home_team_name'],
                'team_side': 'home',
                'win_probability': home_prob,
                'betting_odds': home_odds,
                'expected_value': home_ev,
                'kelly_fraction': home_kelly
            })
            
            # 원정팀 베팅 추가
            results.append({
                'date': game['date'],
                'match': game_id,
                'team': game['away_team_name'],
                'team_side': 'away',
                'win_probability': away_prob,
                'betting_odds': away_odds,
                'expected_value': away_ev,
                'kelly_fraction': away_kelly
            })
        
        return pd.DataFrame(results)

    def calculate_true_expected_profit(self, bet_amount: float, odds: float, 
                                     probability: float, is_parlay: bool = False, 
                                     parlay_size: int = 0) -> float:
        """실제 기대 수익 계산 (승패 모두 고려)"""
        # 기본 배당률에 프로모션 적용 (순수익 부분에만 부스트 적용)
        boosted_odds = odds
        if is_parlay:
            if parlay_size == 2:
                # 2팀 파라레이: 순수익(odds-1)에 10% 부스트
                boosted_odds = 1 + ((odds - 1) * 1.10)
            elif parlay_size == 3:
                # 3팀 파라레이: 순수익(odds-1)에 20% 부스트
                boosted_odds = 1 + ((odds - 1) * 1.20)
        
        # 승리 시 수익
        win_profit = bet_amount * (boosted_odds - 1)
        # 패배 시 손실 (배팅금액 전체)
        loss_amount = bet_amount
        
        # 기대 수익 = (승리확률 × 승리수익) - (패배확률 × 손실금액)
        expected_profit = (probability * win_profit) - ((1 - probability) * loss_amount)
        
        return expected_profit

    def select_diverse_parlays(self, parlays: List[Dict], max_overlap: int = 1) -> List[Dict]:
        """중복을 최소화한 파라레이 선택"""
        selected = []
        used_matches = {}  # 각 경기가 사용된 횟수 추적
        
        # EV 순으로 정렬
        sorted_parlays = sorted(parlays, key=lambda x: x['expected_value'], reverse=True)
        
        for parlay in sorted_parlays:
            # 현재 파라레이의 경기들이 이미 max_overlap 이상 사용됐는지 확인
            overlap_check = [
                used_matches.get(match, 0) < max_overlap 
                for match in parlay['matches']
            ]
            
            # 모든 경기가 제한 횟수 이하로 사용된 경우에만 선택
            if all(overlap_check):
                selected.append(parlay)
                # 사용 횟수 업데이트
                for match in parlay['matches']:
                    used_matches[match] = used_matches.get(match, 0) + 1
        
        return selected 