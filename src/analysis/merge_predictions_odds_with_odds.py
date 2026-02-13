import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import argparse


def is_jupyter():
    """Jupyter 환경인지 확인"""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


class PredictionsOddsMergerWithOdds:
    """배당 포함 모델(with_odds) 전용 예측-배당 병합 클래스"""
    
    def __init__(self):
        """예측 결과와 배당률 데이터를 병합하는 클래스 (with_odds 모델 전용)"""
        self.project_root = Path(__file__).parent.parent.parent
        self.predictions_dir = self.project_root / "src" / "predictions"
        self.odds_dir = self.project_root / "data" / "odds"
        self.output_dir = self.project_root / "src" / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        self.model_tag = 'with_odds'
        
        # 팀명 매핑 (전체 이름 → 약자)
        self.team_name_to_abbrev = {
            'Hawks': 'ATL',
            'Celtics': 'BOS',
            'Nets': 'BKN',
            'Hornets': 'CHA',
            'Bulls': 'CHI',
            'Cavaliers': 'CLE',
            'Mavericks': 'DAL',
            'Nuggets': 'DEN',
            'Pistons': 'DET',
            'Warriors': 'GSW',
            'Rockets': 'HOU',
            'Pacers': 'IND',
            'Clippers': 'LAC',
            'Lakers': 'LAL',
            'Grizzlies': 'MEM',
            'Heat': 'MIA',
            'Bucks': 'MIL',
            'Timberwolves': 'MIN',
            'Pelicans': 'NOP',
            'Knicks': 'NYK',
            'Thunder': 'OKC',
            'Magic': 'ORL',
            '76ers': 'PHI',
            'Suns': 'PHX',
            'Trail Blazers': 'POR',
            'Kings': 'SAC',
            'Spurs': 'SAS',
            'Raptors': 'TOR',
            'Jazz': 'UTA',
            'Wizards': 'WAS'
        }
    
    def load_latest_predictions(self) -> List[Dict]:
        """최신 with_odds 앙상블 예측 파일 로드"""
        # with_odds 태그 파일 찾기
        pred_files = list(self.predictions_dir.glob("ensemble_predictions_*_with_odds.json"))
        
        if not pred_files:
            raise FileNotFoundError(
                f"[WITH_ODDS] 예측 파일을 찾을 수 없습니다: {self.predictions_dir}\n"
                "predict_ensemble_with_odds.py를 먼저 실행해주세요."
            )
        
        latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
        print(f"\n[WITH_ODDS] 예측 파일 로드: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_latest_odds(self) -> List[Dict]:
        """최신 배당률 파일 로드"""
        odds_files = list(self.odds_dir.glob("processed_nba_odds_*.json"))
        if not odds_files:
            raise FileNotFoundError(f"배당률 파일을 찾을 수 없습니다: {self.odds_dir}")
        
        latest_file = max(odds_files, key=lambda x: x.stat().st_mtime)
        print(f"배당률 파일 로드: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def merge_data(self) -> List[Dict]:
        """예측 데이터와 배당률 데이터 병합"""
        predictions = self.load_latest_predictions()
        odds_data = self.load_latest_odds()
        
        # Odds 데이터를 경기별로 그룹화 (game_id 기준)
        odds_by_game = {}
        for odds in odds_data:
            game_id = odds['game_id']
            if game_id not in odds_by_game:
                odds_by_game[game_id] = {
                    'home_team': odds['home_team'],
                    'away_team': odds['away_team'],
                    'date': odds['date']
                }
            
            # 홈팀 또는 원정팀 odds 저장
            if odds['is_home']:
                odds_by_game[game_id]['home_odds'] = odds['odds']
                odds_by_game[game_id]['home_odds_probability'] = odds['probability']
            else:
                odds_by_game[game_id]['away_odds'] = odds['odds']
                odds_by_game[game_id]['away_odds_probability'] = odds['probability']
        
        # 병합 수행
        merged_data = []
        matched_count = 0
        unmatched_predictions = []
        
        for pred in predictions:
            home_team_name = pred['home_team_name']
            away_team_name = pred['away_team_name']
            
            # 팀명을 약자로 변환
            home_abbrev = self.team_name_to_abbrev.get(home_team_name)
            away_abbrev = self.team_name_to_abbrev.get(away_team_name)
            
            if not home_abbrev or not away_abbrev:
                print(f"⚠️  팀명 매핑 실패: {home_team_name} vs {away_team_name}")
                unmatched_predictions.append(pred)
                continue
            
            # Odds 데이터에서 매칭되는 경기 찾기
            matched_game = None
            matched_game_id = None
            
            for game_id, game_odds in odds_by_game.items():
                if (game_odds['home_team'] == home_abbrev and 
                    game_odds['away_team'] == away_abbrev):
                    matched_game = game_odds
                    matched_game_id = game_id
                    break
            
            if matched_game:
                # 병합된 레코드 생성
                merged_record = {
                    'date': pred['date'],
                    'game_id': matched_game_id,
                    
                    # 팀 정보
                    'home_team_name': home_team_name,
                    'away_team_name': away_team_name,
                    'home_team_abbrev': home_abbrev,
                    'away_team_abbrev': away_abbrev,
                    
                    # 앙상블 예측
                    'predicted_winner': pred['predicted_winner'],
                    'home_win_probability': pred['home_win_probability'],
                    'away_win_probability': 1 - pred['home_win_probability'],
                    
                    # 개별 모델 예측
                    'model1_home_win_prob': pred.get('model1_home_win_prob'),
                    'model2_home_win_prob': pred.get('model2_home_win_prob'),
                    'model3_home_win_prob': pred.get('model3_home_win_prob'),
                    'model4_home_win_prob': pred.get('model4_home_win_prob'),
                    'model5_home_win_prob': pred.get('model5_home_win_prob'),
                    'model6_home_win_prob': pred.get('model6_home_win_prob'),
                    'model7_home_win_prob': pred.get('model7_home_win_prob'),
                    'model8_home_win_prob': pred.get('model8_home_win_prob'),
                    
                    # 배당률 (예측에 이미 포함된 값 사용)
                    'home_odds': pred.get('home_odds') or matched_game.get('home_odds'),
                    'away_odds': pred.get('away_odds') or matched_game.get('away_odds'),
                    'home_odds_probability': matched_game.get('home_odds_probability'),
                    'away_odds_probability': matched_game.get('away_odds_probability'),
                    
                    # 배당 버킷 정보 (with_odds 모델 전용)
                    'home_odds_bucket': pred.get('home_odds_bucket'),
                    'away_odds_bucket': pred.get('away_odds_bucket')
                }
                
                merged_data.append(merged_record)
                matched_count += 1
            else:
                print(f"⚠️  매칭 실패: {home_team_name} vs {away_team_name}")
                unmatched_predictions.append(pred)
        
        # 병합 결과 요약
        print(f"\n=== [WITH_ODDS] 병합 완료 ===")
        print(f"총 예측 경기 수: {len(predictions)}")
        print(f"매칭 성공: {matched_count}개")
        print(f"매칭 실패: {len(unmatched_predictions)}개")
        
        if unmatched_predictions:
            print("\n매칭되지 않은 경기:")
            for pred in unmatched_predictions:
                print(f"  - {pred['home_team_name']} vs {pred['away_team_name']}")
        
        return merged_data
    
    def save_merged_data(self, merged_data: List[Dict]) -> Path:
        """병합된 데이터 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"merged_predictions_odds_{timestamp}_with_odds.json"
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[WITH_ODDS] 병합 데이터 저장 완료: {output_path}")
        return output_path
    
    def display_merged_data(self, merged_data: List[Dict]):
        """병합된 데이터 출력"""
        print("\n=== [WITH_ODDS] 병합된 데이터 미리보기 ===")
        
        # 배당 버킷 설명
        bucket_descriptions = {
            0: '압도적 페이버릿 (<-400)',
            1: '강한 페이버릿 (-400~-250)',
            2: '페이버릿 (-250~-150)',
            3: '약한 페이버릿 (-150~-100)',
            4: '약한 언더독 (-100~+150)',
            5: '언더독 (+150~+250)',
            6: '강한 언더독 (+250~+400)',
            7: '압도적 언더독 (>+400)'
        }
        
        for game in merged_data:
            print(f"\n📅 {game['date']} - {game['home_team_name']} vs {game['away_team_name']}")
            print(f"   예상 승자: {game['predicted_winner']}")
            print(f"   앙상블 확률: 홈 {game['home_win_probability']:.1%} / 원정 {game['away_win_probability']:.1%}")
            
            # 개별 모델 확률 표시
            print(f"   개별 모델:")
            model_names = {
                'model1': 'LightGBM',
                'model2': 'CatBoost',
                'model3': 'XGBoost',
                'model4': 'LightGBM-DART',
                'model5': 'CatBoost-Ordered',
                'model6': 'XGBoost-Hist',
                'model7': 'RandomForest',
                'model8': 'ExtraTrees'
            }
            for i in range(1, 9):
                key = f'model{i}_home_win_prob'
                if game.get(key) is not None and game[key] != 0:
                    print(f"     - {model_names[f'model{i}']}: 홈 {game[key]:.1%}")
            
            # 배당률 정보
            home_odds = game.get('home_odds')
            away_odds = game.get('away_odds')
            if home_odds and away_odds:
                print(f"   배당률: 홈 {home_odds:+.0f} / 원정 {away_odds:+.0f}")
            
            if game.get('home_odds_probability') and game.get('away_odds_probability'):
                print(f"   배당 확률: 홈 {game['home_odds_probability']:.1%} / 원정 {game['away_odds_probability']:.1%}")
            
            # 배당 버킷 정보 (with_odds 전용)
            home_bucket = game.get('home_odds_bucket')
            away_bucket = game.get('away_odds_bucket')
            if home_bucket is not None and away_bucket is not None:
                print(f"   배당 버킷: 홈 {int(home_bucket)} ({bucket_descriptions.get(int(home_bucket), '?')}) / "
                      f"원정 {int(away_bucket)} ({bucket_descriptions.get(int(away_bucket), '?')})")


def run_merge() -> Path:
    """with_odds 병합 실행"""
    print("\n" + "="*70)
    print("🏀 [WITH_ODDS] 배당 포함 모델 예측-배당 병합")
    print("="*70)
    
    merger = PredictionsOddsMergerWithOdds()
    
    # 데이터 병합
    merged_data = merger.merge_data()
    
    # 결과 출력
    merger.display_merged_data(merged_data)
    
    # 저장
    output_path = merger.save_merged_data(merged_data)
    
    print(f"\n✅ [WITH_ODDS] 병합 완료! 총 {len(merged_data)}개 경기")
    return output_path


def main():
    """메인 실행 함수"""
    try:
        output_path = run_merge()
        print(f"\n저장 위치: {output_path}")
        
    except FileNotFoundError as e:
        print(f"\n❌ 파일을 찾을 수 없습니다: {str(e)}")
        raise
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    main()

