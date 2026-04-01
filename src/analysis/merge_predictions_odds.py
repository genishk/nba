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


class PredictionsOddsMerger:
    def __init__(self, model_tag: str = 'active'):
        """예측 결과와 배당률 데이터를 병합하는 클래스
        
        Args:
            model_tag: 모델 태그 ('active', 'shadow', 'fixed')
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.predictions_dir = self.project_root / "src" / "predictions"
        self.odds_dir = self.project_root / "data" / "odds"
        self.output_dir = self.project_root / "src" / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        self.model_tag = model_tag
        
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
        """최신 앙상블 예측 파일 로드 (태그 기반)"""
        # 태그 기반 파일 먼저 찾기
        pred_files = list(self.predictions_dir.glob(f"ensemble_predictions_*_{self.model_tag}.json"))
        
        # 태그 파일이 없으면 일반 파일로 fallback
        if not pred_files:
            pred_files = list(self.predictions_dir.glob("ensemble_predictions_*.json"))
            # 태그가 포함된 파일 제외 (active, shadow, with_odds)
            pred_files = [f for f in pred_files 
                         if not f.stem.endswith('_active') and not f.stem.endswith('_shadow')
                         and 'with_odds' not in f.name]
        
        if not pred_files:
            raise FileNotFoundError(f"예측 파일을 찾을 수 없습니다: {self.predictions_dir}")
        
        latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
        print(f"\n[{self.model_tag.upper()}] 예측 파일 로드: {latest_file.name}")
        
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
                    
                    # 배당률
                    'home_odds': matched_game.get('home_odds'),
                    'away_odds': matched_game.get('away_odds'),
                    'home_odds_probability': matched_game.get('home_odds_probability'),
                    'away_odds_probability': matched_game.get('away_odds_probability')
                }
                
                merged_data.append(merged_record)
                matched_count += 1
            else:
                print(f"⚠️  매칭 실패: {home_team_name} vs {away_team_name}")
                unmatched_predictions.append(pred)
        
        # 병합 결과 요약
        print(f"\n=== 병합 완료 ===")
        print(f"총 예측 경기 수: {len(predictions)}")
        print(f"매칭 성공: {matched_count}개")
        print(f"매칭 실패: {len(unmatched_predictions)}개")
        
        if unmatched_predictions:
            print("\n매칭되지 않은 경기:")
            for pred in unmatched_predictions:
                print(f"  - {pred['home_team_name']} vs {pred['away_team_name']}")
        
        return merged_data
    
    def save_merged_data(self, merged_data: List[Dict]) -> Path:
        """병합된 데이터 저장 (태그 포함)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"merged_predictions_odds_{timestamp}_{self.model_tag}.json"
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[{self.model_tag.upper()}] 병합 데이터 저장 완료: {output_path}")
        return output_path
    
    def display_merged_data(self, merged_data: List[Dict]):
        """병합된 데이터 출력"""
        print("\n=== 병합된 데이터 미리보기 ===")
        
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
            
            print(f"   배당률: 홈 {game['home_odds']:+d} / 원정 {game['away_odds']:+d}")
            print(f"   배당 확률: 홈 {game['home_odds_probability']:.1%} / 원정 {game['away_odds_probability']:.1%}")


def parse_args():
    """커맨드라인 인자 파싱"""
    # Jupyter 환경에서는 기본값 사용 (둘 다 실행)
    if is_jupyter():
        class Args:
            model_tag = 'both'
        return Args()
    
    parser = argparse.ArgumentParser(description='NBA 예측-배당률 병합')
    parser.add_argument(
        '--model-tag', '-t',
        type=str,
        choices=['active', 'shadow', 'fixed', 'both'],
        default='both',
        help='사용할 모델 태그 (both: 둘 다 실행(기본값), active: 운영모델, shadow: 테스트모델, fixed: 고정모델)'
    )
    return parser.parse_args()


def run_merge(model_tag: str = 'active') -> Path:
    """단일 태그 병합 실행"""
    merger = PredictionsOddsMerger(model_tag=model_tag)
    
    # 데이터 병합
    merged_data = merger.merge_data()
    
    # 결과 출력
    merger.display_merged_data(merged_data)
    
    # 저장
    output_path = merger.save_merged_data(merged_data)
    
    print(f"\n✅ [{model_tag.upper()}] 병합 완료! 총 {len(merged_data)}개 경기")
    return output_path


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    try:
        if args.model_tag == 'both':
            # Active와 Shadow 둘 다 실행
            print("\n" + "="*60)
            print("🏀 [ACTIVE 병합 시작]")
            print("="*60)
            output_active = run_merge('active')
            
            print("\n" + "="*60)
            print("🌙 [SHADOW 병합 시작]")
            print("="*60)
            output_shadow = run_merge('shadow')
            
            print("\n" + "="*60)
            print("✅ Active & Shadow 병합 완료!")
            print(f"  - Active: {output_active}")
            print(f"  - Shadow: {output_shadow}")
            print("="*60)
        else:
            run_merge(args.model_tag)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    main()

