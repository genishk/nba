import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import pandas as pd

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.espn_api import ESPNNBADataCollector
from src.data.processor import DataProcessor
from src.models.predictor import NBAPredictorModel

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LivePredictor:
    def __init__(self, model_path: Optional[Path] = None):
        """실시간 예측 시스템 초기화
        
        Args:
            model_path: 학습된 모델 파일 경로 (None이면 최신 모델 자동 로드)
        """
        self.collector = ESPNNBADataCollector()
        self.processor = DataProcessor()
        self.predictor = NBAPredictorModel()
        
        # 데이터 저장 경로 설정
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.upcoming_dir = self.data_dir / "upcoming" / "games"
        self.upcoming_dir.mkdir(parents=True, exist_ok=True)
        
        if model_path is None:
            # saved_models 폴더에서 최신 모델 찾기
            model_dir = Path(__file__).parent / "saved_models"
            model_files = list(model_dir.glob("nba_model_*.joblib"))
            if not model_files:
                raise FileNotFoundError("모델 파일을 찾을 수 없습니다")
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
            logging.info(f"최신 모델 파일 로드: {model_path.name}")
        
        self.predictor.load_model(model_path.name)
        
    def collect_upcoming_games(self, days_ahead: int = 5) -> Dict[str, Any]:
        """예정된 경기 데이터 수집"""
        logging.info(f"\n=== {days_ahead}일 후까지의 예정 경기 수집 시작 ===")
        return self.collector.collect_upcoming_data(days_ahead)  # 데이터만 반환
    
    def predict_games(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """경기 예측 수행"""
        try:
            games_df = self.processor.process_game_data(data)
            features_df = self.processor.extract_features(games_df, data['team_stats'])
            
            if features_df.empty:
                raise ValueError("처리된 경기 데이터가 없습니다")
            
            # 필요한 특성 확인
            missing_features = [col for col in self.predictor.feature_columns 
                              if col not in features_df.columns]
            if missing_features:
                logging.warning(f"누락된 특성: {missing_features}")
                # 누락된 특성을 0으로 채움
                for col in missing_features:
                    features_df[col] = 0
            
            predictions = []
            for idx, game in features_df.iterrows():
                try:
                    win_prob = self.predictor.predict_proba(game.to_frame().T)[0]
                    confidence = self._adjust_confidence(win_prob, game)
                    
                    prediction = {
                        'game_id': game['game_id'],
                        'date': game['date'],
                        'home_team': game['home_team_name'],
                        'away_team': game['away_team_name'],
                        'home_win_probability': win_prob,
                        'prediction': 'Home Win' if win_prob > 0.5 else 'Away Win',
                        'confidence': confidence
                    }
                    predictions.append(prediction)
                    
                except Exception as e:
                    logging.error(f"예측 실패 (game_id: {game['game_id']}): {str(e)}")
                    continue
            
            return {
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"예측 프로세스 실패: {str(e)}")
            return {'predictions': [], 'timestamp': datetime.now().isoformat()}
    
    def _adjust_confidence(self, 
                          raw_prob: float,
                          game_data: pd.Series) -> float:
        """확신도 보정"""
        # 1. 최근 성적 반영
        form_diff = abs(game_data['home_recent_form'] - game_data['away_recent_form'])
        form_factor = 1 + (form_diff * 0.1)
        
        # 2. 부상자 수 반영
        injury_diff = game_data['away_injuries_count'] - game_data['home_injuries_count']
        injury_factor = 1 + (injury_diff * 0.03)
        
        # 3. 승률 차이 반영
        win_rate_diff = abs(
            game_data['home_overall_record_win_rate'] - 
            game_data['away_overall_record_win_rate']
        )
        win_rate_factor = 1 + (win_rate_diff * 0.15)
        
        # 4. 최종 확신도 계산
        confidence = raw_prob * form_factor * injury_factor * win_rate_factor
        
        # 5. 최소 확신도만 제한
        return max(confidence, 0.55)
    
    def calculate_betting_value(self, predictions: List[Dict], min_confidence: float = 0.6) -> List[Dict]:
        """베팅 가치 분석"""
        valuable_bets = []
        
        for pred in predictions:
            # 높은 확신도의 경기만 선택
            if pred['confidence'] >= min_confidence:
                valuable_bets.append({
                    **pred,
                    'bet_recommendation': 'High Value'
                })
        
        return valuable_bets

# 테스트 코드
if __name__ == "__main__":
    # 모델 파일 경로를 지정하지 않으면 최신 모델 자동 로드
    live_predictor = LivePredictor()
    
    # 예정된 경기 데이터 수집
    upcoming_games = live_predictor.collect_upcoming_games(days_ahead=5)
    
    # 예측 수행
    prediction_results = live_predictor.predict_games(upcoming_games)
    
    # 베팅 가치 분석
    valuable_bets = live_predictor.calculate_betting_value(
        prediction_results['predictions']
    )
    
    # 결과 출력
    print("\n=== 높은 가치의 베팅 추천 ===")
    for bet in valuable_bets:
        print(f"\n{bet['away_team']} @ {bet['home_team']}")
        print(f"예측: {bet['prediction']} (확신도: {bet['confidence']:.1%})") 