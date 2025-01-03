import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 이제 src 모듈을 임포트할 수 있습니다
from src.data.processor import DataProcessor
from src.data.espn_api import ESPNNBADataCollector

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import datetime
import joblib
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NBAPredictorModel:
    def __init__(self, model_dir: Optional[Path] = None):
        """NBA 경기 예측 모델 초기화
        
        Args:
            model_dir: 모델 저장 디렉토리 (기본값: src/models/saved_models)
        """
        if model_dir is None:
            model_dir = Path(__file__).parent / "saved_models"
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """예측에 사용할 특성 준비"""
        y = (df['home_team_score'] > df['away_team_score']).astype(int)
        
        # 특성 그룹별 스케일링 정의
        scaling_groups = {
            'percentage': [col for col in df.columns if 'Pct' in col],
            'rating': [col for col in df.columns if 'rating' in col],
            'count': [col for col in df.columns if any(x in col for x in ['rebounds', 'assists', 'points'])]
        }
        
        # 그룹별 스케일링 적용
        X = df[self.feature_columns].copy()
        for group, cols in scaling_groups.items():
            group_cols = [col for col in cols if col in X.columns]
            if group_cols:
                X[group_cols] = self.scaler.fit_transform(X[group_cols])
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, float]:
        """모델 학습"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 모델 하이퍼파라미터 조정
        self.model = xgb.XGBClassifier(
            n_estimators=200,          # 증가: 더 많은 트리로 학습
            learning_rate=0.03,        # 감소: 오버피팅 방지
            max_depth=3,               # 감소: 오버피팅 방지
            reg_lambda=2.0,            # 증가: L2 정규화 강화
            reg_alpha=1.0,             # 증가: L1 정규화 강화
            subsample=0.7,             # 감소: 각 트리마다 데이터 샘플링
            colsample_bytree=0.7,      # 감소: 각 트리마다 특성 샘플링
            random_state=42,
            eval_metric='logloss'
        )
        
        # 모델 학습
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=True
        )
        
        # 모델 평가
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # 특성 중요도 계산
        importance = dict(zip(X.columns, self.model.feature_importances_))
        
        # 결과 저장
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': importance
        }
        
        logging.info(f"학습 완료:")
        logging.info(f"- 훈련 정확도: {train_score:.4f}")
        logging.info(f"- 테스트 정확도: {test_score:.4f}")
        
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """승리 확률 예측
        
        Args:
            X: 특성 데이터
            
        Returns:
            예측 확률 (홈팀 승리 확률)
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        X_scaled = self.scaler.transform(X[self.feature_columns])
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def calculate_roi(self, df: pd.DataFrame, prob_threshold: float = 0.55) -> Dict[str, float]:
        """배당률 기반 수익률 계산
        
        Args:
            df: 경기 데이터 (예측 확률과 배당률 포함)
            prob_threshold: 배팅 결정을 위한 확률 임계값
            
        Returns:
            수익률 관련 지표
        """
        # 승리 확률 예측
        df['pred_prob'] = self.predict_proba(df)
        
        # 배팅 결정 및 수익 계산
        df['bet'] = df['pred_prob'] > prob_threshold
        df['actual_win'] = df['home_team_score'] > df['away_team_score']
        
        # 배팅한 경기만 선택
        bet_games = df[df['bet']]
        
        if len(bet_games) == 0:
            return {'roi': 0, 'total_bets': 0, 'win_rate': 0}
        
        # 수익률 계산
        correct_bets = bet_games['actual_win'].sum()
        total_bets = len(bet_games)
        win_rate = correct_bets / total_bets
        
        # 가상의 고정 배당률 사용 (실제로는 odds 컬럼 사용)
        roi = (correct_bets * 1.9 - total_bets) / total_bets * 100
        
        return {
            'roi': roi,
            'total_bets': total_bets,
            'win_rate': win_rate
        }
    
    def save_model(self, filename: str = None) -> None:
        """모델 저장"""
        if filename is None:
            filename = f"nba_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        
        model_path = self.model_dir / filename
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, model_path)
        
        logging.info(f"모델 저장 완료: {model_path}")
    
    def load_model(self, filename: str) -> None:
        """모델 로드"""
        model_path = self.model_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        saved_data = joblib.load(model_path)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.feature_columns = saved_data['feature_columns']
        
        logging.info(f"모델 로드 완료: {model_path}")

# 테스트 코드
if __name__ == "__main__":
    processor = DataProcessor()
    
    # 이미 저장된 historical 데이터 로드
    data = processor.load_latest_data(data_type='historical')
    
    # 데이터 처리 및 모델 학습
    games_df = processor.process_game_data(data)
    features_df = processor.extract_features(games_df, data['team_stats'])
    
    predictor = NBAPredictorModel()
    X, y = predictor.prepare_features(features_df)
    metrics = predictor.train(X, y)
    predictor.save_model()
