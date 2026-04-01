# src/models/model6_totals.py
# XGBoost Histogram Regressor - Total Score Prediction
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


class BettingModel6Totals:
    """XGBoost Histogram 회귀 모델 - 총점 예측"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_dir = Path(__file__).parent / "saved_models"
        self.model_dir.mkdir(exist_ok=True)
        self.dates = None
    
    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터에서 특성과 타겟(총점) 추출"""
        df = pd.DataFrame(data)
        
        # 날짜 기준으로 정렬
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        self.dates = df['date']
        
        # 타겟: 총점 (홈팀 + 원정팀 점수)
        y = df['home_team_score'].astype(float) + df['away_team_score'].astype(float)
        
        # 기본 특성 선택
        base_features = [
            # 기본 경기력 지표
            'home_rebounds', 'away_rebounds',
            'home_assists', 'away_assists',
            'home_fieldGoalsAttempted', 'away_fieldGoalsAttempted',
            'home_fieldGoalsMade', 'away_fieldGoalsMade',
            'home_fieldGoalPct', 'away_fieldGoalPct',
            'home_freeThrowsAttempted', 'away_freeThrowsAttempted',
            'home_freeThrowsMade', 'away_freeThrowsMade',
            'home_freeThrowPct', 'away_freeThrowPct',
            'home_threePointFieldGoalsAttempted', 'away_threePointFieldGoalsAttempted',
            'home_threePointFieldGoalsMade', 'away_threePointFieldGoalsMade',
            'home_threePointPct', 'away_threePointPct',
            
            # 리더 통계
            'home_leader_points', 'away_leader_points',
            'home_leader_rebounds', 'away_leader_rebounds',
            'home_leader_assists', 'away_leader_assists',
            
            # 팀 기록
            'home_overall_record_win_rate', 'away_overall_record_win_rate',
            'home_home_record_win_rate', 'away_home_record_win_rate',
            'home_road_record_win_rate', 'away_road_record_win_rate',
            'home_vs_away_win_rate',
            
            # 최근 트렌드
            'home_recent_win_rate', 'away_recent_win_rate',
            'home_recent_avg_score', 'away_recent_avg_score',
            'home_recent_home_win_rate', 'away_recent_home_win_rate',
            'home_recent_away_win_rate', 'away_recent_away_win_rate',
            
            # 컨디션
            'home_rest_days', 'away_rest_days'
        ]
        
        # 기본 특성으로 DataFrame 생성
        X = df[base_features].copy()
        
        # 최근 트렌드 특성 복제
        recent_features = [
            'recent_win_rate',
            'recent_avg_score',
            'recent_home_win_rate',
            'recent_away_win_rate'
        ]
        
        # 복제된 특성 추가
        for col in recent_features:
            for team in ['home', 'away']:
                orig_col = f'{team}_{col}'
                new_col = f'{orig_col}_2'
                X[new_col] = X[orig_col]
        
        self.feature_names = X.columns.tolist()
        
        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Histogram 기반 XGBoost 회귀 모델 학습"""
        n_samples = len(X)
        # 로그 증가 가중치 (완만하게 최신 데이터 강조)
        sample_weights = np.log1p(np.linspace(1, np.e**2, n_samples))
        
        # Histogram-based XGBoost with different strategy
        best_params = {
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'max_leaves': 32,
            'n_estimators': 600,
            'learning_rate': 0.05,
            'max_depth': 0,                       # 무제한 깊이 (max_leaves로 제어)
            'min_child_weight': 8,
            'gamma': 0.5,
            'subsample': 0.8,
            'colsample_bytree': 0.75,
            'colsample_bylevel': 0.8,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'max_bin': 256,
            'objective': 'reg:squarederror',      # 회귀용 목적 함수
            'eval_metric': 'mae',                 # 회귀용 평가 지표
            'base_score': 220,                    # NBA 평균 총점 근처로 시작
            'verbosity': 0,
            'random_state': 42
        }
        
        # 모델 초기화 및 학습
        self.model = XGBRegressor(**best_params)
        
        print("\n=== Model6 Totals (XGBoost Histogram Regressor) 학습 시작 ===")
        self.model.fit(X, y, sample_weight=sample_weights)
        
        # 특성 중요도 계산
        importances = self.model.feature_importances_
        importances = 100.0 * (importances / importances.sum())
        
        metrics = {
            'feature_importance': dict(zip(
                self.feature_names,
                importances
            ))
        }
        
        # 상위 10개 중요 특성 출력
        print("\n=== 상위 10개 중요 특성 (%) ===")
        sorted_features = sorted(
            metrics['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.2f}%")
        
        return metrics
    
    def evaluate_recent_games(self, X: pd.DataFrame, y: pd.Series, n_games: int = 50) -> Dict:
        """학습된 모델로 최근 N경기 예측 성능 평가 (회귀)"""
        # 최근 n_games 선택
        X_recent = X[-n_games:]
        y_recent = y[-n_games:]
        dates_recent = self.dates[-n_games:]
        
        # 예측 수행
        y_pred = self.model.predict(X_recent)
        
        # 회귀 평가 지표 계산
        mae = mean_absolute_error(y_recent, y_pred)
        rmse = np.sqrt(mean_squared_error(y_recent, y_pred))
        r2 = r2_score(y_recent, y_pred)
        
        # 결과 저장
        results = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': list(zip(
                dates_recent.dt.strftime('%Y-%m-%d').tolist(),
                y_recent.tolist(),
                y_pred.tolist()
            ))
        }
        
        # 결과 출력
        print(f"\n=== 최근 {n_games}경기 예측 성능 (회귀) ===")
        print(f"MAE (평균 절대 오차): {mae:.2f}점")
        print(f"RMSE (평균 제곱근 오차): {rmse:.2f}점")
        print(f"R² (결정 계수): {r2:.3f}")
        
        print("\n=== 최근 10경기 예측 상세 ===")
        for date, true, pred in results['predictions'][-10:]:
            diff = pred - true
            print(f"날짜: {date}, 실제: {true:.0f}점, 예측: {pred:.1f}점, 차이: {diff:+.1f}점")
        
        return results
    
    def save_model(self, timestamp: str = None) -> None:
        """학습된 모델과 특성 이름을 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다. 먼저 모델을 학습해주세요.")
        
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 모델 파일 경로
        model_path = self.model_dir / f"totals_model6_{timestamp}.joblib"
        feature_path = self.model_dir / f"totals_features6_{timestamp}.json"
        
        # 모델 저장
        joblib.dump(self.model, model_path)
        
        # 특성 이름 저장
        with open(feature_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'model_info': {
                    'type': 'xgboost_hist_regressor',
                    'target': 'total_score',
                    'params': self.model.get_params()
                }
            }, f, indent=2)
        
        print(f"\n=== 모델 저장 완료 ===")
        print(f"모델 저장 경로: {model_path}")
        print(f"특성 정보 저장 경로: {feature_path}")


def get_latest_processed_data() -> List[Dict]:
    """src/data 폴더에서 가장 최신의 processed json 파일 로드"""
    data_dir = Path(__file__).parent.parent / "data"
    
    json_files = list(data_dir.glob("processed_*.json"))
    json_files = [f for f in json_files if 'prediction' not in f.name and 'spread' not in f.name and 'with_odds' not in f.name]
    
    if not json_files:
        raise FileNotFoundError("처리된 데이터 파일을 찾을 수 없습니다.")
    
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"데이터 파일 로드: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # 최신 데이터 로드
    data = get_latest_processed_data()
    
    # 모델 초기화 및 특성 준비
    model = BettingModel6Totals()
    X, y = model.prepare_features(data)
    
    print("\n=== 데이터 준비 완료 ===")
    print(f"특성 수: {len(model.feature_names)}")
    print(f"샘플 수: {len(X)}")
    print(f"타겟(총점) 평균: {y.mean():.1f}점")
    print(f"타겟(총점) 범위: {y.min():.0f} ~ {y.max():.0f}점")
    
    # 전체 데이터로 모델 학습
    metrics = model.train_model(X, y)
    
    # 최근 70경기 성능 평가
    eval_results = model.evaluate_recent_games(X, y, n_games=70)
    
    # 모델 저장
    model.save_model()

