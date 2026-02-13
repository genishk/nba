# src/models/model7_with_odds.py
"""
배당 변수를 포함한 RandomForest 모델
- 기존 model7.py 복사본
- home_odds_bucket, away_odds_bucket 변수 추가
- 모델 파라미터는 동일하게 유지
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


class BettingModel7WithOdds:
    """배당 변수를 포함한 RandomForest 베팅 모델"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_dir = Path(__file__).parent / "saved_models"
        self.model_dir.mkdir(exist_ok=True)
        self.dates = None
    
    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터에서 특성과 레이블 추출 (배당 변수 포함!)"""
        df = pd.DataFrame(data)
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        self.dates = df['date']
        
        y = (df['home_team_score'] > df['away_team_score']).astype(int)
        
        base_features = [
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
            'home_leader_points', 'away_leader_points',
            'home_leader_rebounds', 'away_leader_rebounds',
            'home_leader_assists', 'away_leader_assists',
            'home_overall_record_win_rate', 'away_overall_record_win_rate',
            'home_home_record_win_rate', 'away_home_record_win_rate',
            'home_road_record_win_rate', 'away_road_record_win_rate',
            'home_vs_away_win_rate',
            'home_recent_win_rate', 'away_recent_win_rate',
            'home_recent_avg_score', 'away_recent_avg_score',
            'home_recent_home_win_rate', 'away_recent_home_win_rate',
            'home_recent_away_win_rate', 'away_recent_away_win_rate',
            'home_rest_days', 'away_rest_days'
        ]
        
        X = df[base_features].copy()
        
        recent_features = ['recent_win_rate', 'recent_avg_score', 'recent_home_win_rate', 'recent_away_win_rate']
        for col in recent_features:
            for team in ['home', 'away']:
                orig_col = f'{team}_{col}'
                new_col = f'{orig_col}_2'
                X[new_col] = X[orig_col]
        
        # ★★★ 배당 변수 추가 ★★★
        if 'home_odds_bucket' in df.columns and 'away_odds_bucket' in df.columns:
            X['home_odds_bucket'] = df['home_odds_bucket']
            X['away_odds_bucket'] = df['away_odds_bucket']
            print(f"\n✅ 배당 변수 추가됨: home_odds_bucket, away_odds_bucket")
        else:
            print(f"\n⚠️ 배당 변수가 데이터에 없습니다!")
        
        self.feature_names = X.columns.tolist()
        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """RandomForest 모델 학습"""
        n_samples = len(X)
        sample_weights = np.exp(np.linspace(0, 0.8, n_samples))
        
        # 기존과 동일한 파라미터!
        best_params = {
            'n_estimators': 500,
            'max_depth': 8,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'max_leaf_nodes': 50,
            'bootstrap': True,
            'oob_score': True,
            'class_weight': 'balanced',
            'criterion': 'gini',
            'n_jobs': -1,
            'random_state': 42,
            'verbose': 0
        }
        
        self.model = RandomForestClassifier(**best_params)
        
        print("\n=== Model7 (RandomForest) 배당 포함 학습 시작 ===")
        print(f"총 특성 수: {len(self.feature_names)}")
        self.model.fit(X, y, sample_weight=sample_weights)
        
        print(f"OOB Score: {self.model.oob_score_:.3f}")
        
        importances = self.model.feature_importances_
        importances = 100.0 * (importances / importances.sum())
        
        metrics = {'feature_importance': dict(zip(self.feature_names, importances)), 'oob_score': self.model.oob_score_}
        
        print("\n=== 상위 15개 중요 특성 (%) ===")
        sorted_features = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:15]
        for feature, importance in sorted_features:
            marker = "⭐" if 'odds' in feature else ""
            print(f"{marker}{feature}: {importance:.2f}%")
        
        print("\n=== 배당 변수 중요도 ===")
        for feature in ['home_odds_bucket', 'away_odds_bucket']:
            if feature in metrics['feature_importance']:
                print(f"  {feature}: {metrics['feature_importance'][feature]:.2f}%")
        
        return metrics
    
    def evaluate_recent_games(self, X: pd.DataFrame, y: pd.Series, n_games: int = 50) -> Dict:
        X_recent = X[-n_games:]
        y_recent = y[-n_games:]
        
        y_pred = self.model.predict(X_recent)
        y_pred_proba = self.model.predict_proba(X_recent)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_recent, y_pred),
            'roc_auc': roc_auc_score(y_recent, y_pred_proba),
        }
        
        print(f"\n=== 최근 {n_games}경기 예측 성능 ===")
        print(f"정확도: {results['accuracy']:.3f}, ROC-AUC: {results['roc_auc']:.3f}")
        
        return results
    
    def save_model(self, timestamp: str = None) -> None:
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self.model_dir / f"betting_model7_with_odds_{timestamp}.joblib"
        feature_path = self.model_dir / f"features7_with_odds_{timestamp}.json"
        
        joblib.dump(self.model, model_path)
        
        with open(feature_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'model_info': {'type': 'random_forest_with_odds', 'odds_buckets': 8}
            }, f, indent=2)
        
        print(f"\n=== 모델 저장 완료: {model_path} ===")


def get_latest_processed_data_with_odds() -> List[Dict]:
    data_dir = Path(__file__).parent.parent / "data"
    json_files = list(data_dir.glob("processed_with_odds_*.json"))
    
    if not json_files:
        raise FileNotFoundError("배당 포함 데이터 파일을 찾을 수 없습니다.")
    
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"데이터 파일 로드: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    print("=" * 70)
    print("🏀 배당 변수 포함 RandomForest 모델 학습 (Model7)")
    print("=" * 70)
    
    data = get_latest_processed_data_with_odds()
    model = BettingModel7WithOdds()
    X, y = model.prepare_features(data)
    
    metrics = model.train_model(X, y)
    eval_results = model.evaluate_recent_games(X, y, n_games=70)
    model.save_model()
    
    print("\n✅ 완료!")

