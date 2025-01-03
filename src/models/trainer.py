# src/models/trainer.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class BettingModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_dir = Path(__file__).parent / "saved_models"
        self.model_dir.mkdir(exist_ok=True)
        self.dates = None  # 날짜 정보 저장을 위한 변수 추가
    
    def prepare_features(self, data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터에서 특성과 레이블 추출"""
        df = pd.DataFrame(data)
        
        # 날짜 기준으로 정렬
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        self.dates = df['date']  # 날짜 정보 저장
        
        # 승패 레이블 생성 (홈팀 기준)
        y = (df['home_team_score'] > df['away_team_score']).astype(int)
        
        # 특성 선택
        features = [
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
        
        X = df[features]
        self.feature_names = features
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series, initial_train: bool = True) -> Dict:
        """앙상블 모델 학습 및 평가"""
        # 시간 순서대로 정렬된 데이터에서 최근 20%를 평가 세트로 사용
        eval_size = int(len(X) * 0.2)
        X_eval = X[-eval_size:]
        y_eval = y[-eval_size:]
        
        if initial_train:  # 초기 학습인 경우에만 새 모델 생성
            # 시간 가중치 강소 (0.7 -> 0.3)
            n_samples = len(X)
            time_weights = np.linspace(0.3, 1.0, n_samples)
            
            # 가중치 분포 시각화
            plt.figure(figsize=(10, 5))
            plt.plot(self.dates, time_weights)
            plt.title('Time Weights Distribution')
            plt.xlabel('Date')
            plt.ylabel('Weight')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.model_dir / 'initial_time_weights.png')
            plt.close()
            
            # 개별 모델 정의 (규제 강화)
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=5,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.2,
                reg_lambda=0.2,
                random_state=42
            )
            
            lgb_model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=3,
                min_child_samples=30,
                min_split_gain=0.1,
                reg_alpha=0.2,
                reg_lambda=0.2,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42,
                verbose=-1
            )
            
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42
            )
            
            # 앙상블 모델 생성
            self.model = VotingClassifier(
                estimators=[
                    ('xgb', xgb_model),
                    ('lgb', lgb_model),
                    ('rf', rf_model)
                ],
                voting='soft',
                weights=[1, 1, 1]
            )
            
            # 가중치를 적용하여 모델 학습
            self.model.fit(X, y, sample_weight=time_weights)
        
        # 최근 20% 데이터로 성능 평가
        y_pred = self.model.predict(X_eval)
        y_pred_proba = self.model.predict_proba(X_eval)
        
        # 날짜별 예측 정확도 분석
        date_accuracy = pd.DataFrame({
            'date': self.dates[-eval_size:],
            'actual': y_eval,
            'predicted': y_pred
        })
        date_accuracy['correct'] = (date_accuracy['actual'] == date_accuracy['predicted']).astype(int)
        
        # 날짜별 정확도 시각화
        plt.figure(figsize=(12, 6))
        date_accuracy.set_index('date')['correct'].rolling('7D', min_periods=1).mean().plot()
        plt.title('7-Day Rolling Average Prediction Accuracy')
        plt.xlabel('Date')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.tight_layout()
        
        # 파일명 구분
        accuracy_plot_name = 'initial_accuracy_trend.png' if initial_train else 'finetuned_accuracy_trend.png'
        plt.savefig(self.model_dir / accuracy_plot_name)
        plt.close()
        
        metrics = {
            'accuracy': accuracy_score(y_eval, y_pred),
            'roc_auc': roc_auc_score(y_eval, y_pred_proba[:, 1]),
            'classification_report': classification_report(y_eval, y_pred),
            'eval_size': eval_size,
            'total_size': len(X),
            'time_analysis': {
                'early_accuracy': date_accuracy['correct'][:eval_size//2].mean(),
                'late_accuracy': date_accuracy['correct'][eval_size//2:].mean()
            }
        }
        
        # 교차 검증은 전체 데이터에 대해 수행
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        metrics['cv_scores'] = cv_scores
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # 특성 중요도 시각화도 구분
        importance_plot_name = 'initial_feature_importance.png' if initial_train else 'finetuned_feature_importance.png'
        self.analyze_feature_importance(X, importance_plot_name)
        
        return metrics
    
    def analyze_feature_importance(self, X: pd.DataFrame, plot_name: str) -> Dict:
        """특성 중요도 분석"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 각 모델의 특성 중요도를 정규화하여 계산
        xgb_model = self.model.named_estimators_['xgb']
        lgb_model = self.model.named_estimators_['lgb']
        rf_model = self.model.named_estimators_['rf']
        
        xgb_importance = xgb_model.feature_importances_
        lgb_importance = lgb_model.feature_importances_
        rf_importance = rf_model.feature_importances_
        
        # 각각의 특성 중요도를 0-1 사이로 정규화
        xgb_importance = xgb_importance / np.sum(xgb_importance)
        lgb_importance = lgb_importance / np.sum(lgb_importance)
        rf_importance = rf_importance / np.sum(rf_importance)
        
        # 정규화된 값들의 평균 계산
        importance_values = (xgb_importance + lgb_importance + rf_importance) / 3
        importance_dict = dict(zip(self.feature_names, importance_values))
        
        # 특성 중요도 시각화 및 저장
        plt.figure(figsize=(12, 6))
        sorted_idx = np.argsort(importance_values)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, importance_values[sorted_idx])
        plt.yticks(pos, np.array(self.feature_names)[sorted_idx])
        plt.xlabel('Feature Importance (Average)')
        plt.title('Ensemble Model Feature Importance')
        plt.tight_layout()
        plt.savefig(self.model_dir / plot_name)
        plt.close()
        
        return importance_dict
    
    def save_model(self, timestamp: str):
        """모델 저장"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        model_path = self.model_dir / f"betting_model_{timestamp}.joblib"
        joblib.dump(self.model, model_path)
        print(f"모델이 저장되었습니다: {model_path}")
    
    def fine_tune(self, X: pd.DataFrame, y: pd.Series, n_recent_games: int = 100) -> None:
        """최근 N경기 데이터로 모델 파인튜닝"""
        if self.model is None:
            raise ValueError("기존 학습된 모델이 없습니다.")
        
        # 최근 N경기 선택
        X_recent = X[-n_recent_games:]
        y_recent = y[-n_recent_games:]
        
        # 각 모델 가져오기
        xgb_model = self.model.named_estimators_['xgb']
        lgb_model = self.model.named_estimators_['lgb']
        rf_model = self.model.named_estimators_['rf']
        
        # 파인튜닝을 위한 파라미터 조정 (learning rate 감소)
        xgb_model.set_params(learning_rate=0.01)
        lgb_model.set_params(learning_rate=0.01)
        
        # 시간 가중치 감소 (0.8 -> 0.5)
        time_weights = np.linspace(0.5, 1.0, len(X_recent))
        
        # 파인튜닝 가중치 분포 시각화
        plt.figure(figsize=(10, 5))
        plt.plot(self.dates[-n_recent_games:], time_weights)
        plt.title('Fine-tuning Time Weights Distribution')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.model_dir / 'finetuning_time_weights.png')
        plt.close()
        
        # 각 모델 파인튜닝
        xgb_model.fit(X_recent, y_recent, sample_weight=time_weights)
        lgb_model.fit(X_recent, y_recent, sample_weight=time_weights)
        rf_model.fit(X_recent, y_recent, sample_weight=time_weights)
        
        # VotingClassifier 재구성
        self.model.estimators_ = [xgb_model, lgb_model, rf_model]

def get_latest_processed_data() -> List[Dict]:
    """src/data 폴더에서 가장 최신의 processed json 파일 로드"""
    data_dir = Path(__file__).parent.parent / "data"
    json_files = list(data_dir.glob("processed_*.json"))
    
    if not json_files:
        raise FileNotFoundError("처리된 데이터 파일을 찾을 수 없습니다.")
    
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"데이터 파일 로드: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

# 테스트 코드
if __name__ == "__main__":
    # 최신 데이터 로드
    data = get_latest_processed_data()
    
    # 모델 학습
    trainer = BettingModelTrainer()
    X, y = trainer.prepare_features(data)
    
    # 초기 모델 학습
    print("\n=== 초기 모델 학습 ===")
    metrics = trainer.train(X, y, initial_train=True)
    print("\n=== 초기 모델 성능 ===")
    print(f"정확도: {metrics['accuracy']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    
    # 최근 100경기로 파인튜닝
    print("\n=== 파인튜닝 시작 ===")
    trainer.fine_tune(X, y, n_recent_games=100)
    
    # 파인튜닝된 모델 평가
    print("\n=== 파인튜닝된 모델 평가 ===")
    metrics = trainer.train(X, y, initial_train=False)  # 기존 모델 유지
    
    # 결과 출력 (파인튜닝된 모델의 성능)
    print("\n=== 모델 성능 ===")
    print(f"정확도: {metrics['accuracy']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"\n=== 평가 데이터 크기 ===")
    print(f"전체 데이터: {metrics['total_size']}")
    print(f"평가 데이터: {metrics['eval_size']}")
    print("\n=== 교차 검증 결과 ===")
    print(f"CV 평균 정확도: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']*2:.3f})")
    print("\n=== 분류 리포트 ===")
    print(metrics['classification_report'])
    print("\n=== 시간 기반 분석 ===")
    print(f"초기 평가 데이터 정확도: {metrics['time_analysis']['early_accuracy']:.3f}")
    print(f"후기 평가 데이터 정확도: {metrics['time_analysis']['late_accuracy']:.3f}")
    
    # 특성 중요도 분석
    importance_dict = trainer.analyze_feature_importance(X, 'initial_feature_importance.png')
    print("\n=== 상위 5개 중요 특성 ===")
    for feature, importance in sorted(
        importance_dict.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]:
        print(f"{feature}: {importance:.3f}")
    
    # 파인튜닝된 모델 저장
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save_model(timestamp)