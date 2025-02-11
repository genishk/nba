## SpecStory

## NBA Betting Model Performance and Validation (1/24/2025, 2:24:07 PM)

_**User**_

코드 분석 및 응답 절차:
1. 제공된 코드 전체를 처음부터 끝까지 완전히 읽고 이해합니다.
2. 관련된 모든 메서드와 클래스를 확인합니다.
3. 코드의 각 부분이 어떻게 연결되어 있는지 파악합니다.
4. 실제 데이터와 함께 코드의 흐름을 검증합니다.
5. 발견한 내용을 바탕으로 응답을 작성하기 전에 다시 한번 전체를 검토합니다.
6. 확실하지 않은 부분이 있다면, 추가 질문을 통해 명확히 합니다.
7. 절대 코드의 일부분만 보고 성급한 판단을 하지 않으며, 항상 전체적인 맥락에서 분석합니다.
8. 네가 나의 질문에 초개인화된 응답을 할 수 있도록, 의도와 목적을 명확히 이해할 때까지 질문한 뒤 응답해 줘. 또한 기존 파일에 대한 질문을 받거나, 수정을 해야될 땐 절대 형식이나 내용을 임의로 가정하지 말고 실제 데이터를 참고해.
9. 전체적으로 논리적으로 내 요구에 맞는지 10번 이상 확인해서 논리적 허점이 없고, 정확히 계산되게 수정한 코드를 제시해.
위 아홉가지 프롬프트를 반드시 명심하고 지켜.

질문 및 요구사항:

현재 3가지 모델이 나오고 있고, 지금 일단 model3를 보면 

# src/models/model3.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
import xgboost as xgb

class BettingModel3:
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
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def optimize_params(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """베이지안 최적화로 XGBoost 최적 파라미터 탐색"""
        def objective(
            learning_rate,
            max_depth,
            min_child_weight,
            gamma,
            subsample,
            colsample_bytree,
            reg_alpha,
            reg_lambda
        ):
            params = {
                'n_estimators': 1000,  # 고정 파라미터
                'learning_rate': learning_rate,
                'max_depth': int(max_depth),
                'min_child_weight': min_child_weight,
                'gamma': gamma,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'verbosity': 0,
                'random_state': 42
            }
            
            # TimeSeriesSplit으로 시계열 교차 검증
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, pred)
                scores.append(score)
            
            return np.mean(scores)

        # 파라미터 범위 설정
        param_bounds = {
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
            'min_child_weight': (1, 10),
            'gamma': (0, 5),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'reg_alpha': (0, 10),
            'reg_lambda': (0, 10)
        }

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=param_bounds,
            random_state=42
        )

        print("\n=== 베이지안 최적화 시작 ===")
        optimizer.maximize(
            init_points=5,
            n_iter=25
        )

        # 최적 파라미터 추출
        best_params = {
            'n_estimators': 1000,
            'learning_rate': optimizer.max['params']['learning_rate'],
            'max_depth': int(optimizer.max['params']['max_depth']),
            'min_child_weight': optimizer.max['params']['min_child_weight'],
            'gamma': optimizer.max['params']['gamma'],
            'subsample': optimizer.max['params']['subsample'],
            'colsample_bytree': optimizer.max['params']['colsample_bytree'],
            'reg_alpha': optimizer.max['params']['reg_alpha'],
            'reg_lambda': optimizer.max['params']['reg_lambda'],
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'verbosity': 0,
            'random_state': 42
        }

        print("\n=== 최적 파라미터 ===")
        for param, value in best_params.items():
            print(f"{param}: {value}")

        return best_params

    def train_model(self, X: pd.DataFrame, y: pd.Series, params: Dict = None) -> Dict:
        """최적 파라미터로 XGBoost 모델 학습"""
        
        if params is None:
            # 기본 파라미터 설정
            params = {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_child_weight': 3,
                'gamma': 0.5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'verbosity': 0,
                'random_state': 42
            }
        
        # 모델 초기화 및 학습
        self.model = XGBClassifier(**params)
        
        print("\n=== 모델 학습 시작 ===")
        self.model.fit(X, y)
        
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
        """학습된 모델로 최근 N경기 예측 성능 평가"""
        from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
        
        # 최근 n_games 선택
        X_recent = X[-n_games:]
        y_recent = y[-n_games:]
        dates_recent = self.dates[-n_games:]
        
        # 예측 수행
        y_pred = self.model.predict(X_recent)
        y_pred_proba = self.model.predict_proba(X_recent)[:, 1]
        
        # 혼동 행렬 계산
        conf_matrix = confusion_matrix(y_recent, y_pred)
        
        # 결과 저장
        results = {
            'accuracy': accuracy_score(y_recent, y_pred),
            'roc_auc': roc_auc_score(y_recent, y_pred_proba),
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': list(zip(
                dates_recent.dt.strftime('%Y-%m-%d').tolist(),
                y_recent.tolist(),
                y_pred.tolist(),
                y_pred_proba.tolist()
            ))
        }
        
        # 결과 출력
        print(f"\n=== 최근 {n_games}경기 예측 성능 ===")
        print(f"정확도: {results['accuracy']:.3f}")
        print(f"ROC-AUC: {results['roc_auc']:.3f}")
        print("\n혼동 행렬:")
        print(f"TN: {conf_matrix[0,0]}, FP: {conf_matrix[0,1]}")
        print(f"FN: {conf_matrix[1,0]}, TP: {conf_matrix[1,1]}")
        
        print("\n=== 최근 10경기 예측 상세 ===")
        for date, true, pred, prob in results['predictions'][-10:]:
            print(f"날짜: {date}, 실제: {true}, 예측: {pred}, 승리확률: {prob:.3f}")
        
        return results
    
    def save_model(self, timestamp: str = None) -> None:
        """학습된 모델과 특성 이름을 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다. 먼저 모델을 학습해주세요.")
        
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 모델 파일 경로
        model_path = self.model_dir / f"betting_model_{timestamp}.joblib"
        feature_path = self.model_dir / f"features_{timestamp}.json"
        
        # 모델 저장
        joblib.dump(self.model, model_path)
        
        # 특성 이름 저장
        with open(feature_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'model_info': {
                    'type': 'xgboost',
                    'params': self.model.get_params()
                }
            }, f, indent=2)
        
        print(f"\n=== 모델 저장 완료 ===")
        print(f"모델 저장 경로: {model_path}")
        print(f"특성 정보 저장 경로: {feature_path}")


def get_latest_processed_data() -> List[Dict]:
    """src/data 폴더에서 가장 최신의 processed json 파일 로드 (prediction 제외)"""
    data_dir = Path(__file__).parent.parent / "data"
    
    # prediction이 포함되지 않은 processed_ 파일만 찾기
    json_files = list(data_dir.glob("processed_*.json"))
    json_files = [f for f in json_files if 'prediction' not in f.name]
    
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
    model = BettingModel3()
    X, y = model.prepare_features(data)
    
    print("\n=== 데이터 준비 완료 ===")
    print(f"특성 수: {len(model.feature_names)}")
    print(f"샘플 수: {len(X)}")
    
    # # 베이지안 최적화로 최적 파라미터 찾기
    # best_params = model.optimize_params(X, y)
    
    # 최적 파라미터로 모델 학습
    metrics = model.train_model(X, y)
    
    # 최근 50경기 성능 평가
    eval_results = model.evaluate_recent_games(X, y, n_games=50)
    
    # 모델 저장
    model.save_model() 

이런 코드를 가지고 있는데 성능은

데이터 파일 로드: processed_20250123_151221.json

=== 데이터 준비 완료 ===
특성 수: 45
샘플 수: 665

=== 모델 학습 시작 ===

=== 상위 10개 중요 특성 (%) ===
away_recent_win_rate: 5.36%
home_recent_home_win_rate: 3.51%
away_recent_avg_score: 3.43%
away_fieldGoalsMade: 3.07%
home_overall_record_win_rate: 2.85%
away_overall_record_win_rate: 2.70%
home_fieldGoalsMade: 2.53%
away_freeThrowsMade: 2.52%
away_recent_away_win_rate: 2.42%
away_recent_home_win_rate: 2.41%

=== 최근 50경기 예측 성능 ===
정확도: 1.000
ROC-AUC: 1.000

혼동 행렬:
TN: 21, FP: 0
FN: 0, TP: 29

=== 최근 10경기 예측 상세 ===
날짜: 2025-01-22, 실제: 1, 예측: 1, 승리확률: 0.933
날짜: 2025-01-22, 실제: 1, 예측: 1, 승리확률: 0.979
날짜: 2025-01-23, 실제: 0, 예측: 0, 승리확률: 0.124
날짜: 2025-01-23, 실제: 0, 예측: 0, 승리확률: 0.048
날짜: 2025-01-23, 실제: 0, 예측: 0, 승리확률: 0.083
날짜: 2025-01-23, 실제: 1, 예측: 1, 승리확률: 0.959
날짜: 2025-01-23, 실제: 1, 예측: 1, 승리확률: 0.917
날짜: 2025-01-23, 실제: 1, 예측: 1, 승리확률: 0.882
날짜: 2025-01-23, 실제: 1, 예측: 1, 승리확률: 0.937
날짜: 2025-01-23, 실제: 0, 예측: 0, 승리확률: 0.110

=== 모델 저장 완료 ===
모델 저장 경로: c:\Users\SeokHyun\Desktop\Coding\Auto-Bet\src\models\saved_models\betting_model_20250123_190909.joblib
특성 정보 저장 경로: c:\Users\SeokHyun\Desktop\Coding\Auto-Bet\src\models\saved_models\features_20250123_190909.json


이렇게 좋게 나오긴 하는데 한가지 문제가 있어.

현재 검증이랑 학습 데이터가 코드를 보면 겹치거든? 근데 이렇게 할 수 밖에 없는 이유는
nba는 당연히 최근 데이터, 기세가 가장 중요하기 때문에 가장 최근 데이터를 학습에 포함시켜야 하기 때문이야. 그리고 가장 최근, 미래의 경기를 예측하는거기 때문에 검증도 최근 경기로 해야되는거고.

이거 그냥 성능 좋으니 이대로 해도 될까? 아니면 @Web 이거에 대한 괜찮은 개선법이 있을까?

---

_**Assistant**_

현재 상황을 분석해보고 개선 방안을 제안하겠습니다.

### 현재 문제점
1. 학습 데이터와 검증 데이터가 겹치면서 과적합(overfitting)이 발생할 수 있습니다.
2. 현재 성능 지표(accuracy=1.0, ROC-AUC=1.0)는 실제 성능보다 과대 평가되었을 가능성이 높습니다.
3. 하지만 NBA에서는 최근 데이터와 기세가 중요하기 때문에, 최근 데이터를 완전히 배제할 수도 없습니다.

### 개선 방안

다음과 같이 `evaluate_recent_games` 메서드를 수정하여 "슬라이딩 윈도우" 방식의 평가를 구현하는 것을 제안합니다:

```python:src/models/model3.py
def evaluate_recent_games(self, X: pd.DataFrame, y: pd.Series, n_games: int = 50, window_size: int = 100) -> Dict:
    """학습된 모델로 최근 N경기 예측 성능 평가 (슬라이딩 윈도우 방식)"""
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    
    total_samples = len(X)
    predictions = []
    actual_values = []
    probabilities = []
    dates_list = []
    
    # 최근 n_games에 대해 각각 예측 수행
    for i in range(total_samples - n_games, total_samples):
        # 현재 시점까지의 데이터로 모델 재학습
        # window_size만큼의 최근 데이터만 사용
        train_start = max(0, i - window_size)
        X_train = X.iloc[train_start:i]
        y_train = y.iloc[train_start:i]
        
        # 임시 모델 생성 및 학습
        temp_model = XGBClassifier(**self.model.get_params())
        temp_model.fit(X_train, y_train)
        
        # 다음 경기 예측
        X_pred = X.iloc[i:i+1]
        y_true = y.iloc[i]
        
        pred = temp_model.predict(X_pred)[0]
        pred_proba = temp_model.predict_proba(X_pred)[0, 1]
        
        predictions.append(pred)
        actual_values.append(y_true)
        probabilities.append(pred_proba)
        dates_list.append(self.dates.iloc[i])
    
    # 성능 평가
    conf_matrix = confusion_matrix(actual_values, predictions)
    
    results = {
        'accuracy': accuracy_score(actual_values, predictions),
        'roc_auc': roc_auc_score(actual_values, probabilities),
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': list(zip(
            [d.strftime('%Y-%m-%d') for d in dates_list],
            actual_values,
            predictions,
            probabilities
        ))
    }
    
    # 결과 출력
    print(f"\n=== 최근 {n_games}경기 예측 성능 (슬라이딩 윈도우) ===")
    print(f"학습 윈도우 크기: {window_size}경기")
    print(f"정확도: {results['accuracy']:.3f}")
    print(f"ROC-AUC: {results['roc_auc']:.3f}")
    print("\n혼동 행렬:")
    print(f"TN: {conf_matrix[0,0]}, FP: {conf_matrix[0,1]}")
    print(f"FN: {conf_matrix[1,0]}, TP: {conf_matrix[1,1]}")
    
    print("\n=== 최근 10경기 예측 상세 ===")
    for date, true, pred, prob in results['predictions'][-10:]:
        print(f"날짜: {date}, 실제: {true}, 예측: {pred}, 승리확률: {prob:.3f}")
    
    return results
```

### 개선된 방식의 특징

1. **슬라이딩 윈도우 방식**: 
   - 각 예측 시점마다 이전 `window_size`개의 경기만을 사용하여 모델을 재학습합니다.
   - 이는 실제 베팅 상황을 더 잘 시뮬레이션합니다.

2. **데이터 누수 방지**:
   - 각 예측은 해당 시점 이전의 데이터만을 사용합니다.
   - 미래 정보가 학습에 포함되지 않습니다.

3. **최신성 유지**:
   - `window_size` 파라미터로 얼마나 최근의 데이터를 사용할지 조절할 수 있습니다.
   - NBA의 특성상 최근 기세를 반영하면서도 과적합을 방지할 수 있습니다.

### 사용 방법

```python
# 메인 코드에서 평가 시
eval_results = model.evaluate_recent_games(
    X, y, 
    n_games=50,     # 평가할 최근 경기 수
    window_size=100 # 학습에 사용할 이전 경기 수
)
```

이 방식을 사용하면:
1. 더 현실적인 성능 평가가 가능합니다.
2. 과적합 위험을 줄일 수 있습니다.
3. 실제 베팅 상황과 유사한 환경에서 모델을 테스트할 수 있습니다.

`window_size`는 하이퍼파라미터로 취급하여, 교차 검증을 통해 최적값을 찾을 수 있습니다. NBA 시즌의 특성을 고려하면 100~200 게임 정도가 적절할 것으로 예상됩니다.

---

_**User**_

지금 저걸 돌리면 결과적으로 model에는 어떤 게 저장되는거야? 가장 최근 경기까지 포함된 전체 데이터로 학습하는거야? 아니면 일부만 사용해?

---

_**Assistant**_

현재 코드를 분석해보면:

1. `train_model` 메서드에서는 전체 데이터(`X`, `y`)를 사용하여 `self.model`을 학습시킵니다.

2. `evaluate_recent_games` 메서드는 슬라이딩 윈도우 방식으로 성능을 평가하기 위해 임시 모델들(`temp_model`)을 만들어 사용하지만, 이는 평가용으로만 사용되고 `self.model`에는 영향을 주지 않습니다.

따라서 최종적으로 `self.model`에는 전체 데이터로 학습된 모델이 저장됩니다. 이는 다음과 같은 장단점이 있습니다:

### 장점
- 가능한 모든 데이터를 활용하여 학습하므로, 실제 예측 시 더 많은 정보를 활용할 수 있습니다.
- 최근 데이터의 패턴도 모두 학습에 포함됩니다.

### 단점
- 평가 방식(`evaluate_recent_games`)과 실제 저장되는 모델의 학습 방식이 다릅니다.
- 실제 운영 시에는 전체 데이터로 학습된 모델을 사용하게 되므로, 평가 결과가 실제 성능을 정확히 반영하지 않을 수 있습니다.

만약 평가 방식과 동일한 방식으로 최종 모델을 저장하고 싶다면, `train_model` 메서드를 다음과 같이 수정할 수 있습니다:

```python:src/models/model3.py
def train_model(self, X: pd.DataFrame, y: pd.Series, window_size: int = 100, params: Dict = None) -> Dict:
    """최신 window_size 만큼의 데이터로 XGBoost 모델 학습"""
    
    if params is None:
        # 기본 파라미터 설정
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 3,
            'gamma': 0.5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1,
            'reg_lambda': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'verbosity': 0,
            'random_state': 42
        }
    
    # 최신 window_size 만큼의 데이터만 사용
    X_train = X.iloc[-window_size:]
    y_train = y.iloc[-window_size:]
    
    # 모델 초기화 및 학습
    self.model = XGBClassifier(**params)
    
    print("\n=== 모델 학습 시작 ===")
    print(f"학습 데이터 크기: {len(X_train)} (최근 {window_size}경기)")
    self.model.fit(X_train, y_train)
    
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
```

이렇게 수정하면:
1. 최종 모델도 최근 `window_size` 경기만을 사용하여 학습됩니다.
2. 평가 방식과 실제 모델 

---



---

