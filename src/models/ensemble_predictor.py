import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import joblib
from datetime import datetime

class EnsemblePredictor:
    def __init__(self):
        """앙상블 예측기 초기화"""
        self.models = []
        self.feature_names = None
        self.model_dir = Path(__file__).parent / "saved_models"
        self.current_tag = None  # 현재 로드된 모델 태그 (active/shadow)
        
    def load_latest_models(self, model_tag: str = 'active') -> List[Tuple[str, str]]:
        """각 모델의 최신 버전 로드 (1~8번 모델 지원)
        
        Args:
            model_tag: 모델 태그 ('active', 'shadow', 'fixed')
                      - 'active': 현재 운영 중인 모델
                      - 'shadow': 테스트 중인 새 모델
                      - 'fixed': 기존 고정 모델 (하위 호환)
        """
        loaded_models = []
        self.current_tag = model_tag
        
        # 모델 타입 정보
        model_types = {
            1: 'lightgbm',
            2: 'catboost', 
            3: 'xgboost',
            4: 'lightgbm_dart',
            5: 'catboost_ordered',
            6: 'xgboost_hist',
            7: 'random_forest',
            8: 'extra_trees'
        }
        
        print(f"\n=== [{model_tag.upper()}] 모델 로드 시작 ===")
        
        # 각 모델 타입별로 최신 모델 로드 (1~8)
        for model_num in range(1, 9):
            # 모델 파일 찾기 (지정된 태그 우선, 없으면 fallback)
            model_files = list(self.model_dir.glob(f"betting_model{model_num}_{model_tag}*.joblib"))
            
            # fallback: fixed -> 일반 타임스탬프 버전
            if not model_files and model_tag != 'fixed':
                model_files = list(self.model_dir.glob(f"betting_model{model_num}_fixed*.joblib"))
            if not model_files:
                model_files = list(self.model_dir.glob(f"betting_model{model_num}_*.joblib"))
            if not model_files:
                print(f"모델{model_num} 파일을 찾을 수 없습니다. 건너뜁니다.")
                continue
                
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            # 특성 파일 찾기 (지정된 태그 우선)
            feature_file = self.model_dir / f"features{model_num}_{model_tag}.json"
            if not feature_file.exists():
                # fallback: fixed -> 일반 버전
                feature_file = self.model_dir / f"features{model_num}_fixed.json"
            if not feature_file.exists():
                feature_files = list(self.model_dir.glob(f"features{model_num}_*.json"))
                if not feature_files:
                    print(f"모델{model_num} 특성 파일을 찾을 수 없습니다. 건너뜁니다.")
                    continue
                feature_file = max(feature_files, key=lambda x: x.stat().st_mtime)
            
            # 모델 로드
            model = joblib.load(latest_model)
            
            # 특성 정보 로드
            with open(feature_file, 'r') as f:
                feature_info = json.load(f)
            
            loaded_models.append({
                'model': model,
                'features': feature_info['feature_names'],
                'type': f'model{model_num}',
                'algorithm': model_types.get(model_num, 'unknown')
            })
            
            print(f"  모델{model_num} ({model_types.get(model_num, 'unknown')}) 로드 완료")
            print(f"    - 모델: {latest_model.name}")
            print(f"    - 특성: {feature_file.name}")
        
        if not loaded_models:
            raise FileNotFoundError(f"로드할 수 있는 [{model_tag}] 모델이 없습니다.")
        
        self.models = loaded_models
        print(f"\n=== [{model_tag.upper()}] 총 {len(loaded_models)}개 모델 로드 완료 ===")
        return loaded_models
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터에서 특성 추출"""
        # 날짜 기준으로 정렬
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
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
        X = data[base_features].copy()
        
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
        
        return X
    
    def predict_games(self, df: pd.DataFrame, weights: Dict[str, float] = None) -> pd.DataFrame:
        """앙상블 예측 수행 (1~8번 모델 지원)"""
        X = self.prepare_features(df)
        
        if weights is None:
            # 기본 가중치 설정 (로드된 모델에 대해 동일 가중치)
            weights = {model_info['type']: 1/len(self.models) for model_info in self.models}
        
        # 각 모델의 예측 확률 계산 및 저장
        model_predictions = {}
        weighted_predictions = []
        
        # 가중치 합계 계산 (정규화용)
        total_weight = sum(weights.get(model_info['type'], 0) for model_info in self.models)
        if total_weight == 0:
            total_weight = 1  # 0으로 나누기 방지
        
        for model_info in self.models:
            model = model_info['model']
            model_type = model_info['type']
            prob = model.predict_proba(X)[:, 1]
            
            # 개별 모델 확률 저장 (가중치 적용 전)
            model_predictions[model_type] = prob
            
            # 가중치 적용된 확률 (정규화)
            weight = weights.get(model_type, 0)
            if total_weight > 0:
                weighted_predictions.append(prob * (weight / total_weight))
        
        # 가중 평균 계산
        ensemble_probabilities = np.sum(weighted_predictions, axis=0)
        
        # 결과 DataFrame 생성
        results_df = df[['date', 'home_team_name', 'away_team_name']].copy()
        results_df['home_win_probability'] = ensemble_probabilities
        
        # 각 모델의 개별 확률 추가 (1~8번 모델)
        for i in range(1, 9):
            col_name = f'model{i}_home_win_prob'
            results_df[col_name] = model_predictions.get(f'model{i}', np.zeros(len(df)))
        
        results_df['predicted_winner'] = np.where(
            ensemble_probabilities > 0.5,
            results_df['home_team_name'],
            results_df['away_team_name']
        )
        results_df['win_probability'] = np.where(
            ensemble_probabilities > 0.5,
            ensemble_probabilities,
            1 - ensemble_probabilities
        )
        
        # 날짜 형식 변환
        results_df['date'] = pd.to_datetime(results_df['date']).dt.strftime('%Y-%m-%d')
        
        # 모델 이름 매핑
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
        
        # 결과 출력
        print("\n=== 앙상블 예측 결과 ===")
        for _, row in results_df.iterrows():
            print(f"\n{row['date']} 경기:")
            print(f"{row['home_team_name']} vs {row['away_team_name']}")
            print(f"예상 승자: {row['predicted_winner']}")
            print(f"승리 확률: {row['win_probability']:.1%}")
            
            # 로드된 모델들의 확률만 출력
            for model_info in self.models:
                model_type = model_info['type']
                model_name = model_names.get(model_type, model_type)
                col_name = f'{model_type}_home_win_prob'
                if col_name in row:
                    print(f"  - {model_name}: {row[col_name]:.1%}")
        
        return results_df
    
    def save_predictions(self, predictions: pd.DataFrame, model_tag: Optional[str] = None) -> Path:
        """예측 결과 저장
        
        Args:
            predictions: 예측 결과 DataFrame
            model_tag: 저장할 태그 (None이면 현재 로드된 태그 사용)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "predictions"
        output_dir.mkdir(exist_ok=True)
        
        # 태그 결정 (파라미터 > 현재 로드된 태그 > 없음)
        tag = model_tag or self.current_tag
        
        if tag:
            output_path = output_dir / f"ensemble_predictions_{timestamp}_{tag}.json"
        else:
            output_path = output_dir / f"ensemble_predictions_{timestamp}.json"
        
        predictions.to_json(output_path, orient='records', indent=2)
        
        tag_display = f"[{tag.upper()}] " if tag else ""
        print(f"\n=== {tag_display}앙상블 예측 결과 저장 완료 ===")
        print(f"저장 경로: {output_path}")
        
        return output_path
    
    def load_prediction_data(self) -> pd.DataFrame:
        """최신 예측용 데이터 로드"""
        data_dir = Path(__file__).parent.parent / "data"
        
        # model_input_features로 시작하는 가장 최신 파일 찾기
        pred_files = list(data_dir.glob("model_input_features_*.json"))
        if not pred_files:
            raise FileNotFoundError("예측할 데이터 파일을 찾을 수 없습니다.")
        
        latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
        print(f"\n예측 데이터 파일 로드: {latest_file.name}")
        
        # JSON 파일 로드 및 DataFrame 변환
        with open(latest_file, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data) 