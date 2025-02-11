import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class BettingPredictor:
    def __init__(self):
        """예측 모델 초기화"""
        self.model = None
        self.feature_names = None
        self.model_dir = Path(__file__).parent / "saved_models"
        
    def load_latest_model(self) -> Tuple[str, str]:
        """가장 최근에 저장된 모델과 특성 파일 로드"""
        # 모델 파일 찾기
        model_files = list(self.model_dir.glob("betting_model3_*.joblib"))
        if not model_files:
            raise FileNotFoundError("저장된 모델을 찾을 수 없습니다.")
        
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        # betting_model3_20240319_123456.joblib에서 20240319_123456 부분 추출
        timestamp = '_'.join(latest_model.stem.split('_')[2:])
        
        # 해당 타임스탬프의 특성 파일 찾기
        feature_file = self.model_dir / f"features3_{timestamp}.json"
        if not feature_file.exists():
            raise FileNotFoundError(f"특성 파일을 찾을 수 없습니다: {feature_file}")
        
        # 모델과 특성 정보 로드
        self.model = joblib.load(latest_model)
        with open(feature_file, 'r') as f:
            feature_info = json.load(f)
            self.feature_names = feature_info['feature_names']
        
        print(f"\n=== 모델 로드 완료 ===")
        print(f"모델 파일: {latest_model.name}")
        print(f"특성 파일: {feature_file.name}")
        
        return latest_model, feature_file
    
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
    
    # def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
    #     """데이터에서 특성 추출"""
    #     # 날짜 기준으로 정렬
    #     data['date'] = pd.to_datetime(data['date'])
    #     data = data.sort_values('date')
        
    #     # 특성 선택
    #     features = [
    #         # 기본 경기력 지표
    #         'home_rebounds', 'away_rebounds',
    #         'home_assists', 'away_assists',
    #         'home_fieldGoalsAttempted', 'away_fieldGoalsAttempted',
    #         'home_fieldGoalsMade', 'away_fieldGoalsMade',
    #         'home_fieldGoalPct', 'away_fieldGoalPct',
    #         'home_freeThrowsAttempted', 'away_freeThrowsAttempted',
    #         'home_freeThrowsMade', 'away_freeThrowsMade',
    #         'home_freeThrowPct', 'away_freeThrowPct',
    #         'home_threePointFieldGoalsAttempted', 'away_threePointFieldGoalsAttempted',
    #         'home_threePointFieldGoalsMade', 'away_threePointFieldGoalsMade',
    #         'home_threePointPct', 'away_threePointPct',
            
    #         # 리더 통계
    #         'home_leader_points', 'away_leader_points',
    #         'home_leader_rebounds', 'away_leader_rebounds',
    #         'home_leader_assists', 'away_leader_assists',
            
    #         # 팀 기록
    #         'home_overall_record_win_rate', 'away_overall_record_win_rate',
    #         'home_home_record_win_rate', 'away_home_record_win_rate',
    #         'home_road_record_win_rate', 'away_road_record_win_rate',
    #         'home_vs_away_win_rate',
            
    #         # 최근 트렌드
    #         'home_recent_win_rate', 'away_recent_win_rate',
    #         'home_recent_avg_score', 'away_recent_avg_score',
    #         'home_recent_home_win_rate', 'away_recent_home_win_rate',
    #         'home_recent_away_win_rate', 'away_recent_away_win_rate',
            
    #         # 컨디션
    #         'home_rest_days', 'away_rest_days'
    #     ]
        
    #     X = data[features]
        
    #     return X
    
    
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
                X[new_col] = X[orig_col]  # 동일한 값으로 복제
        
        return X
        
    def predict_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """경기 결과 예측"""
        X = self.prepare_features(df)
        
        # 승리 확률 예측
        win_probabilities = self.model.predict_proba(X)[:, 1]
        
        # 결과 DataFrame 생성
        results_df = df[['date', 'home_team_name', 'away_team_name']].copy()
        results_df['home_win_probability'] = win_probabilities
        results_df['predicted_winner'] = np.where(
            win_probabilities > 0.5,
            results_df['home_team_name'],
            results_df['away_team_name']
        )
        results_df['win_probability'] = np.where(
            win_probabilities > 0.5,
            win_probabilities,
            1 - win_probabilities
        )
        
        # 날짜 형식 변환
        results_df['date'] = pd.to_datetime(results_df['date']).dt.strftime('%Y-%m-%d')
        
        return results_df
    
    def save_predictions(self, predictions: pd.DataFrame) -> Path:
        """예측 결과 저장"""
        # 저장 경로 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "predictions"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"predictions_{timestamp}.json"
        
        # 예측 결과를 JSON으로 저장
        predictions.to_json(output_path, orient='records', indent=2)
        
        print(f"\n=== 예측 결과 저장 완료 ===")
        print(f"저장 경로: {output_path}")
        
        return output_path

def main():
    predictor = BettingPredictor()
    
    # 1. 최신 모델 로드
    predictor.load_latest_model()
    
    # 2. 예측할 데이터 로드
    pred_data = predictor.load_prediction_data()
    
    # 3. 예측 수행
    predictions = predictor.predict_games(pred_data)
    
    # 4. 결과 출력
    print("\n=== 예측 결과 ===")
    for _, row in predictions.iterrows():
        print(f"\n{row['date']} 경기:")
        print(f"{row['home_team_name']} vs {row['away_team_name']}")
        print(f"예상 승자: {row['predicted_winner']}")
        print(f"승리 확률: {row['win_probability']:.1%}")
    
    # 5. 결과 저장
    predictor.save_predictions(predictions)

if __name__ == "__main__":
    main() 