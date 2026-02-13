from models.ensemble_predictor import EnsemblePredictor
from pathlib import Path
import json
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


def parse_args():
    """커맨드라인 인자 파싱"""
    # Jupyter 환경에서는 기본값 사용 (둘 다 실행)
    if is_jupyter():
        class Args:
            model_tag = 'both'
        return Args()
    
    parser = argparse.ArgumentParser(description='NBA 앙상블 예측 실행')
    parser.add_argument(
        '--model-tag', '-t',
        type=str,
        choices=['active', 'shadow', 'fixed', 'both'],
        default='both',
        help='사용할 모델 태그 (both: 둘 다 실행(기본값), active: 운영모델, shadow: 테스트모델, fixed: 고정모델)'
    )
    return parser.parse_args()

def run_prediction(model_tag: str = 'active'):
    """단일 모델 세트로 예측 실행"""
    # 예측기 초기화
    predictor = EnsemblePredictor()
    
    # 모델 로드 (태그 지정)
    loaded_models = predictor.load_latest_models(model_tag=model_tag)
    print(f"\n총 {len(loaded_models)}개 [{model_tag.upper()}] 모델 로드 완료")
    
    # 예측 데이터 로드
    data = predictor.load_prediction_data()
    print(f"예측할 경기 수: {len(data)}")
    
    # 모델별 가중치 설정 (총합이 1이 되도록 자동 정규화됨)
    weights = {
        'model1': 0,    # LightGBM
        'model2': 1,    # CatBoost
        'model3': 0,    # XGBoost
        'model4': 0,    # LightGBM-DART
        'model5': 0,    # CatBoost-Ordered
        'model6': 0,    # XGBoost-Hist
        'model7': 0,    # RandomForest
        'model8': 0     # ExtraTrees
    }
    
    # 앙상블 예측 수행
    predictions = predictor.predict_games(data, weights=weights)
    
    # 예측 결과 저장 (태그 포함)
    output_path = predictor.save_predictions(predictions, model_tag=model_tag)
    
    return predictor, predictions, output_path

def main():
    args = parse_args()
    
    try:
        if args.model_tag == 'both':
            # Active와 Shadow 둘 다 실행
            print("\n" + "="*60)
            print("🏀 [ACTIVE 모델 예측 시작]")
            print("="*60)
            predictor_active, predictions_active, output_active = run_prediction('active')
            
            print("\n" + "="*60)
            print("🌙 [SHADOW 모델 예측 시작]")
            print("="*60)
            predictor_shadow, predictions_shadow, output_shadow = run_prediction('shadow')
            
            # 사용할 predictor와 predictions는 active 기준
            predictor = predictor_active
            predictions = predictions_active
            output_path = output_active
            
            print("\n" + "="*60)
            print("✅ Active & Shadow 모델 예측 완료!")
            print(f"  - Active: {output_active}")
            print(f"  - Shadow: {output_shadow}")
            print("="*60)
        else:
            # 단일 모델 세트 실행
            predictor, predictions, output_path = run_prediction(args.model_tag)
        
        # 예측 신뢰도 통계
        high_confidence = (predictions['win_probability'] >= 0.7).sum()
        medium_confidence = ((predictions['win_probability'] >= 0.6) & 
                           (predictions['win_probability'] < 0.7)).sum()
        low_confidence = (predictions['win_probability'] < 0.6).sum()
        
        # 모델 표시 이름 매핑
        model_display_names = {
            'model1': 'LightGBM',
            'model2': 'CatBoost',
            'model3': 'XGBoost',
            'model4': 'LightGBM-DART',
            'model5': 'CatBoost-Ordered',
            'model6': 'XGBoost-Hist',
            'model7': 'RandomForest',
            'model8': 'ExtraTrees'
        }
        
        # 앙상블 예측 결과 콘솔 출력
        tag_display = f"[{args.model_tag.upper()}] " if args.model_tag != 'both' else ""
        print(f"\n=== {tag_display}앙상블 예측 결과 ===")
        print(f"저장 경로: {output_path}")
        
        print(f"\n=== {tag_display}예측 신뢰도 분석 ===")
        print(f"높은 신뢰도 (70% 이상): {high_confidence}경기")
        print(f"중간 신뢰도 (60-70%): {medium_confidence}경기")
        print(f"낮은 신뢰도 (60% 미만): {low_confidence}경기")
            
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 