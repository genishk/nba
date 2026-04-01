# src/main_totals.py
# NBA Total Score 예측 파이프라인

import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import subprocess
import time


class NBATotalsPipeline:
    """NBA 총점 예측 파이프라인"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_logging()
        
        # 디렉토리 구조
        self.dirs = {
            'src_data': self.project_root / 'src' / 'data',
            'raw': self.project_root / 'data' / 'raw' / 'historical',
            'upcoming': self.project_root / 'data' / 'upcoming' / 'games',
            'saved_models': self.project_root / 'src' / 'models' / 'saved_models',
            'predictions': self.project_root / 'src' / 'predictions',
            'analysis': self.project_root / 'src' / 'analysis',
            'odds': self.project_root / 'data' / 'odds',
            'logs': self.project_root / 'logs'
        }
        
        # 디렉토리 생성
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'totals_pipeline_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('NBATotalsPipeline')
    
    def run_script(self, script_name: str, description: str) -> bool:
        """Python 스크립트 실행"""
        try:
            self.logger.info(f"🚀 Starting: {description}")
            script_path = self.project_root / 'src' / script_name
            
            if not script_path.exists():
                self.logger.error(f"❌ Script not found: {script_path}")
                return False
            
            # 환경변수 설정
            env = os.environ.copy()
            env['PROJECT_ROOT'] = str(self.project_root)
            env['PYTHONPATH'] = str(self.project_root)
            
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=True,
                env=env,
                cwd=str(self.project_root)
            )
            
            if result.stdout:
                # 출력 로깅 (너무 길면 마지막 부분만)
                output_lines = result.stdout.strip().split('\n')
                if len(output_lines) > 30:
                    self.logger.info("Output (last 30 lines):")
                    for line in output_lines[-30:]:
                        self.logger.info(f"  {line}")
                else:
                    for line in output_lines:
                        self.logger.info(f"  {line}")
            
            self.logger.info(f"✅ Completed: {description}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Error in {description}: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {str(e)}")
            return False
    
    def _check_files(self, directory: Path, pattern: str, min_count: int = 1) -> bool:
        """파일 존재 확인"""
        files = list(directory.glob(pattern))
        if len(files) < min_count:
            self.logger.warning(f"⚠️ Files not found: {directory}/{pattern}")
            return False
        self.logger.info(f"📁 Found {len(files)} files: {pattern}")
        return True
    
    def run_pipeline(self) -> bool:
        """전체 Totals 파이프라인 실행"""
        try:
            self.logger.info("\n" + "="*60)
            self.logger.info("🏀 NBA TOTALS PREDICTION PIPELINE")
            self.logger.info("="*60)
            
            # 파이프라인 단계 정의
            # (스크립트, 설명, 필수여부)
            steps = [
                # === 데이터 수집 (기존과 동일) ===
                # ('data/espn_api.py', 'Collecting NBA game data', True),
                # ('data/processor_model.py', 'Processing training data', True),
                
                # === 모델 학습 (필요시 주석 해제) ===
                # ('models/model1_totals.py', 'Training LightGBM Totals model', False),
                # ('models/model2_totals.py', 'Training CatBoost Totals model', False),
                # ('models/model3_totals.py', 'Training XGBoost Totals model', False),
                # ('models/model4_totals.py', 'Training LightGBM-GBDT Totals model', False),
                # ('models/model5_totals.py', 'Training CatBoost-Ordered Totals model', False),
                # ('models/model6_totals.py', 'Training XGBoost-Hist Totals model', False),
                # ('models/model7_totals.py', 'Training RandomForest Totals model', False),
                # ('models/model8_totals.py', 'Training ExtraTrees Totals model', False),
                
                # === 예측 인풋 준비 (기존과 동일) ===
                # ('data/processor_modelinput.py', 'Preparing prediction input', True),
                
                # === Totals 전용 단계 ===
                ('predict_ensemble_totals.py', 'Running totals ensemble predictions', True),
                ('odds_fetcher_totals.py', 'Fetching totals odds (Over/Under)', True),
                ('analysis/merge_predictions_totals.py', 'Merging predictions with odds', True),
            ]
            
            total_steps = len([s for s in steps if not s[0].startswith('#')])
            current_step = 0
            
            for script, description, is_required in steps:
                # 주석 처리된 단계 스킵
                if script.startswith('#'):
                    continue
                    
                current_step += 1
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"📌 Step {current_step}/{total_steps}: {description}")
                self.logger.info(f"{'='*50}")
                
                success = self.run_script(script, description)
                
                if not success:
                    if is_required:
                        self.logger.error(f"❌ Pipeline failed at: {description}")
                        return False
                    else:
                        self.logger.warning(f"⚠️ Optional step failed, continuing...")
                
                time.sleep(1)
            
            self.logger.info("\n" + "="*60)
            self.logger.info("✅ TOTALS PIPELINE COMPLETED!")
            self.logger.info("="*60)
            
            # 결과 파일 위치 안내
            self.logger.info("\n📁 Output files:")
            self.logger.info(f"  - Predictions: {self.dirs['predictions']}/totals_predictions_*.json")
            self.logger.info(f"  - Merged: {self.dirs['analysis']}/merged_totals_*.json")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """메인 실행"""
    print("\n" + "="*60)
    print("🏀 NBA TOTALS PREDICTION PIPELINE")
    print("="*60)
    print("\nThis pipeline predicts total scores for NBA games")
    print("and generates Over/Under betting recommendations.\n")
    
    pipeline = NBATotalsPipeline()
    
    try:
        success = pipeline.run_pipeline()
        
        if success:
            print("\n✅ Pipeline completed successfully!")
        else:
            print("\n❌ Pipeline failed! Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

