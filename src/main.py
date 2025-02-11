import os
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Dict, Any
import subprocess
import time
import json

class NBABettingPipeline:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_logging()
        
        # 필요한 디렉토리 구조 정의 및 생성
        self.dirs = {
            'src_data': self.project_root / 'src' / 'data',
            'raw': self.project_root / 'data' / 'raw' / 'historical',
            'upcoming': self.project_root / 'data' / 'upcoming' / 'games',
            'saved_models': self.project_root / 'src' / 'models' / 'saved_models',
            'predictions': self.project_root / 'src' / 'predictions',
            'analysis': self.project_root / 'src' / 'analysis',
            'logs': self.project_root / 'logs'
        }
        
        # 디렉토리 생성 및 존재 확인
        for dir_path in self.dirs.values():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                if not dir_path.exists():
                    raise Exception(f"Failed to create directory: {dir_path}")
            except Exception as e:
                self.logger.error(f"Error creating directory {dir_path}: {str(e)}")
                raise
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'pipeline_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('NBABettingPipeline')
    
    def check_dependencies(self) -> bool:
        """필요한 라이브러리 체크"""
        required_packages = {
            'pandas': 'pandas',
            'numpy': 'numpy',
            'lightgbm': 'lightgbm',
            'catboost': 'catboost',
            'xgboost': 'xgboost',
            'sklearn': 'scikit-learn',
            'streamlit': 'streamlit',
            'requests': 'requests',
            'bayes_opt': 'bayesian-optimization'
        }
        
        missing_packages = []
        for package, pip_name in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(pip_name)
        
        if missing_packages:
            self.logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            self.logger.info("Install missing packages using: pip install " + " ".join(missing_packages))
            return False
        return True
    
    def validate_file_structure(self) -> bool:
        """필요한 파일들의 존재 여부 확인"""
        required_files = [
            'data/espn_api.py',
            'data/processor_model.py',
            'data/processor_modelinput.py',
            'models/model1.py',
            'models/model2.py',
            'models/model3.py',
            'models/betting_optimizer.py',
            'models/ensemble_predictor.py',
            'predict_ensemble.py',
            'app.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.project_root / 'src' / file).exists():
                missing_files.append(file)
        
        if missing_files:
            self.logger.error(f"Missing required files: {', '.join(missing_files)}")
            return False
        return True
    
    def validate_previous_step_output(self, current_script: str) -> bool:
        """각 단계별 필요한 입력 파일 존재 확인"""
        self.logger.info(f"Validating outputs for: {current_script}")
        
        validations = {
            'processor_model.py': lambda: self._check_files(self.dirs['raw'], 'nba_data_*.json', 1),
            'model1.py': lambda: self._check_files(self.dirs['src_data'], 'processed_*.json', 1),
            'model2.py': lambda: self._check_files(self.dirs['src_data'], 'processed_*.json', 1),
            'model3.py': lambda: self._check_files(self.dirs['src_data'], 'processed_*.json', 1),
            'processor_modelinput.py': lambda: (
                self._check_files(self.dirs['saved_models'], 'betting_model1_*.joblib', 1) and
                self._check_files(self.dirs['saved_models'], 'betting_model2_*.joblib', 1) and
                self._check_files(self.dirs['saved_models'], 'betting_model3_*.joblib', 1)
            ),
            'predict_ensemble.py': lambda: self._check_files(self.dirs['src_data'], 'model_input_features_*.json', 1),
            'app.py': lambda: self._check_files(self.dirs['predictions'], 'ensemble_predictions_*.json', 1)
        }
        
        script_name = current_script.split('/')[-1]
        if script_name in validations:
            return validations[script_name]()
        return True
    
    def _check_files(self, directory: Path, pattern: str, min_count: int) -> bool:
        """파일 존재 여부 및 개수 확인"""
        try:
            files = list(directory.glob(pattern))
            if len(files) < min_count:
                self.logger.error(f"Required files not found: {directory}/{pattern}")
                return False
            self.logger.info(f"Found {len(files)} files matching {pattern} in {directory}")
            return True
        except Exception as e:
            self.logger.error(f"Error checking files in {directory}: {str(e)}")
            return False
    
    def run_script(self, script_name: str, description: str) -> bool:
        """Python 스크립트 실행"""
        try:
            self.logger.info(f"Starting: {description}")
            script_path = self.project_root / 'src' / script_name
            
            if not script_path.exists():
                self.logger.error(f"Script not found: {script_path}")
                return False
            
            # 환경변수에 모든 중요 디렉토리 경로 추가
            env = os.environ.copy()
            env['PROJECT_ROOT'] = str(self.project_root)
            env['PYTHONPATH'] = str(self.project_root)
            env['DATA_DIR'] = str(self.project_root / 'data')
            env['UPCOMING_DIR'] = str(self.dirs['upcoming'])
            env['RAW_DIR'] = str(self.dirs['raw'])
            env['SRC_DATA_DIR'] = str(self.dirs['src_data'])
            
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=True,
                env=env,
                cwd=str(self.project_root)  # 작업 디렉토리를 프로젝트 루트로 설정
            )
            
            if result.stdout:
                self.logger.info(f"Output: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Stderr: {result.stderr}")
            
            self.logger.info(f"Completed: {description}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error in {description}: {str(e)}")
            self.logger.error(f"Stderr: {e.stderr}")
            return False
        
        except Exception as e:
            self.logger.error(f"Unexpected error in {description}: {str(e)}")
            return False
    
    def run_pipeline(self) -> bool:
        """전체 파이프라인 실행"""
        try:
            # 사전 검증
            if not self.check_dependencies():
                return False
            if not self.validate_file_structure():
                return False
            
            steps = [
                ('data/espn_api.py', 'Collecting historical NBA data'),
                ('data/processor_model.py', 'Processing historical data for model training'),
                ('models/model1.py', 'Training LightGBM model'),
                ('models/model2.py', 'Training CatBoost model'),
                ('models/model3.py', 'Training XGBoost model'),
                ('data/processor_modelinput.py', 'Preparing prediction input data'),
                ('predict_ensemble.py', 'Running ensemble predictions'),
                ('app.py', 'Starting web interface')
            ]
            
            for i, (script, description) in enumerate(steps):
                self.logger.info(f"\n=== Step {i+1}/{len(steps)}: {description} ===")
                
                # 이전 단계의 결과물 확인
                if i > 0:
                    if not self.validate_previous_step_output(script):
                        self.logger.error(f"Previous step output missing for: {description}")
                        return False
                
                if not self.run_script(script, description):
                    self.logger.error(f"Pipeline failed at: {description}")
                    return False
                
                # 각 단계 완료 후 잠시 대기
                time.sleep(2)
            
            self.logger.info("\n=== Pipeline completed successfully! ===")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            return False
    
    def cleanup_old_files(self, days: int = 7):
        """오래된 파일 정리"""
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        patterns = {
            self.dirs['raw']: ['nba_data_*.json'],
            self.dirs['src_data']: ['processed_*.json', 'model_input_features_*.json'],
            self.dirs['predictions']: ['ensemble_predictions_*.json'],
            self.dirs['analysis']: ['betting_analysis_*.json'],
            self.dirs['logs']: ['pipeline_*.log']
        }
        
        for directory, file_patterns in patterns.items():
            for pattern in file_patterns:
                for file_path in directory.glob(pattern):
                    if file_path.stat().st_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            self.logger.info(f"Removed old file: {file_path}")
                        except Exception as e:
                            self.logger.error(f"Error removing {file_path}: {str(e)}")

def main():
    pipeline = NBABettingPipeline()
    
    try:
        success = pipeline.run_pipeline()
        
        if success:
            print("\n=== Pipeline completed successfully! ===")
            print("You can now access the web interface.")
            # 오래된 파일 정리 (7일 이상된 파일)
            pipeline.cleanup_old_files()
        else:
            print("\n=== Pipeline failed! ===")
            print("Check the logs for details.")
            
    except KeyboardInterrupt:
        print("\n=== Pipeline interrupted by user ===")
        sys.exit(1)
    except Exception as e:
        print(f"\n=== Unexpected error: {str(e)} ===")
        sys.exit(1)

if __name__ == "__main__":
    main() 