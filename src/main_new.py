import os
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Dict, Any
import subprocess
import time
import json
import argparse

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
            'models/model4.py',
            'models/model5.py',
            'models/model6.py',
            'models/model7.py',
            'models/model8.py',
            'models/betting_optimizer.py',
            'models/ensemble_predictor.py',
            'predict_ensemble.py',
            'odds_fetcher.py',
            'app_new2.py'
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
            'model4.py': lambda: self._check_files(self.dirs['src_data'], 'processed_*.json', 1),
            'model5.py': lambda: self._check_files(self.dirs['src_data'], 'processed_*.json', 1),
            'model6.py': lambda: self._check_files(self.dirs['src_data'], 'processed_*.json', 1),
            'model7.py': lambda: self._check_files(self.dirs['src_data'], 'processed_*.json', 1),
            'model8.py': lambda: self._check_files(self.dirs['src_data'], 'processed_*.json', 1),
            'processor_modelinput.py': lambda: (
                # 최소 1개 이상의 모델 파일이 있으면 됨 (모든 모델이 필수는 아님)
                self._check_files(self.dirs['saved_models'], 'betting_model*_*.joblib', 1)
            ),
            'predict_ensemble.py': lambda: self._check_files(self.dirs['src_data'], 'model_input_features_*.json', 1),
            'odds_fetcher.py': lambda: self._check_files(self.dirs['predictions'], 'ensemble_predictions_*.json', 1),
            'app_new2.py': lambda: self._check_files(self.project_root / 'data' / 'odds', 'processed_nba_odds_*.json', 1)
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
    
    def run_script(self, script_name: str, description: str, extra_args: list = None) -> bool:
        """Python 스크립트 실행
        
        Args:
            script_name: 실행할 스크립트 경로
            description: 작업 설명
            extra_args: 추가 커맨드라인 인자
        """
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
            
            # 명령어 구성
            cmd = [sys.executable, str(script_path)]
            if extra_args:
                cmd.extend(extra_args)
            
            result = subprocess.run(
                cmd,
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
    
    def run_pipeline(self, model_tag: str = 'active', skip_common_steps: bool = False) -> bool:
        """전체 파이프라인 실행
        
        Args:
            model_tag: 모델 태그 ('active', 'shadow', 'both')
            skip_common_steps: True면 공통 단계(데이터 수집 등) 스킵 (both 모드에서 shadow 실행 시 사용)
        """
        try:
            # 사전 검증 (공통 단계 스킵 시에도 검증은 수행)
            if not self.check_dependencies():
                return False
            if not self.validate_file_structure():
                return False
            
            self.logger.info(f"\n{'='*60}")
            if skip_common_steps:
                self.logger.info(f"🏀 NBA 파이프라인 시작 (모델 태그: {model_tag.upper()}, 공통 단계 스킵)")
            else:
                self.logger.info(f"🏀 NBA 파이프라인 시작 (모델 태그: {model_tag.upper()})")
            self.logger.info(f"{'='*60}")
            
            # steps: (스크립트, 설명, 추가인자, 공통단계여부)
            # 공통단계(True): 데이터 수집/처리 등 모델 태그와 무관한 작업
            # 모델별단계(False): 예측/병합 등 모델 태그에 따라 다르게 실행되는 작업
            steps = [
                ('data/espn_api.py', 'Collecting historical NBA data', None, True),
                ('data/processor_model.py', 'Processing historical data for model training', None, True),
                # ('models/model1.py', 'Training LightGBM model', None, True),
                # ('models/model2.py', 'Training CatBoost model', None, True),
                # ('models/model3.py', 'Training XGBoost model', None, True),
                # ('models/model4.py', 'Training LightGBM-DART model', None, True),
                # ('models/model5.py', 'Training CatBoost-Ordered model', None, True),
                # ('models/model6.py', 'Training XGBoost-Hist model', None, True),
                # ('models/model7.py', 'Training RandomForest model', None, True),
                # ('models/model8.py', 'Training ExtraTrees model', None, True),
                ('data/processor_modelinput.py', 'Preparing prediction input data', None, True),
                ('predict_ensemble.py', 'Running ensemble predictions', ['--model-tag', model_tag], False),
                ('odds_fetcher.py', 'Fetching current NBA odds', None, True),
                ('analysis/merge_predictions_odds.py', 'Merging predictions with odds data', ['--model-tag', model_tag], False),
                # ('app_new2.py', 'Starting web interface', None, True)
            ]
            
            # 실행할 단계만 필터링
            if skip_common_steps:
                executable_steps = [(s, d, a, c) for s, d, a, c in steps if not c]
                self.logger.info(f"공통 단계 스킵 - {len(executable_steps)}개 모델별 단계만 실행")
            else:
                executable_steps = steps
            
            for i, (script, description, extra_args, is_common) in enumerate(executable_steps):
                self.logger.info(f"\n=== Step {i+1}/{len(executable_steps)}: {description} ===")
                
                # 이전 단계의 결과물 확인
                if i > 0 or skip_common_steps:  # 공통 단계 스킵 시에도 첫 단계부터 검증
                    if not self.validate_previous_step_output(script):
                        self.logger.error(f"Previous step output missing for: {description}")
                        return False
                
                if not self.run_script(script, description, extra_args=extra_args):
                    self.logger.error(f"Pipeline failed at: {description}")
                    return False
                
                # 각 단계 완료 후 잠시 대기
                time.sleep(2)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"✅ 파이프라인 완료! (모델 태그: {model_tag.upper()})")
            self.logger.info(f"{'='*60}")
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
    
    parser = argparse.ArgumentParser(
        description='NBA 베팅 파이프라인 실행',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main_new.py                        # Active + Shadow 둘 다 실행 (기본값)
  python main_new.py --model-tag both       # Active와 Shadow 둘 다 실행
  python main_new.py --model-tag active     # Active 모델만 실행
  python main_new.py --model-tag shadow     # Shadow 모델만 실행
        """
    )
    parser.add_argument(
        '--model-tag', '-t',
        type=str,
        choices=['active', 'shadow', 'both'],
        default='both',
        help='사용할 모델 태그 (both: 둘 다 실행(기본값), active: 운영모델만, shadow: 테스트모델만)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = NBABettingPipeline()
    
    try:
        if args.model_tag == 'both':
            # Active와 Shadow 둘 다 실행
            print("\n" + "="*60)
            print("🏀 Active & Shadow 모델 파이프라인 실행")
            print("="*60)
            
            # Active: 전체 파이프라인 실행 (데이터 수집 포함)
            success_active = pipeline.run_pipeline(model_tag='active', skip_common_steps=False)
            
            # Shadow: 공통 단계 스킵 (예측 + 병합만 실행)
            success_shadow = pipeline.run_pipeline(model_tag='shadow', skip_common_steps=True)
            
            success = success_active and success_shadow
        else:
            success = pipeline.run_pipeline(model_tag=args.model_tag)
        
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