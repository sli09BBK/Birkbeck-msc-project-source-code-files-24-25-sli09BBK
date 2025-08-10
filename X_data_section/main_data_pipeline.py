import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
try:
    from enhanced_data_processor import EnhancedDataProcessor
    from advanced_user_analysis import AdvancedUserAnalysis
    from database_setup import DatabaseSetup, DatabaseConfig
except ImportError as e:
    print(f"Failed to import modules: {e}")
    print("Please ensure all necessary module files are in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Main class for the data processing pipeline"""

    def __init__(self, config_file: str = 'X_data_section/database_config.json'):
        """
        Initialize the data pipeline

        Args:
            config_file: Path to the database configuration file
        """
        self.config_file = config_file
        self.db_config = self.load_database_config()
        self.data_processor = None
        self.user_analyzer = None
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'total_records_processed': 0,
            'successful_operations': [],
            'failed_operations': [],
            'warnings': []
        }

    def load_database_config(self) -> Dict[str, str]:
        """Load database configuration"""
        try:
            if os.path.exists(self.config_file):
                config = DatabaseConfig.load_config(self.config_file)
                logger.info(f"Loaded database configuration from {self.config_file}")
            else:
                config = DatabaseConfig.get_default_config()
                logger.warning(f"Configuration file {self.config_file} does not exist, using default configuration")
                # Prompt user for password
                if not config['password']:
                    password = input("Please enter MySQL database password: ")
                    config['password'] = password
                    # Save configuration
                    DatabaseConfig.save_config(config, self.config_file)

            return config
        except Exception as e:
            logger.error(f"Failed to load database configuration: {e}")
            return DatabaseConfig.get_default_config()

    def setup_database(self) -> bool:
        """Set up the database"""
        try:
            logger.info("Starting database setup")

            db_setup = DatabaseSetup(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                port=int(self.db_config.get('port', 3306))
            )

            success = db_setup.setup_complete_database(self.db_config['database'])

            if success:
                self.pipeline_stats['successful_operations'].append('Database Setup')
                logger.info("Database setup complete")
            else:
                self.pipeline_stats['failed_operations'].append('Database Setup')
                logger.error("Database setup failed")

            db_setup.close_connection()
            return success

        except Exception as e:
            logger.error(f"An error occurred during database setup: {e}")
            self.pipeline_stats['failed_operations'].append(f'Database Setup: {str(e)}')
            return False

    def initialize_processors(self) -> bool:
        """Initialize data processors"""
        try:
            # Initialize data processor
            self.data_processor = EnhancedDataProcessor(self.db_config)
            if not self.data_processor.connect_to_database():
                logger.error("Data processor failed to connect to database")
                return False

            # Initialize user analyzer
            self.user_analyzer = AdvancedUserAnalysis(self.db_config)
            if not self.user_analyzer.connect_to_database():
                logger.error("User analyzer failed to connect to database")
                return False

            logger.info("Data processors initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            return False

    def process_csv_data(self, csv_files: List[str], batch_size: int = 1000) -> bool:
        """Process CSV data files"""
        try:
            if not self.data_processor:
                logger.error("Data processor not initialized")
                return False

            total_processed = 0
            successful_files = []
            failed_files = []


            for csv_file in csv_files:
                if not os.path.exists(csv_file):
                    logger.warning(f"CSV file does not exist: {csv_file}")
                    failed_files.append(csv_file)
                    continue

                try:
                    logger.info(f"Starting to process CSV file: {csv_file}")

                    # Get number of records in file
                    try:
                        df = pd.read_csv(csv_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(csv_file, encoding='gbk')

                    file_records = len(df)

                    # Process data
                    batch_id = self.data_processor.process_csv_data(csv_file, batch_size)

                    if batch_id:
                        successful_files.append(csv_file)
                        total_processed += file_records
                        logger.info(f"File {csv_file} processed, batch ID: {batch_id}")
                    else:
                        failed_files.append(csv_file)
                        logger.error(f"File {csv_file} failed to process")

                except Exception as e:
                    logger.error(f"An error occurred while processing file {csv_file}: {e}")
                    failed_files.append(csv_file)

            # Update statistics
            self.pipeline_stats['total_records_processed'] = total_processed

            if successful_files:
                self.pipeline_stats['successful_operations'].append(f'CSV Data Processing: {len(successful_files)} files')

            if failed_files:
                self.pipeline_stats['failed_operations'].append(f'CSV Processing Failed: {failed_files}')

            logger.info(f"CSV data processing complete, successful: {len(successful_files)}, failed: {len(failed_files)}")
            return len(successful_files) > 0

        except Exception as e:
            logger.error(f"An error occurred during CSV data processing: {e}")
            self.pipeline_stats['failed_operations'].append(f'CSV Data Processing: {str(e)}')
            return False

    def run_user_analysis(self) -> bool:
        """Run user behavior analysis"""
        try:
            if not self.user_analyzer:
                logger.error("User analyzer not initialized")
                return False

            logger.info("Starting user behavior analysis")

            # Run complete analysis
            results = self.user_analyzer.run_complete_analysis()

            if results:
                self.pipeline_stats['successful_operations'].append('User Behavior Analysis')
                logger.info(f"User behavior analysis complete, results saved in: {results['output_dir']}")
                return True
            else:
                self.pipeline_stats['failed_operations'].append('User Behavior Analysis')
                logger.error("User behavior analysis failed")
                return False

        except Exception as e:
            logger.error(f"An error occurred during user behavior analysis: {e}")
            self.pipeline_stats['failed_operations'].append(f'User Behavior Analysis: {str(e)}')
            return False

    def generate_data_quality_report(self) -> Dict:
        """Generate data quality report"""
        try:
            if not self.data_processor:
                logger.warning("Data processor not initialized, cannot generate quality report")
                return {}

            logger.info("Generating data quality report")

            # Get data quality report
            quality_report = self.data_processor.get_data_quality_report()

            # Add pipeline statistics
            quality_report['pipeline_stats'] = self.pipeline_stats

            # Save report
            report_file = f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"Data quality report saved: {report_file}")
            return quality_report

        except Exception as e:
            logger.error(f"Failed to generate data quality report: {e}")
            return {}

    def cleanup_old_data(self, retention_days: int = 365) -> bool:
        """Clean up old data"""
        try:
            if not self.data_processor:
                logger.warning("Data processor not initialized, cannot clean up data")
                return False

            logger.info(f"Starting to clean up data older than {retention_days} days")

            # Clean up old raw data
            cleanup_sql = """
                DELETE FROM raw_data 
                WHERE created_at < DATE_SUB(NOW(), INTERVAL %s DAY)
            """

            self.data_processor.cursor.execute(cleanup_sql, (retention_days,))
            deleted_rows = self.data_processor.cursor.rowcount

            self.data_processor.connection.commit()

            logger.info(f"Cleanup complete, deleted {deleted_rows} old records")
            self.pipeline_stats['successful_operations'].append(f'Data Cleanup: {deleted_rows} records')

            return True

        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            self.pipeline_stats['failed_operations'].append(f'Data Cleanup: {str(e)}')
            return False

    def run_complete_pipeline(self,
                              csv_files: List[str],
                              setup_db: bool = False,
                              run_analysis: bool = True,
                              cleanup_data: bool = False,
                              batch_size: int = 1000,
                              retention_days: int = 365) -> bool:
        """Run the complete data processing pipeline"""
        try:
            self.pipeline_stats['start_time'] = datetime.now()
            logger.info("Starting data processing pipeline")

            # 1. Database setup (optional)
            if setup_db:
                if not self.setup_database():
                    logger.error("Database setup failed, terminating pipeline")
                    return False

            # 2. Initialize processors
            if not self.initialize_processors():
                logger.error("Processor initialization failed, terminating pipeline")
                return False

            # 3. Process CSV data
            if csv_files:
                if not self.process_csv_data(csv_files, batch_size):
                    logger.error("CSV data processing failed")
                    # Do not terminate pipeline, continue with other operations

            # 4. Run user analysis (optional)
            if run_analysis:
                if not self.run_user_analysis():
                    logger.error("User analysis failed")
                    # Do not terminate pipeline, continue with other operations

            # 5. Clean up old data (optional)
            if cleanup_data:
                self.cleanup_old_data(retention_days)

            # 6. Generate quality report
            quality_report = self.generate_data_quality_report()

            self.pipeline_stats['end_time'] = datetime.now()

            # Calculate total time taken
            total_time = (self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']).total_seconds()

            logger.info(f"Data processing pipeline complete, total time taken: {total_time:.2f} seconds")
            logger.info(f"Successful operations: {len(self.pipeline_stats['successful_operations'])}")
            logger.info(f"Failed operations: {len(self.pipeline_stats['failed_operations'])}")

            # Print summary
            self.print_pipeline_summary()

            return len(self.pipeline_stats['failed_operations']) == 0

        except Exception as e:
            logger.error(f"Data processing pipeline execution failed: {e}")
            return False

        finally:
            # Clean up resources
            self.cleanup_resources()

    def print_pipeline_summary(self):
        """Print pipeline execution summary"""
        print("\n" + "=" * 60)
        print("📊 Data Processing Pipeline Execution Summary")
        print("=" * 60)

        if self.pipeline_stats['start_time'] and self.pipeline_stats['end_time']:
            duration = (self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']).total_seconds()
            print(f"⏱️  Execution Time: {duration:.2f} seconds")

        print(f"📈 Records Processed: {self.pipeline_stats['total_records_processed']}")

        print(f"\n✅ Successful Operations ({len(self.pipeline_stats['successful_operations'])}):")
        for operation in self.pipeline_stats['successful_operations']:
            print(f"   - {operation}")

        if self.pipeline_stats['failed_operations']:
            print(f"\n❌ Failed Operations ({len(self.pipeline_stats['failed_operations'])}):")
            for operation in self.pipeline_stats['failed_operations']:
                print(f"   - {operation}")

        if self.pipeline_stats['warnings']:
            print(f"\n⚠️  Warnings ({len(self.pipeline_stats['warnings'])}):")
            for warning in self.pipeline_stats['warnings']:
                print(f"   - {warning}")

        print("=" * 60)

    def cleanup_resources(self):
        """Clean up resources"""
        try:
            if self.data_processor:
                self.data_processor.close_connection()

            if self.user_analyzer:
                self.user_analyzer.close_connection()

            logger.info("Resource cleanup complete")

        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")


def find_csv_files(directory: str) -> List[str]:
    """Find CSV files in a directory"""
    csv_files = []

    if os.path.isfile(directory) and directory.endswith('.csv'):
        return [directory]

    if os.path.isdir(directory):
        # Common data file names
        priority_files = ['zongshuju.csv', 'rednote.csv', 'cleaned_output_data.csv']

        # First, look for priority files
        for priority_file in priority_files:
            priority_path = os.path.join(directory, priority_file)
            if os.path.exists(priority_path):
                csv_files.append(priority_path)

        # If no priority files are found, look for all CSV files
        if not csv_files:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))

    return csv_files


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Xiaohongshu Data Processing Pipeline')

    # Basic parameters
    parser.add_argument('--input', '-i', default='.',
                        help='Input CSV file or directory containing CSV files (default: current directory)')
    parser.add_argument('--config', '-c', default='X_data_section/database_config.json',
                        help='Path to the database configuration file')
    parser.add_argument('--batch-size', '-b', type=int, default=1000,
                        help='Batch size for processing')

    # Feature flags
    parser.add_argument('--setup-db', action='store_true',
                        help='Set up the database (use for first run)')
    parser.add_argument('--no-analysis', action='store_true',
                        help='Skip user behavior analysis')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up old data')
    parser.add_argument('--retention-days', type=int, default=365,
                        help='Number of days to retain data')

    # Log level
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        print("🚀 Starting Xiaohongshu Data Processing Pipeline")
        print(f"📂 Searching directory: {args.input}")

        # Find CSV files
        csv_files = find_csv_files(args.input)

        if not csv_files:
            print(f"❌ No CSV files found in {args.input}")
            print("💡 Please ensure the directory contains one of the following files:")
            print("   - zongshuju.csv")
            print("   - rednote.csv")
            print("   - cleaned_output_data.csv")
            print("   - or other .csv files")
            return 1

        print(f"📁 Found {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
            print(f"   - {os.path.basename(csv_file)} ({file_size:.2f} MB)")

        # Create data pipeline
        pipeline = DataPipeline(args.config)

        # Run pipeline
        success = pipeline.run_complete_pipeline(
            csv_files=csv_files,
            setup_db=args.setup_db,
            run_analysis=not args.no_analysis,
            cleanup_data=args.cleanup,
            batch_size=args.batch_size,
            retention_days=args.retention_days
        )

        if success:
            print("\n🎉 Data processing pipeline executed successfully!")
            print("📊 Check generated report files:")
            print("   - data_quality_report_*.json")
            print("   - Advanced User Analysis Results/ (if user analysis was run)")
            return 0
        else:
            print("\n💥 Data processing pipeline execution failed!")
            print("📋 Please check the log file: data_pipeline.log")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  User interrupted execution")
        return 1
    except Exception as e:
        print(f"\n💥 An error occurred during execution: {e}")
        logger.error(f"Main function execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
