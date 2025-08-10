import mysql.connector
from mysql.connector import Error
import logging
import json
from typing import Dict, Optional
from decimal import Decimal
import os  # Add os module import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Decimal type"""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


class DatabaseSetup:
    """Database setup and initialization class"""

    def __init__(self, host: str = 'localhost', user: str = 'root', password: str = '', port: int = 3306):
        """
        Initialize database setup

        Args:
            host: Database host address
            user: Database username
            password: Database password
            port: Database port
        """
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.connection = None
        self.cursor = None

        # Debug log
        logger.info(
            f"Initializing database connection parameters: host={host}, user={user}, port={port}, password length={len(password) if password else 0}")

    def connect_to_mysql(self) -> bool:
        """Connect to MySQL server (without specifying a database)"""
        try:
            # Ensure password is not None
            password = self.password if self.password is not None else ''

            # Debug log
            logger.info(
                f"Attempting to connect to MySQL: host={self.host}, user={self.user}, port={self.port}, password length={len(password)}")

            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=password,  # Ensure correct password is used
                port=self.port
            )
            self.cursor = self.connection.cursor()
            logger.info("Successfully connected to MySQL server")
            return True
        except Error as e:
            logger.error(f"Failed to connect to MySQL server: {e}")
            return False

    def create_database(self, database_name: str = 'rednote') -> bool:
        """Create database"""
        try:
            # Check if database exists
            self.cursor.execute("SHOW DATABASES")
            databases = [db[0] for db in self.cursor.fetchall()]

            if database_name not in databases:
                create_db_sql = f"CREATE DATABASE {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                self.cursor.execute(create_db_sql)
                logger.info(f"Database {database_name} created successfully")
            else:
                logger.info(f"Database {database_name} already exists")

            # Select database
            self.cursor.execute(f"USE {database_name}")
            return True

        except Error as e:
            logger.error(f"Failed to create database: {e}")
            return False

    def create_optimized_tables(self) -> bool:
        """Create optimized database table structure"""
        try:
            tables = {
                # Raw data table
                'raw_data': """
                    CREATE TABLE IF NOT EXISTS raw_data (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        title VARCHAR(500) NOT NULL COMMENT 'Title',
                        author VARCHAR(100) NOT NULL COMMENT 'Author',
                        interaction_count INT DEFAULT 0 COMMENT 'Interaction Count',
                        publish_time DATETIME COMMENT 'Publish Time',
                        content_url VARCHAR(500) COMMENT 'Content URL',
                        raw_count_text VARCHAR(50) COMMENT 'Raw Interaction Count Text',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation Time',
                        INDEX idx_author (author),
                        INDEX idx_publish_time (publish_time),
                        INDEX idx_interaction_count (interaction_count),
                        INDEX idx_created_at (created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Raw Scraped Data Table'
                """,

                # Cleaned data table
                'cleaned_data': """
                    CREATE TABLE IF NOT EXISTS cleaned_data (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        raw_data_id INT COMMENT 'Raw Data ID',
                        title VARCHAR(500) NOT NULL COMMENT 'Title',
                        author VARCHAR(100) NOT NULL COMMENT 'Author',
                        interaction_count INT DEFAULT 0 COMMENT 'Interaction Count',
                        publish_time DATETIME COMMENT 'Publish Time',
                        content_url VARCHAR(500) COMMENT 'Content URL',
                        title_length INT COMMENT 'Title Length',
                        special_char_count INT COMMENT 'Special Character Count',
                        sentiment_score FLOAT COMMENT 'Sentiment Score',
                        sentiment_label VARCHAR(20) COMMENT 'Sentiment Label',
                        keywords JSON COMMENT 'Keywords',
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Processing Time',
                        FOREIGN KEY (raw_data_id) REFERENCES raw_data(id) ON DELETE SET NULL,
                        UNIQUE KEY unique_title_author (title(255), author),
                        INDEX idx_author (author),
                        INDEX idx_sentiment (sentiment_score),
                        INDEX idx_interaction (interaction_count),
                        INDEX idx_publish_time (publish_time),
                        INDEX idx_processed_at (processed_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Cleaned Data Table'
                """,

                # User behavior analysis table
                'user_behavior_analysis': """
                    CREATE TABLE IF NOT EXISTS user_behavior_analysis (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        author VARCHAR(100) UNIQUE NOT NULL COMMENT 'Author',
                        content_count INT DEFAULT 0 COMMENT 'Content Count',
                        total_interactions INT DEFAULT 0 COMMENT 'Total Interactions',
                        avg_interactions FLOAT DEFAULT 0 COMMENT 'Average Interactions',
                        max_interactions INT DEFAULT 0 COMMENT 'Max Interactions',
                        interaction_std FLOAT DEFAULT 0 COMMENT 'Interaction Standard Deviation',
                        active_days INT DEFAULT 0 COMMENT 'Active Days',
                        daily_avg_posts FLOAT DEFAULT 0 COMMENT 'Daily Average Posts',
                        interaction_efficiency FLOAT DEFAULT 0 COMMENT 'Interaction Efficiency',
                        avg_sentiment FLOAT DEFAULT 0 COMMENT 'Average Sentiment',
                        avg_title_length FLOAT DEFAULT 0 COMMENT 'Average Title Length',
                        avg_special_chars FLOAT DEFAULT 0 COMMENT 'Average Special Character Count',
                        user_cluster INT COMMENT 'User Cluster',
                        cluster_label VARCHAR(50) COMMENT 'Cluster Label',
                        activity_score FLOAT COMMENT 'Activity Score',
                        influence_score FLOAT COMMENT 'Influence Score',
                        anomaly_score FLOAT COMMENT 'Anomaly Score',
                        is_anomaly BOOLEAN DEFAULT FALSE COMMENT 'Is Anomaly User',
                        top_keywords JSON COMMENT 'Top Keywords',
                        first_post_date DATETIME COMMENT 'First Post Date',
                        last_post_date DATETIME COMMENT 'Last Post Date',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update Time',
                        INDEX idx_cluster (user_cluster),
                        INDEX idx_efficiency (interaction_efficiency),
                        INDEX idx_activity_score (activity_score),
                        INDEX idx_influence_score (influence_score),
                        INDEX idx_updated_at (updated_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='User Behavior Analysis Table'
                """,

                # Keyword analysis table
                'keyword_analysis': """
                    CREATE TABLE IF NOT EXISTS keyword_analysis (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        keyword VARCHAR(100) NOT NULL COMMENT 'Keyword',
                        frequency INT DEFAULT 0 COMMENT 'Frequency',
                        total_interactions INT DEFAULT 0 COMMENT 'Total Interactions',
                        avg_interactions FLOAT DEFAULT 0 COMMENT 'Average Interactions',
                        associated_authors JSON COMMENT 'Associated Authors',
                        sentiment_distribution JSON COMMENT 'Sentiment Distribution',
                        trend_score FLOAT DEFAULT 0 COMMENT 'Trend Score',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation Time',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update Time',
                        UNIQUE KEY unique_keyword (keyword),
                        INDEX idx_frequency (frequency),
                        INDEX idx_avg_interactions (avg_interactions),
                        INDEX idx_trend_score (trend_score),
                        INDEX idx_updated_at (updated_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Keyword Analysis Table'
                """,

                # Data quality monitoring table
                'data_quality_metrics': """
                    CREATE TABLE IF NOT EXISTS data_quality_metrics (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        batch_id VARCHAR(100) NOT NULL COMMENT 'Batch ID',
                        total_records INT COMMENT 'Total Records',
                        valid_records INT COMMENT 'Valid Records',
                        duplicate_records INT COMMENT 'Duplicate Records',
                        invalid_records INT COMMENT 'Invalid Records',
                        processing_time_seconds FLOAT COMMENT 'Processing Time (seconds)',
                        error_details JSON COMMENT 'Error Details',
                        data_source VARCHAR(100) COMMENT 'Data Source',
                        quality_score FLOAT COMMENT 'Quality Score',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation Time',
                        INDEX idx_batch_id (batch_id),
                        INDEX idx_quality_score (quality_score),
                        INDEX idx_created_at (created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Data Quality Metrics Table'
                """,

                # Prediction models table
                'prediction_models': """
                    CREATE TABLE IF NOT EXISTS prediction_models (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        model_name VARCHAR(100) NOT NULL COMMENT 'Model Name',
                        model_type VARCHAR(50) NOT NULL COMMENT 'Model Type',
                        target_variable VARCHAR(100) NOT NULL COMMENT 'Target Variable',
                        features JSON COMMENT 'Feature List',
                        model_params JSON COMMENT 'Model Parameters',
                        performance_metrics JSON COMMENT 'Performance Metrics',
                        model_file_path VARCHAR(500) COMMENT 'Model File Path',
                        is_active BOOLEAN DEFAULT TRUE COMMENT 'Is Active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation Time',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update Time',
                        INDEX idx_model_name (model_name),
                        INDEX idx_model_type (model_type),
                        INDEX idx_is_active (is_active),
                        INDEX idx_updated_at (updated_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Prediction Models Table'
                """,

                # System configuration table
                'system_config': """
                    CREATE TABLE IF NOT EXISTS system_config (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        config_key VARCHAR(100) UNIQUE NOT NULL COMMENT 'Configuration Key',
                        config_value TEXT COMMENT 'Configuration Value',
                        config_type VARCHAR(50) DEFAULT 'string' COMMENT 'Configuration Type',
                        description TEXT COMMENT 'Configuration Description',
                        is_active BOOLEAN DEFAULT TRUE COMMENT 'Is Active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation Time',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update Time',
                        INDEX idx_config_key (config_key),
                        INDEX idx_is_active (is_active)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='System Configuration Table'
                """,

                # Task schedule table
                'task_schedule': """
                    CREATE TABLE IF NOT EXISTS task_schedule (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        task_name VARCHAR(100) NOT NULL COMMENT 'Task Name',
                        task_type VARCHAR(50) NOT NULL COMMENT 'Task Type',
                        schedule_expression VARCHAR(100) COMMENT 'Schedule Expression',
                        task_params JSON COMMENT 'Task Parameters',
                        last_run_time DATETIME COMMENT 'Last Run Time',
                        next_run_time DATETIME COMMENT 'Next Run Time',
                        run_count INT DEFAULT 0 COMMENT 'Run Count',
                        success_count INT DEFAULT 0 COMMENT 'Success Count',
                        failure_count INT DEFAULT 0 COMMENT 'Failure Count',
                        is_active BOOLEAN DEFAULT TRUE COMMENT 'Is Active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation Time',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update Time',
                        INDEX idx_task_name (task_name),
                        INDEX idx_task_type (task_type),
                        INDEX idx_next_run_time (next_run_time),
                        INDEX idx_is_active (is_active)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Task Schedule Table'
                """
            }

            for table_name, create_sql in tables.items():
                try:
                    self.cursor.execute(create_sql)
                    logger.info(f"Table {table_name} created successfully")
                except Error as e:
                    logger.error(f"Failed to create table {table_name}: {e}")
                    return False

            self.connection.commit()
            logger.info("All tables created")
            return True

        except Exception as e:
            logger.error(f"Error occurred while creating tables: {e}")
            return False

    def insert_default_config(self) -> bool:
        """Insert default configurations"""
        try:
            default_configs = [
                ('batch_size', '1000', 'int', 'Data processing batch size'),
                ('sentiment_threshold_positive', '0.6', 'float', 'Positive sentiment threshold'),
                ('sentiment_threshold_negative', '0.4', 'float', 'Negative sentiment threshold'),
                ('keyword_extract_count', '5', 'int', 'Number of keywords to extract'),
                ('cluster_min_users', '3', 'int', 'Minimum number of users for clustering'),
                ('anomaly_threshold', '0.1', 'float', 'Anomaly detection threshold'),
                ('model_retrain_days', '7', 'int', 'Model retraining interval in days'),
                ('data_retention_days', '365', 'int', 'Data retention period in days'),
                ('enable_auto_analysis', 'true', 'boolean', 'Enable automatic analysis'),
                ('enable_notifications', 'true', 'boolean', 'Enable notifications')
            ]

            for config_key, config_value, config_type, description in default_configs:
                insert_sql = """
                    INSERT INTO system_config (config_key, config_value, config_type, description)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    config_value = VALUES(config_value),
                    config_type = VALUES(config_type),
                    description = VALUES(description)
                """
                self.cursor.execute(insert_sql, (config_key, config_value, config_type, description))

            self.connection.commit()
            logger.info("Default configurations inserted")
            return True

        except Exception as e:
            logger.error(f"Failed to insert default configurations: {e}")
            return False

    def create_indexes_and_optimize(self) -> bool:
        """Create additional indexes and optimize the database"""
        try:
            # Create compound indexes
            additional_indexes = [
                "CREATE INDEX idx_author_publish_time ON cleaned_data(author, publish_time)",
                "CREATE INDEX idx_sentiment_interaction ON cleaned_data(sentiment_score, interaction_count)",
                "CREATE INDEX idx_cluster_influence ON user_behavior_analysis(user_cluster, influence_score)",
                "CREATE INDEX idx_keyword_frequency ON keyword_analysis(keyword, frequency)"
            ]

            for index_sql in additional_indexes:
                try:
                    self.cursor.execute(index_sql)
                    logger.info(f"Index created successfully: {index_sql.split()[2]}")
                except Error as e:
                    if "Duplicate key name" not in str(e):
                        logger.warning(f"Failed to create index: {e}")

            # Database optimization settings
            optimization_settings = [
                "SET GLOBAL innodb_buffer_pool_size = 1073741824",  # 1GB
                "SET GLOBAL max_connections = 200"
            ]

            # Attempt to set query_cache_size, but this has been removed in MySQL 8.0+
            try:
                self.cursor.execute("SET GLOBAL query_cache_size = 67108864")  # 64MB
                logger.info("Optimization setting applied: SET GLOBAL query_cache_size = 67108864")
            except Error as e:
                if "Unknown system variable" in str(e):
                    logger.info("Skipping query_cache_size setting (not supported by MySQL 8.0+)")
                else:
                    logger.warning(f"Failed to apply query_cache_size setting: {e}")

            for setting in optimization_settings:
                try:
                    self.cursor.execute(setting)
                    logger.info(f"Optimization setting applied: {setting}")
                except Error as e:
                    logger.warning(f"Failed to apply optimization setting: {e}")

            self.connection.commit()
            logger.info("Database optimization complete")
            return True

        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return False

    def get_database_info(self) -> Dict:
        """Get database information"""
        try:
            info = {}

            # Get table information
            self.cursor.execute("SHOW TABLES")
            tables = [table[0] for table in self.cursor.fetchall()]
            info['tables'] = tables

            # Get record count for each table
            table_counts = {}
            for table in tables:
                try:
                    self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = self.cursor.fetchone()[0]
                    table_counts[table] = count
                except Exception: # Catch specific errors like table not found if desired
                    table_counts[table] = 0

            info['table_counts'] = table_counts

            # Get database size
            self.cursor.execute("""
                SELECT
                    table_schema as 'Database',
                    ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) as 'Size_MB'
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
                GROUP BY table_schema
            """)

            result = self.cursor.fetchone()
            if result:
                info['database_size_mb'] = result[1]

            return info

        except Exception as e:
            logger.error(f"Failed to retrieve database information: {e}")
            return {}

    def backup_database(self, backup_path: str) -> bool:
        """Backup database (requires mysqldump tool)"""
        try:
            import subprocess

            # Construct mysqldump command
            cmd = [
                'mysqldump',
                f'--host={self.host}',
                f'--user={self.user}',
                f'--password={self.password}',
                f'--port={self.port}',
                '--single-transaction',
                '--routines',
                '--triggers',
                'rednote'
            ]

            # Execute backup
            with open(backup_path, 'w', encoding='utf-8') as backup_file:
                result = subprocess.run(cmd, stdout=backup_file, stderr=subprocess.PIPE, text=True)

            if result.returncode == 0:
                logger.info(f"Database backup successful: {backup_path}")
                return True
            else:
                logger.error(f"Database backup failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error occurred during database backup: {e}")
            return False

    def setup_complete_database(self, database_name: str = 'rednote') -> bool:
        """Complete database setup process"""
        try:
            logger.info("Starting complete database setup process")

            # 1. Connect to MySQL server
            if not self.connect_to_mysql():
                return False

            # 2. Create database
            if not self.create_database(database_name):
                return False

            # 3. Create tables
            if not self.create_optimized_tables():
                return False

            # 4. Insert default configurations
            if not self.insert_default_config():
                return False

            # 5. Create indexes and optimize
            if not self.create_indexes_and_optimize():
                return False

            # 6. Get database information
            db_info = self.get_database_info()
            logger.info(f"Database setup complete, info: {json.dumps(db_info, indent=2, ensure_ascii=False, cls=DecimalEncoder)}")

            return True

        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False

    def close_connection(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed")


# Configuration management class
class DatabaseConfig:
    """Database configuration management"""

    @staticmethod
    def get_default_config() -> Dict[str, str]:
        """Get default database configuration"""
        return {
            'host': '127.0.0.1',
            'user': 'root',
            'password': 'root',  # Please modify to your actual password
            'database': 'rednote',
            'port': 3306,
            'charset': 'utf8mb4'
        }

    @staticmethod
    def save_config(config: Dict[str, str], config_file: str = 'X_data_section/database_config.json'):
        """Save database configuration to file"""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False, cls=DecimalEncoder)
            logger.info(f"Database configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    @staticmethod
    def load_config(config_file: str = 'X_data_section/database_config.json') -> Dict[str, str]:
        """Load database configuration from file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Database configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"Configuration file {config_file} not found, using default configuration")
            return DatabaseConfig.get_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return DatabaseConfig.get_default_config()


# Example usage and main function
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Database Setup Tool')
    parser.add_argument('--host', default='127.0.0.1', help='Database host address')
    parser.add_argument('--user', default='root', help='Database username')
    parser.add_argument('--password', default='root', help='Database password')
    parser.add_argument('--port', type=int, default=3306, help='Database port')
    parser.add_argument('--database', default='rednote', help='Database name')
    parser.add_argument('--backup', help='Backup file path')
    parser.add_argument('--save-config', action='store_true', help='Save configuration to file')
    parser.add_argument('--config-file', default='X_data_section/database_config.json', help='Configuration file path')

    args = parser.parse_args()

    # Attempt to load from configuration file
    config = None
    if os.path.exists(args.config_file):
        try:
            config = DatabaseConfig.load_config(args.config_file)
            logger.info(f"Loaded from config file: {args.config_file}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")

    # If command line arguments provide password, prioritize command line arguments
    if args.password:
        password = args.password
    elif config and config.get('password'):
        password = config.get('password')
    else:
        password = ''

    # Create database setup instance
    db_setup = DatabaseSetup(
        host=args.host,
        user=args.user,
        password=password,  # Ensure correct password is used
        port=args.port
    )

    try:
        # Execute complete setup
        success = db_setup.setup_complete_database(args.database)

        if success:
            print("✅ Database setup complete!")

            # Save configuration
            if args.save_config:
                config = {
                    'host': args.host,
                    'user': args.user,
                    'password': password,
                    'database': args.database,
                    'port': args.port,
                    'charset': 'utf8mb4'
                }
                DatabaseConfig.save_config(config)

            # Execute backup
            if args.backup:
                db_setup.backup_database(args.backup)

            # Display database information
            db_info = db_setup.get_database_info()
            print("\n📊 Database Information:")
            print(f"Table Count: {len(db_info.get('tables', []))}")
            print(f"Database Size: {db_info.get('database_size_mb', 0)} MB")
            print("\n📋 Table List:")
            for table, count in db_info.get('table_counts', {}).items():
                print(f"  - {table}: {count} records")
        else:
            print("❌ Database setup failed!")

    finally:
        db_setup.close_connection()
