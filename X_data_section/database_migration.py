import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pymysql
from pymysql.cursors import DictCursor
from decimal import Decimal

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_migration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from fixed_enhanced_data_processor import EnhancedDataProcessor
    from fixed_advanced_user_analysis import AdvancedUserAnalysis
    from fixed_database_setup import DatabaseSetup, DatabaseConfig
except ImportError as e:
    print("Module import failed, please check the following:")
    print(f"Error message: {e}")
    print("Available Python files:")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(current_dir):
        if file.endswith('.py'):
            print(f"  - {file}")
    print("\nPlease ensure the following files exist:")
    print("  - fixed_enhanced_data_processor.py")
    print("  - fixed_advanced_user_analysis.py")
    print("  - fixed_database_setup.py")
    sys.exit(1)


class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Decimal type"""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super(DecimalEncoder, self).default(obj)


class DatabaseConfig:
    """Database configuration class"""

    @staticmethod
    def get_default_config() -> Dict[str, str]:
        """Get default configuration"""
        return {
            'host': '127.0.0.1',
            'user': 'root',
            'password': 'root',
            'database': 'rednote',
            'port': '3306',
            'charset': 'utf8mb4'
        }

    @staticmethod
    def load_config(config_file: str) -> Dict[str, str]:
        """Load configuration from file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Database configuration loaded from {config_file}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}, using default configuration")
            return DatabaseConfig.get_default_config()

    @staticmethod
    def save_config(config: Dict[str, str], config_file: str):
        """Save database configuration to file"""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False, cls=DecimalEncoder)
            logger.info(f"Database configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")


class RawDataModel:
    """Raw Data Model"""

    def __init__(self, title=None, content=None, author=None, publish_time=None,
                 like_count=None, comment_count=None, share_count=None, view_count=None,
                 url=None, platform=None, batch_id=None, raw_data=None):
        self.title = title
        self.content = content
        self.author = author
        self.publish_time = publish_time
        self.like_count = like_count
        self.comment_count = comment_count
        self.share_count = share_count
        self.view_count = view_count
        self.url = url
        self.platform = platform
        self.batch_id = batch_id
        self.raw_data = raw_data

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'title': self.title,
            'content': self.content,
            'author': self.author,
            'publish_time': self.publish_time,
            'like_count': self.like_count,
            'comment_count': self.comment_count,
            'share_count': self.share_count,
            'view_count': self.view_count,
            'url': self.url,
            'platform': self.platform,
            'batch_id': self.batch_id,
            'raw_data': self.raw_data
        }


class DatabaseSetup:
    """Database Setup Class"""

    def __init__(self, host: str, user: str, password: str, port: int = 3306):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.connection = None
        self.cursor = None

    def connect_to_server(self) -> bool:
        """Connect to MySQL server"""
        try:
            # Clean spaces from username and password
            clean_host = self.host.strip()
            clean_user = self.user.strip()
            clean_password = self.password.strip() if self.password else ''

            # Debug log
            logger.info(f"Attempting to connect to MySQL server: host={clean_host}, user='{clean_user}', port={self.port}")

            self.connection = pymysql.connect(
                host=clean_host,
                user=clean_user,
                password=clean_password,
                port=self.port,
                charset='utf8mb4',
                cursorclass=DictCursor
            )
            self.cursor = self.connection.cursor()
            logger.info("Successfully connected to MySQL server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MySQL server: {e}")
            if "Access denied" in str(e):
                logger.error("Authentication failed, please check:")
                logger.error(f"- Username: '{clean_user}'")
                logger.error("- Password is correct")
                logger.error("- MySQL service is running")
                logger.error("- User permissions are sufficient")
            return False

    def create_database(self, database_name: str) -> bool:
        """Create database"""
        try:
            # Check if database exists
            self.cursor.execute(f"SHOW DATABASES LIKE '{database_name}'")
            if self.cursor.fetchone():
                logger.info(f"Database {database_name} already exists")
                return True

            # Create database
            create_db_sql = f"""
                CREATE DATABASE {database_name} 
                CHARACTER SET utf8mb4 
                COLLATE utf8mb4_unicode_ci
            """
            self.cursor.execute(create_db_sql)
            logger.info(f"Database {database_name} created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            return False

    def create_tables(self, database_name: str) -> bool:
        """Create all tables"""
        try:
            # Use database
            self.cursor.execute(f"USE {database_name}")

            # Table definitions
            tables = {
                'raw_data': """
                    CREATE TABLE IF NOT EXISTS raw_data (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        title VARCHAR(500),
                        content TEXT,
                        author VARCHAR(100),
                        publish_time DATETIME,
                        like_count VARCHAR(50),
                        comment_count VARCHAR(50),
                        share_count VARCHAR(50),
                        view_count VARCHAR(50),
                        url VARCHAR(500),
                        platform VARCHAR(50) DEFAULT '小红书',
                        batch_id VARCHAR(100),
                        raw_data JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_author (author),
                        INDEX idx_publish_time (publish_time),
                        INDEX idx_platform (platform),
                        INDEX idx_batch_id (batch_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """,

                'cleaned_data': """
                    CREATE TABLE IF NOT EXISTS cleaned_data (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        raw_data_id INT,
                        title VARCHAR(500) NOT NULL,
                        author VARCHAR(100) NOT NULL,
                        interaction_count INT DEFAULT 0,
                        publish_time DATETIME,
                        content_url VARCHAR(500),
                        title_length INT,
                        special_char_count INT,
                        sentiment_score FLOAT,
                        sentiment_label VARCHAR(20),
                        keywords JSON,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (raw_data_id) REFERENCES raw_data(id) ON DELETE CASCADE,
                        UNIQUE KEY unique_title_author (title(255), author),
                        INDEX idx_author (author),
                        INDEX idx_sentiment (sentiment_score),
                        INDEX idx_interaction (interaction_count)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """,

                'user_behavior_analysis': """
                    CREATE TABLE IF NOT EXISTS user_behavior_analysis (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        author VARCHAR(100) UNIQUE NOT NULL,
                        content_count INT DEFAULT 0,
                        total_interactions INT DEFAULT 0,
                        avg_interactions FLOAT DEFAULT 0,
                        max_interactions INT DEFAULT 0,
                        interaction_std FLOAT DEFAULT 0,
                        active_days INT DEFAULT 0,
                        daily_avg_posts FLOAT DEFAULT 0,
                        interaction_efficiency FLOAT DEFAULT 0,
                        avg_sentiment FLOAT DEFAULT 0,
                        avg_title_length FLOAT DEFAULT 0,
                        avg_special_chars FLOAT DEFAULT 0,
                        user_cluster INT,
                        cluster_label VARCHAR(50),
                        top_keywords JSON,
                        first_post_date DATETIME,
                        last_post_date DATETIME,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_cluster (user_cluster),
                        INDEX idx_efficiency (interaction_efficiency)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """,

                'keyword_analysis': """
                    CREATE TABLE IF NOT EXISTS keyword_analysis (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        keyword VARCHAR(100) NOT NULL,
                        frequency INT DEFAULT 0,
                        total_interactions INT DEFAULT 0,
                        avg_interactions FLOAT DEFAULT 0,
                        associated_authors JSON,
                        sentiment_distribution JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        UNIQUE KEY unique_keyword (keyword),
                        INDEX idx_frequency (frequency),
                        INDEX idx_avg_interactions (avg_interactions)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """,

                'data_quality_metrics': """
                    CREATE TABLE IF NOT EXISTS data_quality_metrics (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        batch_id VARCHAR(100) NOT NULL,
                        total_records INT,
                        valid_records INT,
                        duplicate_records INT,
                        invalid_records INT,
                        processing_time_seconds FLOAT,
                        error_details JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_batch_id (batch_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """,

                'prediction_models': """
                    CREATE TABLE IF NOT EXISTS prediction_models (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        model_name VARCHAR(100) NOT NULL,
                        model_type VARCHAR(50),
                        target_variable VARCHAR(100),
                        features JSON,
                        model_params JSON,
                        performance_metrics JSON,
                        model_data LONGBLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_model_name (model_name),
                        INDEX idx_model_type (model_type)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """,

                'system_config': """
                    CREATE TABLE IF NOT EXISTS system_config (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        config_key VARCHAR(100) UNIQUE NOT NULL,
                        config_value JSON,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_config_key (config_key)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """,

                'task_schedule': """
                    CREATE TABLE IF NOT EXISTS task_schedule (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        task_name VARCHAR(100) NOT NULL,
                        task_type VARCHAR(50),
                        schedule_expression VARCHAR(100),
                        task_config JSON,
                        last_run_time DATETIME,
                        next_run_time DATETIME,
                        status VARCHAR(20) DEFAULT 'active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_task_name (task_name),
                        INDEX idx_status (status),
                        INDEX idx_next_run_time (next_run_time)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            }

            # Create tables
            for table_name, create_sql in tables.items():
                self.cursor.execute(create_sql)
                logger.info(f"Table {table_name} created successfully")

            self.connection.commit()
            logger.info("All tables created")
            return True

        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False

    def setup_complete_database(self, database_name: str) -> bool:
        """Complete database setup"""
        try:
            logger.info("Starting complete database setup process")

            # Connect to server
            if not self.connect_to_server():
                return False

            # Create database
            if not self.create_database(database_name):
                return False

            # Create tables
            if not self.create_tables(database_name):
                return False

            # Insert default configurations
            self.insert_default_configs()

            # Create indexes and optimize
            self.create_additional_indexes()
            self.apply_database_optimizations()

            logger.info("Database setup complete")
            return True

        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False

    def insert_default_configs(self):
        """Insert default configurations"""
        try:
            default_configs = [
                ('data_processing_batch_size', 1000, 'Data processing batch size'),
                ('sentiment_analysis_enabled', True, 'Enable sentiment analysis'),
                ('keyword_extraction_enabled', True, 'Enable keyword extraction'),
                ('auto_clustering_enabled', True, 'Enable automatic clustering'),
                ('model_retrain_interval_days', 7, 'Model retraining interval in days')
            ]

            for config_key, config_value, description in default_configs:
                insert_sql = """
                    INSERT INTO system_config (config_key, config_value, description)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    config_value = VALUES(config_value),
                    description = VALUES(description)
                """
                # Use custom encoder for JSON serialization
                config_value_json = json.dumps(config_value, cls=DecimalEncoder, ensure_ascii=False)
                self.cursor.execute(insert_sql, (config_key, config_value_json, description))

            self.connection.commit()
            logger.info("Default configurations inserted")

        except Exception as e:
            logger.error(f"Failed to insert default configurations: {e}")

    def create_additional_indexes(self):
        """Create additional indexes"""
        try:
            additional_indexes = [
                "CREATE INDEX IF NOT EXISTS idx_raw_data_created_at ON raw_data(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_cleaned_data_processed_at ON cleaned_data(processed_at)",
                "CREATE INDEX IF NOT EXISTS idx_user_behavior_updated_at ON user_behavior_analysis(updated_at)"
            ]

            for index_sql in additional_indexes:
                try:
                    self.cursor.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

            self.connection.commit()

        except Exception as e:
            logger.error(f"Failed to create additional indexes: {e}")

    def apply_database_optimizations(self):
        """Apply database optimization settings"""
        try:
            optimizations = [
                "SET GLOBAL innodb_buffer_pool_size = 1073741824",  # 1GB
                "SET GLOBAL max_connections = 200"
            ]

            for opt_sql in optimizations:
                try:
                    self.cursor.execute(opt_sql)
                    logger.info(f"Optimization setting applied: {opt_sql}")
                except Exception as e:
                    logger.warning(f"Failed to apply optimization setting: {e}")

            logger.info("Database optimization complete")

        except Exception as e:
            logger.error(f"Database optimization failed: {e}")

    def close_connection(self):
        """Close connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Failed to close connection: {e}")


class DatabaseMigration:
    """Database Migration Class"""

    def __init__(self, config: Dict[str, str]):
        """
        Initialize database migration

        Args:
            config: Database configuration
        """
        self.config = config
        self.connection = None
        self.cursor = None
        self.migration_stats = {
            'start_time': None,
            'end_time': None,
            'migrated_tables': [],
            'migrated_records': 0,
            'failed_migrations': [],
            'warnings': []
        }

    def connect_to_database(self) -> bool:
        """Connect to database"""
        try:
            # Clean spaces from configuration
            clean_config = {
                'host': self.config['host'].strip(),
                'user': self.config['user'].strip(),
                'password': self.config['password'].strip() if self.config['password'] else '',
                'database': self.config['database'].strip(),
                'port': int(self.config.get('port', 3306)),
                'charset': 'utf8mb4'
            }

            # Debug log
            logger.info(
                f"Attempting to connect to database: host={clean_config['host']}, user='{clean_config['user']}', database={clean_config['database']}")

            self.connection = pymysql.connect(
                host=clean_config['host'],
                user=clean_config['user'],
                password=clean_config['password'],
                database=clean_config['database'],
                port=clean_config['port'],
                charset=clean_config['charset'],
                cursorclass=DictCursor,
                autocommit=False
            )
            logger.info("Database connection successful")
            self.cursor = self.connection.cursor()
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            # Provide more detailed error message
            if "Access denied" in str(e):
                logger.error("Possible solutions:")
                logger.error("1. Check MySQL username and password are correct")
                logger.error("2. Confirm MySQL service is running")
                logger.error("3. Check user has permissions to access this database")
                logger.error("4. Try connecting to MySQL from the command line using the same credentials")
            return False

    def close_connection(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Failed to close database connection: {e}")

    def check_existing_tables(self) -> Dict[str, bool]:
        """Check existing table structure"""
        try:
            # Get all table names
            self.cursor.execute("SHOW TABLES")
            existing_tables = [row[f'Tables_in_{self.config["database"]}'] for row in self.cursor.fetchall()]

            table_status = {
                'raw_data': 'raw_data' in existing_tables,
                'cleaned_data': 'cleaned_data' in existing_tables,
                'user_behavior_analysis': 'user_behavior_analysis' in existing_tables,
                'keyword_analysis': 'keyword_analysis' in existing_tables,
                'data_quality_metrics': 'data_quality_metrics' in existing_tables
            }

            logger.info(f"Existing table check results: {table_status}")
            return table_status

        except Exception as e:
            logger.error(f"Failed to check existing tables: {e}")
            return {}

    def backup_existing_data(self, table_name: str, backup_suffix: str = None) -> bool:
        """Backup existing data"""
        try:
            if backup_suffix is None:
                backup_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')

            backup_table_name = f"{table_name}_backup_{backup_suffix}"

            # Check if table exists
            self.cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            if not self.cursor.fetchone():
                logger.warning(f"Table {table_name} does not exist, skipping backup")
                return True

            # Create backup table
            backup_sql = f"CREATE TABLE {backup_table_name} AS SELECT * FROM {table_name}"
            self.cursor.execute(backup_sql)

            # Get backup record count
            self.cursor.execute(f"SELECT COUNT(*) as count FROM {backup_table_name}")
            result = self.cursor.fetchone()
            backup_count = result['count'] if result else 0

            logger.info(f"Table {table_name} backup complete, backup table: {backup_table_name}, record count: {backup_count}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup table {table_name}: {e}")
            return False

    def migrate_csv_to_raw_data(self, csv_files: List[str], batch_size: int = 1000) -> bool:
        """Migrate CSV files to raw_data table"""
        try:
            logger.info("Starting CSV data migration")

            total_migrated = 0

            for csv_file in csv_files:
                if not os.path.exists(csv_file):
                    logger.warning(f"CSV file does not exist: {csv_file}")
                    continue

                logger.info(f"Migrating CSV file: {csv_file}")

                # Read CSV file
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(csv_file, encoding='gbk')
                    except Exception as e:
                        logger.error(f"Failed to read CSV file {csv_file}: {e}")
                        continue

                # Standardize column names
                column_mapping = {
                    '标题': 'title',
                    '内容': 'content',
                    '作者': 'author',
                    '名称': 'author',  # Some files might use "名称" for author
                    '发布时间': 'publish_time',
                    '点赞数': 'like_count',
                    '评论数': 'comment_count',
                    '分享数': 'share_count',
                    '浏览数': 'view_count',
                    '链接': 'url',
                    '平台': 'platform',
                    'count': 'like_count'  # Some files might use count for interaction count
                }

                # Rename columns
                df = df.rename(columns=column_mapping)

                # Add batch ID
                batch_id = f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(csv_file)}"
                df['batch_id'] = batch_id
                df['platform'] = df.get('platform', '小红书')

                # Insert in batches
                file_migrated = 0
                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i + batch_size]

                    # Prepare data for insertion
                    insert_data = []
                    for _, row in batch_df.iterrows():
                        # Use custom encoder for JSON serialization
                        raw_data_json = json.dumps(row.to_dict(), cls=DecimalEncoder, ensure_ascii=False, default=str)

                        raw_data = RawDataModel(
                            title=row.get('title'),
                            content=row.get('content'),
                            author=row.get('author'),
                            publish_time=self._parse_datetime(row.get('publish_time')),
                            like_count=str(row.get('like_count', '')),
                            comment_count=str(row.get('comment_count', '')),
                            share_count=str(row.get('share_count', '')),
                            view_count=str(row.get('view_count', '')),
                            url=row.get('url'),
                            platform=row.get('platform', '小红书'),
                            batch_id=batch_id,
                            raw_data=raw_data_json
                        )
                        insert_data.append(raw_data.to_dict())

                    # Execute batch insertion
                    if insert_data:
                        success = self._batch_insert_raw_data(insert_data)
                        if success:
                            file_migrated += len(insert_data)
                        else:
                            logger.error(f"Batch insert failed: {csv_file}, batch {i // batch_size + 1}")

                total_migrated += file_migrated
                logger.info(f"File {csv_file} migration complete, records: {file_migrated}")

            self.migration_stats['migrated_records'] += total_migrated
            self.migration_stats['migrated_tables'].append('raw_data')

            logger.info(f"CSV data migration complete, total records: {total_migrated}")
            return True

        except Exception as e:
            logger.error(f"CSV data migration failed: {e}")
            self.migration_stats['failed_migrations'].append(f'CSV migration: {str(e)}')
            return False

    def _parse_datetime(self, date_str) -> Optional[datetime]:
        """Parse datetime string"""
        if pd.isna(date_str) or not date_str:
            return None

        # Common date formats
        date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m-%d %H:%M',
            '%m/%d %H:%M'
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue

        logger.warning(f"Unable to parse date format: {date_str}")
        return None

    def _batch_insert_raw_data(self, data_list: List[Dict]) -> bool:
        """Batch insert raw data"""
        try:
            if not data_list:
                return True

            # Build insert SQL
            columns = list(data_list[0].keys())
            columns = [col for col in columns if col != 'id']  # Exclude auto-increment ID

            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)

            insert_sql = f"""
                INSERT INTO raw_data ({columns_str})
                VALUES ({placeholders})
            """

            # Prepare data
            values_list = []
            for data in data_list:
                values = [data.get(col) for col in columns]
                values_list.append(values)

            # Execute batch insert
            self.cursor.executemany(insert_sql, values_list)
            self.connection.commit()

            return True

        except Exception as e:
            logger.error(f"Failed to batch insert raw data: {e}")
            self.connection.rollback()
            return False

    def create_migration_report(self) -> Dict:
        """Create migration report"""
        try:
            self.migration_stats['end_time'] = datetime.now()

            # Calculate migration time
            if self.migration_stats['start_time']:
                duration = (self.migration_stats['end_time'] - self.migration_stats['start_time']).total_seconds()
                self.migration_stats['duration_seconds'] = duration

            # Get current database status
            table_counts = {}
            tables = ['raw_data', 'cleaned_data', 'user_behavior_analysis', 'keyword_analysis', 'data_quality_metrics']

            for table in tables:
                try:
                    self.cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    result = self.cursor.fetchone()
                    table_counts[table] = result['count'] if result else 0
                except Exception:
                    table_counts[table] = 0

            self.migration_stats['final_table_counts'] = table_counts

            # Save report - using custom encoder
            report_file = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.migration_stats, f, cls=DecimalEncoder, ensure_ascii=False, indent=2, default=str)

            logger.info(f"Migration report saved: {report_file}")
            return self.migration_stats

        except Exception as e:
            logger.error(f"Failed to create migration report: {e}")
            return self.migration_stats

    def run_complete_migration(self, csv_files: List[str] = None, backup_data: bool = True) -> bool:
        """Run complete migration process"""
        try:
            self.migration_stats['start_time'] = datetime.now()
            logger.info("Starting database migration process")

            # 1. Connect to database
            if not self.connect_to_database():
                return False

            # 2. Check existing tables
            table_status = self.check_existing_tables()

            # 3. Backup existing data (optional)
            if backup_data:
                for table_name, exists in table_status.items():
                    if exists:
                        self.backup_existing_data(table_name)

            # 4. Set up new database structure
            db_setup = DatabaseSetup(
                host=self.config['host'],
                user=self.config['user'],
                password=self.config['password'],
                port=int(self.config.get('port', 3306))
            )

            success = db_setup.setup_complete_database(self.config['database'])
            if not success:
                logger.error("Database structure setup failed")
                return False

            db_setup.close_connection()

            # Reconnect (since db_setup might have closed the connection)
            self.connect_to_database()

            # 5. Migrate CSV files (if provided)
            if csv_files:
                self.migrate_csv_to_raw_data(csv_files)

            # 6. Create migration report
            report = self.create_migration_report()

            logger.info("Database migration process complete")
            self.print_migration_summary()

            return len(self.migration_stats['failed_migrations']) == 0

        except Exception as e:
            logger.error(f"Database migration process failed: {e}")
            return False

        finally:
            self.close_connection()

    def print_migration_summary(self):
        """Print migration summary"""
        print("\n" + "=" * 60)
        print("🔄 Database Migration Execution Summary")
        print("=" * 60)

        if self.migration_stats.get('duration_seconds'):
            print(f"⏱️  Migration Time: {self.migration_stats['duration_seconds']:.2f} seconds")

        print(f"📊 Migrated Records: {self.migration_stats['migrated_records']}")

        if self.migration_stats['migrated_tables']:
            print(f"\n✅ Successfully Migrated Tables ({len(self.migration_stats['migrated_tables'])})：")
            for table in self.migration_stats['migrated_tables']:
                print(f"   - {table}")

        if self.migration_stats['failed_migrations']:
            print(f"\n❌ Failed Migrations ({len(self.migration_stats['failed_migrations'])})：")
            for failure in self.migration_stats['failed_migrations']:
                print(f"   - {failure}")

        if self.migration_stats.get('final_table_counts'):
            print(f"\n📈 Final Table Record Counts：")
            for table, count in self.migration_stats['final_table_counts'].items():
                print(f"   - {table}: {count}")

        print("=" * 60)


def find_csv_files_for_migration(directory: str) -> List[str]:
    """Find CSV files for migration"""
    csv_files = []

    if os.path.isfile(directory) and directory.endswith('.csv'):
        return [directory]

    if os.path.isdir(directory):
        # Find common data files
        target_files = [
            'zongshuju.csv',
            'cleaned_output_data.csv',
            '去重后的文件3.csv',
            'rednote.csv'
        ]

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv') and (file in target_files or 'data' in file.lower()):
                    csv_files.append(os.path.join(root, file))

    return csv_files


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Database Migration Tool')

    parser.add_argument('--config', '-c', default='X_data_section/database_config.json',
                        help='Path to database configuration file')
    parser.add_argument('--csv-dir', '-d', default='.',
                        help='Directory for CSV files')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip data backup')
    parser.add_argument('--csv-files', nargs='*',
                        help='Specify CSV files to migrate')

    args = parser.parse_args()

    try:
        # Load database configuration
        if os.path.exists(args.config):
            config = DatabaseConfig.load_config(args.config)
        else:
            config = DatabaseConfig.get_default_config()

        # Validate and fix configuration
        print("🔍 Validating database configuration...")
        print(f"Host: {config['host']}")
        print(f"User: '{config['user']}'")
        print(f"Database: {config['database']}")
        print(f"Port: {config['port']}")

        # If password is empty or connection fails, prompt user for input
        max_retries = 3
        for attempt in range(max_retries):
            if not config.get('password'):
                config['password'] = input(f"Please enter the password for MySQL user '{config['user']}': ")

            # Test connection
            print(f"🔗 Testing database connection (Attempt {attempt + 1}/{max_retries})...")
            test_migration = DatabaseMigration(config)
            if test_migration.connect_to_database():
                test_migration.close_connection()
                print("✅ Database connection test successful")
                break
            else:
                if attempt < max_retries - 1:
                    print("❌ Connection failed, please re-enter password")
                    config['password'] = ''  # Clear password to re-enter
                else:
                    print("❌ Multiple connection failures, please check MySQL configuration")
                    return 1

        # Find CSV files
        if args.csv_files:
            csv_files = args.csv_files
        else:
            csv_files = find_csv_files_for_migration(args.csv_dir)

        if csv_files:
            print(f"📁 Found {len(csv_files)} CSV files:")
            for csv_file in csv_files:
                print(f"   - {csv_file}")
        else:
            print("📁 No CSV files found, only table structure migration will be performed")

        # Execute migration
        migration = DatabaseMigration(config)
        success = migration.run_complete_migration(
            csv_files=csv_files,
            backup_data=not args.no_backup
        )

        if success:
            print("\n🎉 Database migration successful!")
            # Save valid configuration
            DatabaseConfig.save_config(config, args.config)
            return 0
        else:
            print("\n💥 Database migration failed!")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  User interrupted migration")
        return 1
    except Exception as e:
        print(f"\n💥 An error occurred during migration: {e}")
        logger.error(f"Main function execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
