import math

import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import jieba
import jieba.analyse
from snownlp import SnowNLP
import re
import os
import json
from datetime import datetime, timedelta
from collections import Counter
import logging
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Add custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime type"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)


class EnhancedDataProcessor:
    """Enhanced Data Processor, integrating data cleaning, deduplication, sentiment analysis, and MySQL storage"""

    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize data processor

        Args:
            db_config: Database configuration dictionary, containing host, user, password, database, etc.
        """
        self.db_config = db_config
        self.connection = None
        self.cursor = None

    def connect_to_database(self) -> bool:
        """Connect to database"""
        try:
            # Connect to database
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            logger.info("Successfully connected to MySQL database")

            # 检查cleaned_data表是否有interaction_metrics列，如果没有则添加
            try:
                self.cursor.execute("DESCRIBE cleaned_data")
                columns = [row[0] for row in self.cursor.fetchall()]

                if 'interaction_metrics' not in columns:
                    logger.info("向cleaned_data表添加interaction_metrics列")
                    try:
                        self.cursor.execute(
                            "ALTER TABLE cleaned_data ADD COLUMN interaction_metrics JSON COMMENT '互动指标JSON' AFTER keywords"
                        )
                        self.connection.commit()
                        logger.info("成功添加interaction_metrics列")
                    except Error as e:
                        logger.warning(f"添加列失败: {e}")
            except Error as e:
                logger.warning(f"检查cleaned_data表结构失败: {e}")

            return True
        except Error as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def check_table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        try:
            self.cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            return self.cursor.fetchone() is not None
        except Error as e:
            logger.error(f"Error checking if table {table_name} exists: {e}")
            return False

    def describe_table(self, table_name: str) -> List[Tuple]:
        """Get table structure"""
        try:
            self.cursor.execute(f"DESCRIBE {table_name}")
            return self.cursor.fetchall()
        except Error as e:
            logger.error(f"Error getting table {table_name} structure: {e}")
            return []

    def drop_table_if_exists(self, table_name: str):
        """Drop table (if exists)"""
        try:
            self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            logger.info(f"Table {table_name} dropped")
        except Error as e:
            logger.error(f"Error dropping table {table_name}: {e}")

    def recreate_tables(self):
        """Recreate all tables"""
        # Drop tables in foreign key dependency order
        tables_to_drop = ['cleaned_data', 'user_behavior_analysis', 'keyword_analysis', 'data_quality_metrics',
                          'raw_data']

        for table_name in tables_to_drop:
            self.drop_table_if_exists(table_name)

        # Recreate tables
        self.create_optimized_tables()

    def create_optimized_tables(self, force_recreate: bool = False):
        """Create optimized database table structure"""

        # If force recreate, drop all tables first
        if force_recreate:
            self.recreate_tables()
            return

        tables = {
            # Raw data table
            'raw_data': """
                CREATE TABLE IF NOT EXISTS raw_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(500) NOT NULL,
                    author VARCHAR(100) NOT NULL,
                    interaction_count INT DEFAULT 0,
                    publish_time DATETIME,
                    content_url VARCHAR(500),
                    raw_count_text VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_author (author),
                    INDEX idx_publish_time (publish_time),
                    INDEX idx_interaction_count (interaction_count)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """,

            # Cleaned data table
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

            # User behavior analysis table
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

            # Keyword analysis table
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

            # Data quality monitoring table
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
            """
        }

        for table_name, create_sql in tables.items():
            try:
                # Check if table exists
                if self.check_table_exists(table_name):
                    # Check if table structure is correct
                    table_structure = self.describe_table(table_name)
                    logger.info(f"Table {table_name} already exists, structure: {table_structure}")

                    # Validate if key columns exist
                    columns = [col[0] for col in table_structure]
                    required_columns = self.get_required_columns(table_name)

                    missing_columns = [col for col in required_columns if col not in columns]
                    if missing_columns:
                        logger.warning(f"Table {table_name} is missing columns: {missing_columns}")
                        logger.info(f"Recreating table {table_name}")
                        self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                        self.cursor.execute(create_sql)
                        logger.info(f"Table {table_name} recreated successfully")
                    else:
                        logger.info(f"Table {table_name} structure is correct")
                else:
                    # Create new table
                    self.cursor.execute(create_sql)
                    logger.info(f"Table {table_name} created successfully")

            except Error as e:
                logger.error(f"Error processing table {table_name}: {e}")
                # If creation fails, try to drop and recreate
                try:
                    self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    self.cursor.execute(create_sql)
                    logger.info(f"Table {table_name} recreated successfully")
                except Error as e2:
                    logger.error(f"Failed to recreate table {table_name}: {e2}")

    def get_required_columns(self, table_name: str) -> List[str]:
        """Get required columns for a table"""
        required_columns = {
            'raw_data': ['id', 'title', 'author', 'interaction_count', 'publish_time', 'content_url', 'raw_count_text',
                         'created_at'],
            'cleaned_data': ['id', 'raw_data_id', 'title', 'author', 'interaction_count', 'publish_time', 'content_url',
                             'title_length', 'special_char_count', 'sentiment_score', 'sentiment_label', 'keywords',
                             'processed_at'],
            'user_behavior_analysis': ['id', 'author', 'content_count', 'total_interactions', 'avg_interactions'],
            'keyword_analysis': ['id', 'keyword', 'frequency', 'total_interactions', 'avg_interactions'],
            'data_quality_metrics': ['id', 'batch_id', 'total_records', 'valid_records', 'duplicate_records']
        }
        return required_columns.get(table_name, [])

    def clean_interaction_count(self, count_text: str) -> int:
        """Clean interaction count data"""
        # if pd.isna(count_text) or count_text == '':
        #     return 0
        # try:
        #     if isinstance(count_text, float) and math.isnan(count_text):
        #         return 0
        # except:
        #     pass
        # count_text = str(count_text).strip()

        # Handle cases containing "赞" (like)
        if '赞' in count_text:
            return 0

        # Handle cases containing "万" (ten thousand)
        wan_match = re.match(r'(\d+(\.\d+)?)\s*万', count_text)
        if wan_match:
            try:
                return int(float(wan_match.group(1)) * 10000)
            except ValueError:
                return 0

        # Handle pure numbers
        number_match = re.match(r'\d+', count_text)
        if number_match:
            try:
                return int(number_match.group())
            except ValueError:
                return 0

        return 0

    def convert_timestamp_to_datetime(self, timestamp) -> datetime:
        """
        Convert timestamp to datetime object

        Args:
            timestamp: Timestamp in various formats

        Returns:
            datetime: Datetime object
        """
        if timestamp is None:
            return datetime.now()

        # 如果已经是datetime对象，直接返回
        if isinstance(timestamp, datetime):
            return timestamp

        # 如果是字符串，尝试多种格式转换
        if isinstance(timestamp, str):
            # 尝试多种日期格式
            formats = [
                '%Y-%m-%d %H:%M:%S',  # 2023-10-15 14:30:00
                '%Y/%m/%d %H:%M:%S',  # 2023/10/15 14:30:00
                '%Y-%m-%d %H:%M',  # 2023-10-15 14:30
                '%Y/%m/%d %H:%M',  # 2023/10/15 14:30
                '%Y-%m-%d',  # 2023-10-15
                '%Y/%m/%d',  # 2023/10/15
                '%m/%d/%Y %H:%M:%S',  # 10/15/2023 14:30:00
                '%m/%d/%Y',  # 10/15/2023
                '%d/%m/%Y',  # 15/10/2023
                '%Y年%m月%d日 %H:%M:%S',  # 中文格式
                '%Y年%m月%d日',  # 中文格式
            ]

            for date_format in formats:
                try:
                    return datetime.strptime(timestamp, date_format)
                except ValueError:
                    continue

            # 尝试使用pandas的日期解析（更灵活）
            try:
                return pd.to_datetime(timestamp).to_pydatetime()
            except:
                # 如果所有方法都失败，记录警告并返回当前时间
                logger.warning(f"无法解析日期时间字符串: {timestamp}，使用当前时间代替")
                return datetime.now()

        # 如果是数字（时间戳），尝试转换
        try:
            if isinstance(timestamp, (int, float)):
                # 假设是秒级时间戳
                if timestamp > 1e10:  # 毫秒级时间戳
                    return datetime.fromtimestamp(timestamp / 1000)
                else:  # 秒级时间戳
                    return datetime.fromtimestamp(timestamp)
        except:
            pass

        # 如果都失败了，返回当前时间
        logger.warning(f"无法解析时间戳: {timestamp}，使用当前时间代替")
        return datetime.now()

    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze text sentiment"""
        if not text or len(text) < 2:
            return 0.5, 'neutral'

        try:
            sentiment_score = SnowNLP(text).sentiments
            if sentiment_score >= 0.6:
                label = 'positive'
            elif sentiment_score <= 0.4:
                label = 'negative'
            else:
                label = 'neutral'
            return sentiment_score, label
        except:
            return 0.5, 'neutral'

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract keywords"""
        if not text or len(text) < 3:
            return []

        try:
            keywords = jieba.analyse.extract_tags(text, topK=top_k)
            return keywords
        except:
            return []

    def count_special_characters(self, text: str) -> int:
        """Count special characters in text"""
        if not text:
            return 0
        return len(re.findall(r'[^\w\s\u4e00-\u9fff]', text))

    def process_csv_data(self, csv_file_path: str, batch_size: int = 1000) -> str:
        """
        Process CSV data

        Args:
            csv_file_path: Path to the CSV file
            batch_size: Batch size for processing

        Returns:
            batch_id: Unique ID for this processing batch
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        try:
            # Read CSV file
            logger.info(f"Reading CSV file: {csv_file_path}")
            try:
                df = pd.read_csv(csv_file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file_path, encoding='gbk')
            logger.info(f"Successfully read CSV file, {len(df)} records in total")

            # 添加列名映射功能，处理不同格式的CSV文件
            # 定义可能的列名映射关系（常见的变体->标准名）
            column_mappings = {
                '标题': ['Title', 'title', '笔记标题', '内容标题'],
                '名称': ['Author ID', 'author', 'Author', 'author_id', '作者', '用户名', '昵称'],
                '发布时间': ['Publish Time', 'publish_time', 'PublishTime', 'time', '时间'],
                '链接': ['URL', 'url', 'link', 'Link', 'content_url', 'Content URL'],
                'count': ['Likes', 'likes', '点赞', '点赞数', 'likes_count'],
                'count_text': ['Likes Text', 'likes_text', '原始点赞']
            }

            # 检查并映射列名
            original_columns = df.columns.tolist()
            renamed = False

            for target_col, source_cols in column_mappings.items():
                if target_col not in df.columns:
                    # 尝试找到匹配的源列名
                    for source_col in source_cols:
                        if source_col in df.columns:
                            df[target_col] = df[source_col]
                            renamed = True
                            logger.info(f"Mapped column '{source_col}' to '{target_col}'")
                            break

            if renamed:
                logger.info(f"Renamed columns from {original_columns} to {df.columns.tolist()}")

            # 检查必要的列是否存在
            required_columns = ['标题', '名称']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Error processing CSV data: {missing_columns}")
                return None

            # Data quality statistics
            total_records = len(df)
            valid_records = 0
            duplicate_records = 0
            invalid_records = 0
            errors = []

            # Data cleaning and preprocessing
            df = df.dropna(subset=['标题', '名称'])  # Remove records with empty key fields

            # Deduplication
            original_count = len(df)
            df = df.drop_duplicates(subset=['标题', '名称'])
            duplicate_records = original_count - len(df)

            # Process count field
            if 'count' not in df.columns:
                if '点赞数' in df.columns:
                    df['count'] = df['点赞数']
                elif '点赞' in df.columns:
                    df['count'] = df['点赞']
                elif 'Likes' in df.columns:
                    df['count'] = df['Likes']
                elif 'likes' in df.columns:
                    df['count'] = df['likes']
                else:
                    df['count'] = 0

            if 'count_text' not in df.columns:
                df['count_text'] = df['count'].astype(str)

            # Process publish time
            if '发布时间' not in df.columns:
                if 'Publish Time' in df.columns:
                    df['发布时间'] = df['Publish Time']
                elif 'publish_time' in df.columns:
                    df['发布时间'] = df['publish_time']
                elif '时间' in df.columns:
                    df['发布时间'] = df['时间']
                else:
                    # Create random dates
                    np.random.seed(42)
                    now = datetime.now()
                    random_days = np.random.randint(1, 365, size=len(df))
                    df['发布时间'] = [(now - timedelta(days=int(days))).strftime('%Y-%m-%d %H:%M:%S') for days in
                                      random_days]

            # Add URL column if missing
            if '链接' not in df.columns:
                if 'URL' in df.columns:
                    df['链接'] = df['URL']
                elif 'url' in df.columns:
                    df['链接'] = df['url']
                elif 'content_url' in df.columns:
                    df['链接'] = df['content_url']
                else:
                    df['链接'] = ""

            # Process data in batches
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size].copy()

                # Process raw data
                raw_data_records = []
                for _, row in batch_df.iterrows():
                    try:
                        # Convert timestamp to datetime object
                        publish_time = self.convert_timestamp_to_datetime(row['发布时间'])

                        raw_record = (
                            row['标题'],
                            row['名称'],
                            self.clean_interaction_count(row.get('count', 0)),
                            publish_time,
                            row.get('链接', ''),
                            str(row.get('count_text', ''))
                        )
                        raw_data_records.append(raw_record)
                    except Exception as e:
                        invalid_records += 1
                        errors.append(f"Error processing raw data: {str(e)}")

                # Insert raw data
                if raw_data_records:
                    insert_raw_sql = """
                        INSERT INTO raw_data (title, author, interaction_count, publish_time, content_url, raw_count_text)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    self.cursor.executemany(insert_raw_sql, raw_data_records)

                    # Get inserted IDs
                    start_id = self.cursor.lastrowid - len(raw_data_records) + 1

                    # Process cleaned data
                    cleaned_data_records = []
                    for idx, (_, row) in enumerate(batch_df.iterrows()):
                        try:
                            title = row['标题']
                            sentiment_score, sentiment_label = self.analyze_sentiment(title)

                            # Extract keywords from title and content if available
                            keywords_sources = [title]
                            if "内容" in row and isinstance(row["内容"], str) and len(row["内容"]) > 0:
                                keywords_sources.append(row["内容"])

                            # Join text sources and extract keywords
                            keywords = self.extract_keywords(" ".join([s for s in keywords_sources if s]))

                            # Convert timestamp to datetime object
                            publish_time = self.convert_timestamp_to_datetime(row['发布时间'])

                            # Prepare additional metrics if available
                            likes = int(row.get("Likes", 0))
                            favorites = int(row.get("Favorites", 0))
                            comments = int(row.get("Comments", 0))

                            # Store metrics as JSON
                            metrics_json = json.dumps({
                                "likes": likes,
                                "favorites": favorites,
                                "comments": comments
                            }, ensure_ascii=False)

                            cleaned_record = (
                                start_id + idx,  # raw_data_id
                                title,
                                row['名称'],
                                self.clean_interaction_count(row.get('count', 0)),
                                publish_time,
                                row.get('链接', ''),
                                len(title),
                                self.count_special_characters(title),
                                sentiment_score,
                                sentiment_label,
                                json.dumps(keywords, ensure_ascii=False),
                                metrics_json  # Store additional metrics
                            )
                            cleaned_data_records.append(cleaned_record)
                            valid_records += 1
                        except Exception as e:
                            invalid_records += 1
                            errors.append(f"Error processing cleaned data: {str(e)}")

                    # Insert cleaned data - Check if interaction_metrics column exists
                    try:
                        insert_cleaned_sql = """
                            INSERT INTO cleaned_data (raw_data_id, title, author, interaction_count,
                                                    publish_time, content_url, title_length, special_char_count,
                                                    sentiment_score, sentiment_label, keywords, interaction_metrics)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                            interaction_count = VALUES(interaction_count),
                            sentiment_score = VALUES(sentiment_score),
                            sentiment_label = VALUES(sentiment_label),
                            keywords = VALUES(keywords),
                            interaction_metrics = VALUES(interaction_metrics)
                        """
                        self.cursor.executemany(insert_cleaned_sql, cleaned_data_records)
                    except Error as e:
                        # If interaction_metrics column doesn't exist, try without it
                        if "Unknown column 'interaction_metrics'" in str(e):
                            logger.warning("interaction_metrics column not found, using standard schema")
                            # Remove the last element (metrics_json) from each record
                            standard_cleaned_records = [record[:-1] for record in cleaned_data_records]
                            insert_cleaned_sql = """
                                INSERT INTO cleaned_data (raw_data_id, title, author, interaction_count,
                                                        publish_time, content_url, title_length, special_char_count,
                                                        sentiment_score, sentiment_label, keywords)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON DUPLICATE KEY UPDATE
                                interaction_count = VALUES(interaction_count),
                                sentiment_score = VALUES(sentiment_score),
                                sentiment_label = VALUES(sentiment_label),
                                keywords = VALUES(keywords)
                            """
                            self.cursor.executemany(insert_cleaned_sql, standard_cleaned_records)
                        else:
                            # Re-raise if it's another error
                            raise

                self.connection.commit()
                logger.info(f"Batch {i // batch_size + 1} processed")

            # Update user behavior analysis
            self.update_user_behavior_analysis()

            # Update keyword analysis
            self.update_keyword_analysis()

            # Record data quality metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            quality_record = (
                batch_id,
                total_records,
                valid_records,
                duplicate_records,
                invalid_records,
                processing_time,
                json.dumps(errors[:10], ensure_ascii=False)  # Only save the first 10 errors
            )

            quality_sql = """
                INSERT INTO data_quality_metrics (batch_id, total_records, valid_records,
                                                duplicate_records, invalid_records, processing_time_seconds, error_details)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            self.cursor.execute(quality_sql, quality_record)
            self.connection.commit()

            logger.info(f"Data processing complete - Batch ID: {batch_id}")
            logger.info(
                f"Total Records: {total_records}, Valid Records: {valid_records}, Duplicate Records: {duplicate_records}, Invalid Records: {invalid_records}")

            return batch_id

        except Exception as e:
            logger.error(f"Error processing CSV data: {e}")
            if self.connection:
                self.connection.rollback()
            raise

    def update_user_behavior_analysis(self):
        """Update user behavior analysis data"""
        try:
            # Calculate user statistics
            user_stats_sql = """
                SELECT
                    author,
                    COUNT(*) as content_count,
                    SUM(interaction_count) as total_interactions,
                    AVG(interaction_count) as avg_interactions,
                    MAX(interaction_count) as max_interactions,
                    STDDEV(interaction_count) as interaction_std,
                    AVG(sentiment_score) as avg_sentiment,
                    AVG(title_length) as avg_title_length,
                    AVG(special_char_count) as avg_special_chars,
                    MIN(publish_time) as first_post_date,
                    MAX(publish_time) as last_post_date,
                    DATEDIFF(MAX(publish_time), MIN(publish_time)) + 1 as active_days
                FROM cleaned_data
                GROUP BY author
            """

            self.cursor.execute(user_stats_sql)
            user_stats = self.cursor.fetchall()

            # Update user behavior analysis table
            for stats in user_stats:
                author, content_count, total_interactions, avg_interactions, max_interactions, \
                    interaction_std, avg_sentiment, avg_title_length, avg_special_chars, \
                    first_post_date, last_post_date, active_days = stats

                daily_avg_posts = content_count / max(active_days, 1)
                interaction_efficiency = avg_interactions / max(content_count, 1)

                # Get user keywords
                keyword_sql = """
                    SELECT keywords FROM cleaned_data WHERE author = %s
                """
                self.cursor.execute(keyword_sql, (author,))
                keyword_results = self.cursor.fetchall()

                all_keywords = []
                for result in keyword_results:
                    if result[0]:
                        try:
                            keywords = json.loads(result[0])
                            all_keywords.extend(keywords)
                        except:
                            pass

                top_keywords = [k for k, v in Counter(all_keywords).most_common(10)]

                # Insert or update user behavior analysis
                upsert_sql = """
                    INSERT INTO user_behavior_analysis
                    (author, content_count, total_interactions, avg_interactions, max_interactions,
                     interaction_std, active_days, daily_avg_posts, interaction_efficiency,
                     avg_sentiment, avg_title_length, avg_special_chars, top_keywords,
                     first_post_date, last_post_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    content_count = VALUES(content_count),
                    total_interactions = VALUES(total_interactions),
                    avg_interactions = VALUES(avg_interactions),
                    max_interactions = VALUES(max_interactions),
                    interaction_std = VALUES(interaction_std),
                    active_days = VALUES(active_days),
                    daily_avg_posts = VALUES(daily_avg_posts),
                    interaction_efficiency = VALUES(interaction_efficiency),
                    avg_sentiment = VALUES(avg_sentiment),
                    avg_title_length = VALUES(avg_title_length),
                    avg_special_chars = VALUES(avg_special_chars),
                    top_keywords = VALUES(top_keywords),
                    first_post_date = VALUES(first_post_date),
                    last_post_date = VALUES(last_post_date)
                """

                self.cursor.execute(upsert_sql, (
                    author, content_count, total_interactions or 0, avg_interactions or 0,
                    max_interactions or 0, interaction_std or 0, active_days, daily_avg_posts,
                    interaction_efficiency, avg_sentiment or 0.5, avg_title_length or 0,
                    avg_special_chars or 0, json.dumps(top_keywords, ensure_ascii=False),
                    first_post_date, last_post_date
                ))

            self.connection.commit()
            logger.info(f"User behavior analysis updated, processed {len(user_stats)} users")

        except Exception as e:
            logger.error(f"Error updating user behavior analysis: {e}")

    def update_keyword_analysis(self):
        """Update keyword analysis data"""
        try:
            # Get all keyword data
            keyword_sql = """
                SELECT keywords, interaction_count, sentiment_score, author
                FROM cleaned_data
                WHERE keywords IS NOT NULL AND keywords != '[]'
            """

            self.cursor.execute(keyword_sql)
            results = self.cursor.fetchall()

            keyword_stats = {}

            for result in results:
                keywords_json, interaction_count, sentiment_score, author = result
                try:
                    keywords = json.loads(keywords_json)
                    for keyword in keywords:
                        if keyword not in keyword_stats:
                            keyword_stats[keyword] = {
                                'frequency': 0,
                                'total_interactions': 0,
                                'authors': set(),
                                'sentiments': []
                            }

                        keyword_stats[keyword]['frequency'] += 1
                        keyword_stats[keyword]['total_interactions'] += interaction_count or 0
                        keyword_stats[keyword]['authors'].add(author)
                        keyword_stats[keyword]['sentiments'].append(sentiment_score or 0.5)
                except:
                    continue

            # Update keyword analysis table
            for keyword, stats in keyword_stats.items():
                avg_interactions = stats['total_interactions'] / max(stats['frequency'], 1)
                associated_authors = list(stats['authors'])[:20]  # Limit author count

                # Calculate sentiment distribution
                sentiments = stats['sentiments']
                sentiment_dist = {
                    'positive': len([s for s in sentiments if s >= 0.6]),
                    'neutral': len([s for s in sentiments if 0.4 < s < 0.6]),
                    'negative': len([s for s in sentiments if s <= 0.4])
                }

                upsert_sql = """
                    INSERT INTO keyword_analysis
                    (keyword, frequency, total_interactions, avg_interactions,
                     associated_authors, sentiment_distribution)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    frequency = VALUES(frequency),
                    total_interactions = VALUES(total_interactions),
                    avg_interactions = VALUES(avg_interactions),
                    associated_authors = VALUES(associated_authors),
                    sentiment_distribution = VALUES(sentiment_distribution)
                """

                self.cursor.execute(upsert_sql, (
                    keyword,
                    stats['frequency'],
                    stats['total_interactions'],
                    avg_interactions,
                    json.dumps(associated_authors, ensure_ascii=False),
                    json.dumps(sentiment_dist, ensure_ascii=False)
                ))

            self.connection.commit()
            logger.info(f"Keyword analysis updated, processed {len(keyword_stats)} keywords")

        except Exception as e:
            logger.error(f"Error updating keyword analysis: {e}")

    def get_data_quality_report(self, batch_id: Optional[str] = None) -> Dict:
        """Get data quality report"""
        try:
            if batch_id:
                sql = "SELECT * FROM data_quality_metrics WHERE batch_id = %s"
                self.cursor.execute(sql, (batch_id,))
            else:
                sql = "SELECT * FROM data_quality_metrics ORDER BY created_at DESC LIMIT 10"
                self.cursor.execute(sql)

            results = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]

            reports = []
            for result in results:
                report = dict(zip(columns, result))

                # Preprocess datetime objects, convert to string
                for key, value in report.items():
                    if isinstance(value, datetime):
                        report[key] = value.isoformat()

                if report['error_details']:
                    try:
                        report['error_details'] = json.loads(report['error_details'])
                    except:
                        pass
                reports.append(report)

            return {'reports': reports}

        except Exception as e:
            logger.error(f"Error getting data quality report: {e}")
            return {'reports': []}

    def close_connection(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed")

    def process_xiaohongshu_notes_csv(self, csv_file_path: str, batch_size: int = 1000) -> str:
        """
        Process Xiaohongshu notes CSV data with specific columns:
        Author ID, Title, Content, Publish Time, Likes, Favorites, Comments

        Args:
            csv_file_path: Path to the CSV file
            batch_size: Batch size for processing

        Returns:
            batch_id: Unique ID for this processing batch
        """
        batch_id = f"xhs_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        try:
            # Read CSV file
            logger.info(f"Reading Xiaohongshu notes CSV file: {csv_file_path}")
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            logger.info(f"Successfully read CSV file, {len(df)} records in total")

            # Check required columns
            required_columns = ["Author ID", "Title", "Publish Time"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

            # Data quality statistics
            total_records = len(df)
            valid_records = 0
            duplicate_records = 0
            invalid_records = 0
            errors = []

            # Data cleaning and preprocessing
            # Rename columns to match the expected format for internal processing
            column_mapping = {
                "Author ID": "名称",
                "Title": "标题",
                "Content": "内容",
                "Publish Time": "发布时间"
            }

            # Apply column mapping
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = df[old_col]

            # Process interaction metrics (Likes, Favorites, Comments)
            if all(col in df.columns for col in ["Likes", "Favorites", "Comments"]):
                # Convert to numeric values, handling NaN values
                df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce").fillna(0)
                df["Favorites"] = pd.to_numeric(df["Favorites"], errors="coerce").fillna(0)
                df["Comments"] = pd.to_numeric(df["Comments"], errors="coerce").fillna(0)

                # Option 1: Use Likes as the primary interaction count
                df["count"] = df["Likes"]

                # Option 2: Sum all interactions (uncomment to use)
                # df["count"] = df["Likes"] + df["Favorites"] + df["Comments"]

                # Store the raw count as string
                df["count_text"] = df["Likes"].astype(str)
            else:
                # Default count if metrics are not available
                df["count"] = 0
                df["count_text"] = "0"

            # Add URL column if missing
            if "链接" not in df.columns:
                df["链接"] = ""

            # Data cleaning
            df = df.dropna(subset=['标题', '名称'])  # Remove records with empty key fields

            # Deduplication
            original_count = len(df)
            df = df.drop_duplicates(subset=['标题', '名称'])
            duplicate_records = original_count - len(df)

            # Process publish time
            if "发布时间" in df.columns:
                # Convert publish time to datetime objects
                df['发布时间'] = pd.to_datetime(df['发布时间'], errors='coerce')
            else:
                # Create random dates if missing
                np.random.seed(42)
                now = datetime.now()
                random_days = np.random.randint(1, 365, size=len(df))
                df['发布时间'] = [(now - timedelta(days=int(days))) for days in random_days]

            # Process data in batches
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size].copy()

                # Process raw data
                raw_data_records = []
                for _, row in batch_df.iterrows():
                    try:
                        # Convert timestamp to datetime object
                        publish_time = self.convert_timestamp_to_datetime(row['发布时间'])

                        raw_record = (
                            row['标题'],
                            row['名称'],
                            self.clean_interaction_count(row.get('count', 0)),
                            publish_time,
                            row.get('链接', ''),
                            str(row.get('count_text', ''))
                        )
                        raw_data_records.append(raw_record)
                    except Exception as e:
                        invalid_records += 1
                        errors.append(f"Error processing raw data: {str(e)}")

                # Insert raw data
                if raw_data_records:
                    insert_raw_sql = """
                        INSERT INTO raw_data (title, author, interaction_count, publish_time, content_url, raw_count_text)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    self.cursor.executemany(insert_raw_sql, raw_data_records)

                    # Get inserted IDs
                    start_id = self.cursor.lastrowid - len(raw_data_records) + 1

                    # Process cleaned data
                    cleaned_data_records = []
                    for idx, (_, row) in enumerate(batch_df.iterrows()):
                        try:
                            title = row['标题']
                            sentiment_score, sentiment_label = self.analyze_sentiment(title)

                            # Extract keywords from title and content if available
                            keywords_sources = [title]
                            if "内容" in row and isinstance(row["内容"], str) and len(row["内容"]) > 0:
                                keywords_sources.append(row["内容"])

                            # Join text sources and extract keywords
                            keywords = self.extract_keywords(" ".join([s for s in keywords_sources if s]))

                            # Convert timestamp to datetime object
                            publish_time = self.convert_timestamp_to_datetime(row['发布时间'])

                            # Prepare additional metrics if available
                            likes = int(row.get("Likes", 0))
                            favorites = int(row.get("Favorites", 0))
                            comments = int(row.get("Comments", 0))

                            # Store metrics as JSON
                            metrics_json = json.dumps({
                                "likes": likes,
                                "favorites": favorites,
                                "comments": comments
                            }, ensure_ascii=False)

                            cleaned_record = (
                                start_id + idx,  # raw_data_id
                                title,
                                row['名称'],
                                self.clean_interaction_count(row.get('count', 0)),
                                publish_time,
                                row.get('链接', ''),
                                len(title),
                                self.count_special_characters(title),
                                sentiment_score,
                                sentiment_label,
                                json.dumps(keywords, ensure_ascii=False),
                                metrics_json  # Store additional metrics
                            )
                            cleaned_data_records.append(cleaned_record)
                            valid_records += 1
                        except Exception as e:
                            invalid_records += 1
                            errors.append(f"Error processing cleaned data: {str(e)}")

                    # Insert cleaned data - Check if interaction_metrics column exists
                    try:
                        insert_cleaned_sql = """
                            INSERT INTO cleaned_data (raw_data_id, title, author, interaction_count,
                                                    publish_time, content_url, title_length, special_char_count,
                                                    sentiment_score, sentiment_label, keywords, interaction_metrics)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                            interaction_count = VALUES(interaction_count),
                            sentiment_score = VALUES(sentiment_score),
                            sentiment_label = VALUES(sentiment_label),
                            keywords = VALUES(keywords),
                            interaction_metrics = VALUES(interaction_metrics)
                        """
                        self.cursor.executemany(insert_cleaned_sql, cleaned_data_records)
                    except Error as e:
                        # If interaction_metrics column doesn't exist, try without it
                        if "Unknown column 'interaction_metrics'" in str(e):
                            logger.warning("interaction_metrics column not found, using standard schema")
                            # Remove the last element (metrics_json) from each record
                            standard_cleaned_records = [record[:-1] for record in cleaned_data_records]
                            insert_cleaned_sql = """
                                INSERT INTO cleaned_data (raw_data_id, title, author, interaction_count,
                                                        publish_time, content_url, title_length, special_char_count,
                                                        sentiment_score, sentiment_label, keywords)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON DUPLICATE KEY UPDATE
                                interaction_count = VALUES(interaction_count),
                                sentiment_score = VALUES(sentiment_score),
                                sentiment_label = VALUES(sentiment_label),
                                keywords = VALUES(keywords)
                            """
                            self.cursor.executemany(insert_cleaned_sql, standard_cleaned_records)
                        else:
                            # Re-raise if it's another error
                            raise

                self.connection.commit()
                logger.info(f"Batch {i // batch_size + 1} processed")

            # Update user behavior analysis
            self.update_user_behavior_analysis()

            # Update keyword analysis
            self.update_keyword_analysis()

            # Record data quality metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            quality_record = (
                batch_id,
                total_records,
                valid_records,
                duplicate_records,
                invalid_records,
                processing_time,
                json.dumps(errors[:10], ensure_ascii=False)  # Only save the first 10 errors
            )

            quality_sql = """
                INSERT INTO data_quality_metrics (batch_id, total_records, valid_records,
                                                duplicate_records, invalid_records, processing_time_seconds, error_details)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            self.cursor.execute(quality_sql, quality_record)
            self.connection.commit()

            logger.info(f"Xiaohongshu notes data processing complete - Batch ID: {batch_id}")
            logger.info(
                f"Total Records: {total_records}, Valid Records: {valid_records}, Duplicate Records: {duplicate_records}, Invalid Records: {invalid_records}")

            return batch_id

        except Exception as e:
            logger.error(f"Error processing Xiaohongshu notes CSV data: {e}")
            if self.connection:
                self.connection.rollback()
            raise


# Example usage
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': 'root',
        'database': 'rednote',
        'charset': 'utf8mb4'
    }

    # Create data processor
    processor = EnhancedDataProcessor(db_config)

    try:
        # Connect to database
        if processor.connect_to_database():
            # Create tables (if there are structure issues, you can set force_recreate=True)
            processor.create_optimized_tables(force_recreate=False)

            # Process CSV data
            # 这里需要根据实际的爬虫输出文件路径进行修改
            csv_file = r"D:\python\单子\X_crawler\X_rednote_crawler\data\output.csv"
            if os.path.exists(csv_file):
                batch_id = processor.process_csv_data(csv_file)
                print(f"Data processing complete, Batch ID: {batch_id}")

                # Get data quality report
                report = processor.get_data_quality_report(batch_id)
                # Use custom encoder to handle datetime objects
                print("Data Quality Report:", json.dumps(report, indent=2, ensure_ascii=False, cls=DateTimeEncoder))
            else:
                print(f"CSV file {csv_file} does not exist")

    finally:
        processor.close_connection()
