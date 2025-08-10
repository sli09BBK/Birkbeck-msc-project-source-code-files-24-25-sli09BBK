import json
import logging
import os
import sys

from kafka import KafkaConsumer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from X_data_section.enhanced_data_processor import EnhancedDataProcessor

# Load database configuration
def load_db_config(config_path='E:\X_crawler\X_crawler\X_data_section\database_config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load Kafka configuration
def load_kafka_config(config_path='E:\X_crawler\X_crawler\X_data_section\database_config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# Logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Kafka Consumer configuration
# This module consumes data collected from Kafka topics.
# It performs data cleaning, transformation, and stores the processed data into the database.
# The pipeline ensures that only standardized and high-quality data is persisted for downstream analysis.

def main():
    db_config = load_db_config()
    kafka_config = load_kafka_config()
    processor = EnhancedDataProcessor(db_config)
    if not processor.connect_to_database():
        logger.error('Database connection failed, exiting.')
        return

    topic = kafka_config.get('topic', 'raw_data')
    bs = kafka_config.get('bootstrap_servers', 'localhost:9092')
    group_id = kafka_config.get('group_id', 'xiaohongshu_data_group')
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bs,
        group_id=group_id,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True
    )
    logger.info('Kafka Consumer started, waiting for data...')

    try:
        for message in consumer:
            data = message.value
            try:
                # Field mapping, compatible with original CSV logic
                row = {
                    'title': data.get('title', ''),
                    'author': data.get('author', ''),
                    'content': data.get('content', ''),
                    'publish_time': data.get('publish_time', ''),
                    'count': data.get('likes', 0),
                    'url': data.get('url', '')
                }
                publish_time = processor.convert_timestamp_to_datetime(row['publish_time'])
                raw_record = (
                    row['title'],
                    row['author'],
                    processor.clean_interaction_count(row.get('count', 0)),
                    publish_time,
                    row.get('url', ''),
                    str(row.get('count', ''))
                )
                insert_raw_sql = """
                    INSERT INTO raw_data (title, author, interaction_count, publish_time, content_url, raw_count_text)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                processor.cursor.execute(insert_raw_sql, raw_record)
                raw_data_id = processor.cursor.lastrowid

                # Assemble interaction_metrics field
                interaction_metrics = {
                    "likes": data.get("likes", 0),
                    "comments": data.get("comments", 0),
                    "favorites": data.get("favorites", 0)
                }
                metrics_json = json.dumps(interaction_metrics, ensure_ascii=False)

                # Insert into cleaned_data after cleaning
                title = row['title']
                sentiment_score, sentiment_label = processor.analyze_sentiment(title)
                keywords = processor.extract_keywords(title)
                cleaned_record = (
                    raw_data_id,
                    title,
                    row['author'],
                    processor.clean_interaction_count(row.get('count', 0)),
                    publish_time,
                    row.get('url', ''),
                    len(title),
                    processor.count_special_characters(title),
                    sentiment_score,
                    sentiment_label,
                    json.dumps(keywords, ensure_ascii=False),
                    metrics_json
                )
                # Insert SQL with interaction_metrics field
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
                try:
                    processor.cursor.execute(insert_cleaned_sql, cleaned_record)
                except Exception as e:
                    # If the table does not have the interaction_metrics field, degrade to standard schema
                    if "Unknown column 'interaction_metrics'" in str(e):
                        logger.warning("interaction_metrics column not found, using standard schema")
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
                        processor.cursor.execute(insert_cleaned_sql, cleaned_record[:-1])
                    else:
                        raise
                processor.connection.commit()
                logger.info(f"Successfully consumed and stored: {title}")

                # Real-time update of analysis tables
                processor.update_user_behavior_analysis()
                processor.update_keyword_analysis()
                logger.info("Analysis tables updated in real time")
            except Exception as e:
                logger.error(f"Data processing or storage failed: {e}, data: {data}")
                processor.connection.rollback()
    except KeyboardInterrupt:
        logger.info('Interrupt signal received, exiting safely...')
    finally:
        processor.close_connection()
        consumer.close()



if __name__ == '__main__':
    main() 
 