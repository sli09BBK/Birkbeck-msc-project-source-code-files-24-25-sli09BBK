from kafka import KafkaProducer
import json
import os

class XiaohongshuKafkaProducer:
    def __init__(self, bootstrap_servers=None, topic=None, config_path='X_Rednote_Kafka/kafka_config.json'):
        # Priority: parameters > json config > default
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        self.topic = topic or config.get('topic', 'raw_data')
        bs = bootstrap_servers or config.get('bootstrap_servers', 'localhost:9092')
        self.producer = KafkaProducer(
            bootstrap_servers=bs,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8')
        )

    def send_data(self, data: dict):
        try:
            future = self.producer.send(self.topic, value=data)
            record_metadata = future.get(timeout=10)
            print(f"Message sent successfully: topic={record_metadata.topic}, partition={record_metadata.partition}, offset={record_metadata.offset}")
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False

# Example usage
if __name__ == '__main__':
    producer = XiaohongshuKafkaProducer()
    sample_data = {
        'author': 'test',
        'title': 'test title',
        'content': 'test content',
        'publish_time': '2024-07-06',
        'likes': 123,
        'favorites': 45,
        'comments': 6
    }
    success = producer.send_data(sample_data)
    print('Send result:', success) 