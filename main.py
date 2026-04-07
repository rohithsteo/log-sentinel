import os
import time
import sqlite3
from datetime import datetime
from sklearn.ensemble import IsolationForest
import numpy as np

DB = "logs.db"
LOG_FILE = "app.log"

class LogAnomalyDetector:
    def __init__(self):
        self.conn = sqlite3.connect(DB)
        self.create_table()
        self.model = IsolationForest(contamination=0.1)

    def create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            level TEXT,
            message TEXT
        )
        """)
        self.conn.commit()

    def parse_log(self, line):
        try:
            parts = line.strip().split(" | ")
            return parts[0], parts[1], parts[2]
        except:
            return None, None, None

    def store_log(self, timestamp, level, message):
        self.conn.execute(
            "INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)",
            (timestamp, level, message)
        )
        self.conn.commit()

    def fetch_messages(self):
        cursor = self.conn.execute("SELECT message FROM logs")
        return [row[0] for row in cursor]

    def vectorize(self, messages):
        return np.array([[len(m), m.count("error"), m.count("fail")] for m in messages])

    def train(self):
        messages = self.fetch_messages()
        if len(messages) < 10:
            return False
        X = self.vectorize(messages)
        self.model.fit(X)
        return True

    def detect(self):
        messages = self.fetch_messages()
        X = self.vectorize(messages)
        preds = self.model.predict(X)

        anomalies = []
        cursor = self.conn.execute("SELECT * FROM logs")
        rows = cursor.fetchall()

        for i, p in enumerate(preds):
            if p == -1:
                anomalies.append(rows[i])
        return anomalies

    def monitor(self):
        print("Monitoring logs... Press Ctrl+C to stop")

        with open(LOG_FILE, "r") as f:
            f.seek(0, os.SEEK_END)

            while True:
                line = f.readline()

                if not line:
                    time.sleep(1)
                    continue

                timestamp, level, message = self.parse_log(line)

                if timestamp:
                    self.store_log(timestamp, level, message)

                if self.train():
                    anomalies = self.detect()

                    if anomalies:
                        print("\n🚨 Anomalies Detected:")
                        for a in anomalies[-3:]:
                            print(a)

if __name__ == "__main__":
    detector = LogAnomalyDetector()
    detector.monitor()
