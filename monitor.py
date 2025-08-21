import os
import time
import logging
import sqlite3
from datetime import datetime
import multiprocessing
from predict11 import process_single_video

class VideoMonitor:
    def __init__(self, source_directory):
        """Initialize the video monitor with multiprocessing support"""
        self.source_directory = source_directory
        self.db_path = "processed_file.db"
        
        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('video_processing.log'),
                    logging.StreamHandler()
                ]
            )
        self.logger.info(f"Initializing VideoMonitor for directory: {source_directory}")
        
        # Initialize database
        self.init_tracking_db()

    def init_tracking_db(self):
        """Initialize SQLite database with concurrency support"""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS processed_files (
                        file_name TEXT PRIMARY KEY,
                        file_path TEXT UNIQUE,
                        queued_date TIMESTAMP,
                        process_start TIMESTAMP,
                        process_end TIMESTAMP,
                        status TEXT CHECK(status IN ('queued', 'processing', 'success', 'failed')),
                        error_message TEXT
                    )
                ''')
                conn.commit()
                self.logger.info("Database initialized with concurrency support")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise

    def scan_for_new_files(self):
        """Scan directory for new files and add to queue"""
        try:
            current_files = {
                os.path.join(self.source_directory, f)
                for f in os.listdir(self.source_directory)
                if f.lower().endswith(('.mp4', '.avi', '.mov'))
            }

            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT file_path FROM processed_files")
                existing_files = {row[0] for row in cursor.fetchall()}

                new_files = current_files - existing_files
                for file_path in new_files:
                    file_name = os.path.basename(file_path)
                    try:
                        cursor.execute('''
                            INSERT INTO processed_files 
                            (file_name, file_path, queued_date, status)
                            VALUES (?, ?, ?, 'queued')
                            ON CONFLICT(file_name) DO NOTHING
                        ''', (file_name, file_path, datetime.now()))
                        self.logger.info(f"Queued new file: {file_name}")
                    except sqlite3.IntegrityError:
                        continue
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error scanning files: {str(e)}")

    def claim_next_file(self):
        """Atomically claim the next available file for processing"""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                
                # Find and claim the oldest queued file
                cursor.execute('''
                    UPDATE processed_files
                    SET status = 'processing', process_start = ?
                    WHERE file_name = (
                        SELECT file_name FROM processed_files
                        WHERE status = 'queued'
                        ORDER BY queued_date ASC
                        LIMIT 1
                    )
                    RETURNING file_path
                ''', (datetime.now(),))
                
                result = cursor.fetchone()
                conn.commit()
                return result[0] if result else None
        except sqlite3.OperationalError as e:
            self.logger.warning(f"Database busy: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Claim error: {str(e)}")
            return None

    def update_file_status(self, file_path, status, error=None):
        """Update processing status in database"""
        file_name = os.path.basename(file_path)
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE processed_files
                    SET status = ?,
                        process_end = ?,
                        error_message = ?
                    WHERE file_name = ?
                ''', (status, datetime.now(), error, file_name))
                conn.commit()
                self.logger.info(f"Updated {file_name} to {status}")
        except Exception as e:
            self.logger.error(f"Status update failed: {str(e)}")

    def worker_loop(self):
        """Worker process main loop"""
        self.logger.info("Worker started")
        while True:
            file_path = self.claim_next_file()
            if file_path:
                try:
                    self.logger.info(f"Processing started: {file_path}")
                    success = process_single_video(file_path)
                    status = 'success' if success else 'failed'
                    error = None if success else "Processing failed"
                    self.update_file_status(file_path, status, error)
                except Exception as e:
                    self.logger.error(f"Processing error: {str(e)}")
                    self.update_file_status(file_path, 'failed', str(e))
            else:
                time.sleep(5)

    def directory_watcher_loop(self):
        """Directory watcher main loop"""
        self.logger.info("Directory watcher started")
        while True:
            self.scan_for_new_files()
            time.sleep(60)

def main():
    # Configuration
    SOURCE_DIR = "/home/newFeed"
    WORKERS = 4  # Number of parallel workers
    
    # Initialize monitor
    monitor = VideoMonitor(SOURCE_DIR)
    
    # Start directory watcher in separate process
    watcher = multiprocessing.Process(target=monitor.directory_watcher_loop)
    watcher.start()
    
    # Start worker processes
    workers = []
    for _ in range(WORKERS):
        worker = multiprocessing.Process(target=monitor.worker_loop)
        worker.start()
        workers.append(worker)
    
    # Handle termination
    try:
        watcher.join()
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        watcher.terminate()
        for worker in workers:
            worker.terminate()

if __name__ == "__main__":
    main()
