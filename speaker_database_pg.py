"""
PostgreSQL Speaker Database Manager
Handles storing and retrieving speaker embeddings in PostgreSQL
"""

import psycopg2
import numpy as np
from datetime import datetime
import pickle

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "mydb", 
    "user": "postgres",
    "password": "Pass,822",
    "port": 5432
}

class SpeakerDatabasePG:
    def __init__(self):
        """Initialize the PostgreSQL speaker database connection"""
        self.db_config = DB_CONFIG
        self.test_connection()
    
    def test_connection(self):
        """Test database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            print("✓ Connected to PostgreSQL database")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            print("Please run setup_database.py first or check your credentials")
            raise
    
    def get_connection(self):
        """Get a database connection"""
        return psycopg2.connect(**self.db_config)
    
    def add_speaker(self, speaker_code, display_name, embedding, audio_sample_path):
        """
        Add a new speaker to the database
        
        Args:
            speaker_code: Unique identifier for the speaker
            display_name: Speaker's name
            embedding: Speaker's voice embedding (numpy array)
            audio_sample_path: Path to the audio sample used for registration
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Serialize embedding using pickle
            embedding_bytes = pickle.dumps(embedding)
            
            # Check if speaker already exists
            cur.execute(
                "SELECT id FROM speakers WHERE speaker_code = %s",
                (speaker_code,)
            )
            exists = cur.fetchone()
            
            if exists:
                # Update existing speaker
                cur.execute("""
                    UPDATE speakers 
                    SET display_name = %s, 
                        embedding = %s, 
                        audio_samples = array_append(audio_samples, %s),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE speaker_code = %s
                """, (display_name, embedding_bytes, str(audio_sample_path), speaker_code))
                print(f"✓ Speaker '{display_name}' (Code: {speaker_code}) updated successfully!")
            else:
                # Insert new speaker
                cur.execute("""
                    INSERT INTO speakers (speaker_code, display_name, embedding, audio_samples)
                    VALUES (%s, %s, %s, ARRAY[%s])
                """, (speaker_code, display_name, embedding_bytes, str(audio_sample_path)))
                print(f"✓ Speaker '{display_name}' (Code: {speaker_code}) registered successfully!")
            
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding speaker: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    def get_speaker(self, speaker_code):
        """
        Get speaker information by code
        
        Args:
            speaker_code: The speaker's unique identifier
            
        Returns:
            dict: Speaker information or None if not found
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT speaker_code, display_name, embedding, audio_samples, 
                       created_at, updated_at, active
                FROM speakers 
                WHERE speaker_code = %s AND active = TRUE
            """, (speaker_code,))
            
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            if row:
                return {
                    'speaker_code': row[0],
                    'name': row[1],
                    'embedding': pickle.loads(row[2]),
                    'audio_samples': row[3],
                    'created_at': row[4],
                    'updated_at': row[5],
                    'active': row[6]
                }
            return None
        except Exception as e:
            print(f"Error getting speaker: {e}")
            return None
    
    def get_all_speakers(self):
        """
        Get all active speakers in the database
        
        Returns:
            dict: All speakers with their information
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT speaker_code, display_name, embedding, audio_samples, 
                       created_at, updated_at, active
                FROM speakers 
                WHERE active = TRUE
                ORDER BY display_name
            """)
            
            speakers = {}
            for row in cur.fetchall():
                speakers[row[0]] = {
                    'name': row[1],
                    'embedding': pickle.loads(row[2]),
                    'audio_samples': row[3],
                    'created_at': row[4].strftime("%Y-%m-%d %H:%M:%S") if row[4] else None,
                    'updated_at': row[5].strftime("%Y-%m-%d %H:%M:%S") if row[5] else None,
                    'active': row[6]
                }
            
            cur.close()
            conn.close()
            
            print(f"✓ Loaded {len(speakers)} speakers from database")
            return speakers
        except Exception as e:
            print(f"Error getting all speakers: {e}")
            return {}
    
    def remove_speaker(self, speaker_code):
        """
        Remove a speaker from the database (soft delete)
        
        Args:
            speaker_code: The speaker's unique identifier
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Get speaker name before deleting
            cur.execute(
                "SELECT display_name FROM speakers WHERE speaker_code = %s",
                (speaker_code,)
            )
            result = cur.fetchone()
            
            if result:
                name = result[0]
                # Soft delete - set active to false
                cur.execute("""
                    UPDATE speakers 
                    SET active = FALSE, updated_at = CURRENT_TIMESTAMP
                    WHERE speaker_code = %s
                """, (speaker_code,))
                conn.commit()
                print(f"✓ Speaker '{name}' (Code: {speaker_code}) removed from database")
                cur.close()
                conn.close()
                return True
            else:
                print(f"✗ Speaker code '{speaker_code}' not found in database")
                cur.close()
                conn.close()
                return False
        except Exception as e:
            print(f"Error removing speaker: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    def list_speakers(self):
        """Print all active speakers in the database"""
        speakers = self.get_all_speakers()
        
        if not speakers:
            print("No speakers in database")
            return
        
        print("\n" + "="*80)
        print(f"{'Code':<20} {'Name':<25} {'Registered':<25} {'Samples'}")
        print("="*80)
        for code, info in speakers.items():
            sample_count = len(info['audio_samples']) if info['audio_samples'] else 0
            print(f"{code:<20} {info['name']:<25} {info['created_at']:<25} {sample_count}")
        print("="*80)
        print(f"Total speakers: {len(speakers)}\n")
    
    def search_by_embedding(self, test_embedding, threshold=0.75):
        """
        Search for the best matching speaker by embedding similarity
        
        Args:
            test_embedding: The embedding to match against
            threshold: Minimum similarity threshold
            
        Returns:
            tuple: (speaker_code, speaker_name, similarity_score) or (None, None, best_score)
        """
        speakers = self.get_all_speakers()
        
        best_match_code = None
        best_match_name = None
        best_score = 0
        
        for code, info in speakers.items():
            similarity = np.dot(test_embedding, info['embedding']) / (
                np.linalg.norm(test_embedding) * np.linalg.norm(info['embedding'])
            )
            
            if similarity > best_score:
                best_score = similarity
                best_match_code = code
                best_match_name = info['name']
        
        if best_score >= threshold:
            return best_match_code, best_match_name, best_score
        else:
            return None, None, best_score
    
    def get_speaker_count(self):
        """Get count of active speakers"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM speakers WHERE active = TRUE")
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            return count
        except Exception as e:
            print(f"Error getting speaker count: {e}")
            return 0


if __name__ == "__main__":
    # Test the database
    try:
        db = SpeakerDatabasePG()
        print(f"\nTotal speakers in database: {db.get_speaker_count()}")
        db.list_speakers()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease run: python setup_database.py first")
