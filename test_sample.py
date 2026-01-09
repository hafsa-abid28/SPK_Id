"""
Simple test script - just edit the filenames below and run!
Tests speaker registration and verification with your audio files.
"""
import sys
from pathlib import Path

# Import our speaker recognition system
from speaker_database_pg import SpeakerDatabasePG
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import soundfile as sf
import librosa
import gc

# Initialize
encoder = VoiceEncoder(device='cpu')
db = SpeakerDatabasePG()

# ============================================================
# EDIT THESE SETTINGS
# ============================================================

# Speakers to register (audio_file, speaker_code, display_name)
SPEAKERS_TO_REGISTER = [
    ("sample.mp3", "john", "John"),
    ("sample2.mp3", "jane", "Jane"),
]

# Audio files to test/identify
AUDIO_FILES_TO_TEST = [
    "sample.mp3",
    "sample3.mp3",
]

# Verification threshold (0.75 = 75% match required)
THRESHOLD = 0.75


def register_speaker_simple(audio_path, speaker_code, display_name):
    """Register a speaker with automatic optimization."""
    try:
        if not Path(audio_path).exists():
            print(f"   ✗ File not found: {audio_path}")
            return False
        
        # Check duration
        info = sf.info(audio_path)
        duration = info.duration
        
        print(f"   Audio: {duration:.1f}s ({duration/60:.1f} min)")
        
        # Auto-optimize for long audio (>5 minutes)
        if duration > 300:
            print("   Using optimized processing...")
            wav, _ = librosa.load(audio_path, sr=16000, duration=300)
            
            # Remove silence
            frame_length = int(0.025 * 16000)
            hop_length = int(0.010 * 16000)
            energy = np.array([np.sum(np.abs(wav[i:i+frame_length]**2)) 
                               for i in range(0, len(wav)-frame_length, hop_length)])
            threshold = np.mean(energy) * 0.3
            voice_frames = energy > threshold
            
            voice_segments = []
            in_voice = False
            start_idx = 0
            
            for i, is_voice in enumerate(voice_frames):
                sample_idx = i * hop_length
                if is_voice and not in_voice:
                    start_idx = sample_idx
                    in_voice = True
                elif not is_voice and in_voice:
                    if sample_idx - start_idx > 16000:
                        voice_segments.append(wav[start_idx:sample_idx])
                    in_voice = False
            
            if voice_segments:
                wav_clean = np.concatenate(voice_segments)
            else:
                wav_clean = wav
            
            # Process chunks
            chunk_size = 16000 * 60
            if len(wav_clean) > chunk_size:
                embeddings = []
                for i in range(0, len(wav_clean), chunk_size):
                    chunk = wav_clean[i:i+chunk_size]
                    if len(chunk) > 16000:
                        emb = encoder.embed_utterance(chunk)
                        embeddings.append(emb)
                        gc.collect()
                embedding = np.mean(embeddings, axis=0)
            else:
                embedding = encoder.embed_utterance(wav_clean)
            
            del wav, wav_clean
            gc.collect()
        else:
            # Standard processing
            print("   Using standard processing...")
            wav = preprocess_wav(audio_path)
            embedding = encoder.embed_utterance(wav)
        
        # Save to database
        success = db.add_speaker(speaker_code, display_name, embedding, audio_path)
        
        if success:
            print(f"   ✓ Registered successfully!")
            return True
        else:
            print(f"   ✗ Registration failed")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def identify_speaker_simple(audio_path):
    """Identify a speaker with automatic optimization."""
    try:
        if not Path(audio_path).exists():
            print(f"   ✗ File not found: {audio_path}")
            return None, 0
        
        # Check duration
        info = sf.info(audio_path)
        duration = info.duration
        
        # Auto-optimize for long audio (>1 minute) - Process ENTIRE audio
        if duration > 60:
            print(f"   Long audio ({duration:.1f}s), processing entire audio in chunks...")
            
            # Load entire audio
            wav, _ = librosa.load(audio_path, sr=16000)
            
            # Process in 45-second chunks with quality filtering
            chunk_duration = 45
            chunk_size = chunk_duration * 16000
            
            chunk_embeddings = []
            chunk_confidences = []
            
            for i in range(0, len(wav), chunk_size):
                chunk = wav[i:i+chunk_size]
                if len(chunk) > 32000:  # At least 2 seconds
                    energy = np.abs(chunk)
                    
                    if len(energy) > 100:
                        energy_sorted = np.sort(energy)
                        energy_threshold = energy_sorted[int(len(energy_sorted) * 0.3)]
                    else:
                        energy_threshold = np.mean(energy) * 0.25
                    
                    voice_mask = energy > energy_threshold
                    
                    if np.sum(voice_mask) > 32000:  # At least 2s of voice
                        chunk_clean = chunk[voice_mask]
                        if len(chunk_clean) > 32000:
                            emb = encoder.embed_utterance(chunk_clean)
                            voice_ratio = np.sum(voice_mask) / len(voice_mask)
                            chunk_embeddings.append(emb)
                            chunk_confidences.append(voice_ratio)
                gc.collect()
            
            if not chunk_embeddings:
                test_embedding = encoder.embed_utterance(wav)
            else:
                weights = np.array(chunk_confidences)
                weights = weights / np.sum(weights)
                test_embedding = np.average(chunk_embeddings, axis=0, weights=weights)
                print(f"   Analyzed {len(chunk_embeddings)} high-quality voice segments")
            
            del wav
            gc.collect()
        else:
            wav = preprocess_wav(audio_path)
            test_embedding = encoder.embed_utterance(wav)
        
        # Search database
        speakers = db.get_all_speakers()
        
        if not speakers:
            print("   ✗ No speakers in database")
            return None, 0
        
        # Find best match
        best_code = None
        best_name = None
        best_score = 0
        
        for code, info in speakers.items():
            similarity = np.dot(test_embedding, info['embedding']) / (
                np.linalg.norm(test_embedding) * np.linalg.norm(info['embedding'])
            )
            
            if similarity > best_score:
                best_score = similarity
                best_code = code
                best_name = info['name']
        
        return (best_code, best_name, best_score)
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return None, 0


def register_speakers():
    """Register all speakers from list."""
    print("="*60)
    print("REGISTERING SPEAKERS")
    print("="*60)
    
    for audio_file, speaker_code, display_name in SPEAKERS_TO_REGISTER:
        print(f"\n{display_name} ({speaker_code})")
        register_speaker_simple(audio_file, speaker_code, display_name)


def test_identification():
    """Test identification on all audio files."""
    print("\n" + "="*60)
    print("TESTING SPEAKER IDENTIFICATION")
    print("="*60)
    
    speakers = db.get_all_speakers()
    
    if not speakers:
        print("\n✗ No registered speakers found!")
        print("Please register speakers first.")
        return
    
    print(f"\nLoaded {len(speakers)} registered speakers")
    
    for audio_file in AUDIO_FILES_TO_TEST:
        print(f"\n{'='*60}")
        print(f"Testing: {audio_file}")
        print('='*60)
        
        result = identify_speaker_simple(audio_file)
        
        if result and result[0]:
            speaker_code, speaker_name, score = result
            
            if score >= THRESHOLD:
                print(f"   ✓ VERIFIED: {speaker_name}")
                print(f"   Speaker Code: {speaker_code}")
                print(f"   Confidence: {score*100:.2f}%")
            else:
                print(f"   ✗ NOT VERIFIED (best match: {speaker_name})")
                print(f"   Confidence: {score*100:.2f}% (threshold: {THRESHOLD*100:.0f}%)")
        else:
            print("   ✗ Identification failed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SPEAKER RECOGNITION TEST")
    print("="*60)
    print("\nThis will:")
    print("1. Register speakers from SPEAKERS_TO_REGISTER list")
    print("2. Test/identify audio files from AUDIO_FILES_TO_TEST list")
    print(f"3. Verification threshold: {THRESHOLD*100:.0f}%")
    print("="*60)
    
    input("\nPress Enter to start...")
    
    try:
        # Step 1: Register speakers
        register_speakers()
        
        # Step 2: Test identification
        test_identification()
        
        print("\n" + "="*60)
        print("TEST COMPLETED")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
