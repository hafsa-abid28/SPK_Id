"""
Speaker Registration Script - PostgreSQL Version
Register speakers into the PostgreSQL database with their voice embeddings
Automatically optimizes processing for long audio files (>5 minutes)
"""

from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from pathlib import Path
import sys
import librosa
import soundfile as sf
from speaker_database_pg import SpeakerDatabasePG
import gc

# Initialize the voice encoder (CPU mode)
encoder = VoiceEncoder(device='cpu')
print("✓ Voice encoder loaded")


def register_speaker(audio_path, speaker_name, speaker_code):
    """
    Register a new speaker in the database
    
    Args:
        audio_path: Path to the audio file
        speaker_name: Display name of the speaker
        speaker_code: Unique identifier for the speaker
    
    Returns:
        bool: True if registration successful
    """
    try:
        # Check if audio file exists
        audio_file = Path(audio_path)
        if not audio_file.exists():
            print(f"✗ Audio file not found: {audio_path}")
            return False
        
        print(f"\n{'='*60}")
        print(f"Registering speaker: {speaker_name} (Code: {speaker_code})")
        print(f"Audio file: {audio_path}")
        print(f"{'='*60}")
        
        # Check audio duration and automatically choose processing method
        info = sf.info(audio_path)
        total_duration = info.duration
        print(f"Audio duration: {total_duration:.2f}s ({total_duration/60:.1f} min)")
        
        # AUTOMATIC OPTIMIZATION: Long audio (>5 minutes)
        if total_duration > 300:
            print("Using optimized processing for long audio...")
            
            # Load audio info
            sample_rate = info.samplerate
            max_duration = 600  # Use first 10 minutes for better accuracy
            
            print(f"  - Sample rate: {sample_rate} Hz")
            print(f"  - Using first {max_duration}s with voice activity")
            
            # Load audio
            wav, _ = librosa.load(audio_path, sr=16000, duration=max_duration)
            
            # Remove silence using improved VAD
            print("  - Removing silence (improved VAD)...")
            frame_length = int(0.025 * 16000)
            hop_length = int(0.010 * 16000)
            
            # Calculate energy with better normalization
            energy = np.array([np.sum(np.abs(wav[i:i+frame_length]**2)) 
                               for i in range(0, len(wav)-frame_length, hop_length)])
            
            # Use adaptive threshold based on energy distribution
            energy_sorted = np.sort(energy)
            threshold = energy_sorted[int(len(energy_sorted) * 0.4)]  # 40th percentile
            voice_frames = energy > threshold
            
            # Extract voice segments
            voice_segments = []
            in_voice = False
            start_idx = 0
            
            for i, is_voice in enumerate(voice_frames):
                sample_idx = i * hop_length
                if is_voice and not in_voice:
                    start_idx = sample_idx
                    in_voice = True
                elif not is_voice and in_voice:
                    # Only keep segments longer than 2 seconds for quality
                    if sample_idx - start_idx > 32000:  # 2 seconds minimum
                        voice_segments.append(wav[start_idx:sample_idx])
                    in_voice = False
            
            if voice_segments:
                wav_clean = np.concatenate(voice_segments)
                voice_duration = len(wav_clean) / 16000
                print(f"  - Voice: {voice_duration:.1f}s (removed {max_duration - voice_duration:.1f}s silence)")
            else:
                wav_clean = wav
                voice_duration = len(wav_clean) / 16000
            
            # Process in chunks with better quality
            print("  - Generating high-quality embedding...")
            chunk_size = 16000 * 90  # 90 second chunks for better embeddings
            if len(wav_clean) > chunk_size:
                embeddings = []
                num_chunks = (len(wav_clean) + chunk_size - 1) // chunk_size
                print(f"  - Processing {num_chunks} chunks...")
                
                for i in range(0, len(wav_clean), chunk_size):
                    chunk = wav_clean[i:i+chunk_size]
                    if len(chunk) > 16000:
                        emb = encoder.embed_utterance(chunk)
                        embeddings.append(emb)
                        print(f"    Chunk {len(embeddings)}/{num_chunks}", end='\r')
                        gc.collect()
                
                print()
                embedding = np.mean(embeddings, axis=0)
            else:
                embedding = encoder.embed_utterance(wav_clean)
            
            del wav, wav_clean
            gc.collect()
            
            print(f"✓ Embedding generated")
            print(f"  - Dimension: {embedding.shape[0]}")
            print(f"  - Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
            print(f"  - Processed duration: {voice_duration:.1f}s")
            
        # STANDARD PROCESSING: Short audio (<=5 minutes)
        else:
            print("Using standard processing...")
            wav = preprocess_wav(audio_path)
            embedding = encoder.embed_utterance(wav)
            
            print(f"✓ Embedding generated")
            print(f"  - Dimension: {embedding.shape[0]}")
            print(f"  - Duration: {total_duration:.2f}s")
        
        # Save to database
        db = SpeakerDatabasePG()
        success = db.add_speaker(speaker_code, speaker_name, embedding, audio_path)
        
        if success:
            print(f"\n✓ Successfully registered '{speaker_name}' (Code: {speaker_code})")
            print(f"  - Embedding dimension: {embedding.shape[0]}")
            print(f"  - Audio file: {audio_path}")
            return True
        else:
            print(f"\n✗ Failed to register speaker")
            return False
            
    except Exception as e:
        print(f"✗ Error during registration: {e}")
        import traceback
        traceback.print_exc()
        return False


def register_multiple_speakers(audio_dir):
    """
    Register multiple speakers from a directory
    Expected filename format: speakercode_displayname.wav
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        print(f"✗ Directory not found: {audio_dir}")
        return
    
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
    
    if not audio_files:
        print("✗ No audio files found in directory")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    for audio_file in audio_files:
        # Parse filename (format: code_name.ext)
        filename = audio_file.stem
        parts = filename.split("_", 1)
        
        if len(parts) == 2:
            speaker_code, speaker_name = parts
            register_speaker(str(audio_file), speaker_name, speaker_code)
        else:
            print(f"⚠ Skipping {audio_file.name} - invalid format (use: code_name.ext)")


def interactive_registration():
    """Interactive registration mode"""
    db = SpeakerDatabasePG()
    
    while True:
        print("\n" + "="*60)
        print("Speaker Registration System - PostgreSQL")
        print("="*60)
        print("1. Register single speaker")
        print("2. Register multiple speakers from directory")
        print("3. List all speakers")
        print("4. Remove speaker")
        print("5. Exit")
        print("="*60)
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            audio_path = input("\nEnter audio file path: ").strip()
            speaker_name = input("Enter speaker name: ").strip()
            speaker_code = input("Enter speaker code (unique ID): ").strip()
            
            if audio_path and speaker_name and speaker_code:
                register_speaker(audio_path, speaker_name, speaker_code)
            else:
                print("✗ All fields are required")
        
        elif choice == "2":
            audio_dir = input("\nEnter directory path: ").strip()
            if audio_dir:
                register_multiple_speakers(audio_dir)
            else:
                print("✗ Directory path is required")
        
        elif choice == "3":
            db.list_speakers()
        
        elif choice == "4":
            speaker_code = input("\nEnter speaker code to remove: ").strip()
            if speaker_code:
                db.remove_speaker(speaker_code)
            else:
                print("✗ Speaker code is required")
        
        elif choice == "5":
            print("\nGoodbye!")
            break
        
        else:
            print("✗ Invalid option")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - run interactive mode
        interactive_registration()
    elif len(sys.argv) == 4:
        # Command line mode: python speaker_registration_pg.py audio_path speaker_name speaker_code
        audio_path = sys.argv[1]
        speaker_name = sys.argv[2]
        speaker_code = sys.argv[3]
        register_speaker(audio_path, speaker_name, speaker_code)
    else:
        print("Usage:")
        print("  Interactive mode: python speaker_registration_pg.py")
        print("  CLI mode: python speaker_registration_pg.py <audio_path> <speaker_name> <speaker_code>")
        print("\nExample:")
        print("  python speaker_registration_pg.py sample.mp3 'John Doe' john_doe")
