"""
Speaker Verification Script - PostgreSQL Version
Verify speakers against the PostgreSQL database
Automatically optimizes processing for long audio files (>1 minute)
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


def verify_speaker(audio_path, threshold=0.75, top_n=3):
    """
    Verify a speaker against all registered speakers in the database
    
    Args:
        audio_path: Path to the audio file to verify
        threshold: Minimum similarity threshold for verification (default: 0.75)
        top_n: Number of top matches to display
    
    Returns:
        tuple: (speaker_code, speaker_name, similarity_score) or (None, None, best_score)
    """
    try:
        # Check if audio file exists
        audio_file = Path(audio_path)
        if not audio_file.exists():
            print(f"✗ Audio file not found: {audio_path}")
            return None, None, 0
        
        print(f"\n{'='*60}")
        print(f"Verifying speaker from: {audio_path}")
        print(f"Threshold: {threshold*100:.2f}%")
        print(f"{'='*60}")
        
        # Check audio duration and automatically choose processing method
        info = sf.info(audio_path)
        duration = info.duration
        print(f"Audio duration: {duration:.2f}s ({duration/60:.1f} min)")
        
        # AUTOMATIC OPTIMIZATION: Long audio (>1 minute) - Process ENTIRE audio in chunks
        if duration > 60:
            print(f"Processing entire audio in chunks to find all speakers...")
            
            # Load entire audio
            wav, _ = librosa.load(audio_path, sr=16000)
            
            # Process in 45-second chunks for better accuracy
            chunk_duration = 45  # seconds
            chunk_size = chunk_duration * 16000
            num_chunks = (len(wav) + chunk_size - 1) // chunk_size
            
            print(f"  - Analyzing {num_chunks} chunks of {chunk_duration}s each...")
            
            # Collect all chunk embeddings with quality filtering
            chunk_embeddings = []
            chunk_confidences = []
            
            for i in range(0, len(wav), chunk_size):
                chunk = wav[i:i+chunk_size]
                if len(chunk) > 32000:  # At least 2 seconds
                    # Improved silence removal
                    energy = np.abs(chunk)
                    
                    # Use adaptive threshold
                    if len(energy) > 100:
                        energy_sorted = np.sort(energy)
                        energy_threshold = energy_sorted[int(len(energy_sorted) * 0.3)]
                    else:
                        energy_threshold = np.mean(energy) * 0.25
                    
                    voice_mask = energy > energy_threshold
                    
                    # Only use chunks with sufficient voice content (at least 2 seconds)
                    if np.sum(voice_mask) > 32000:  # At least 2s of voice
                        chunk_clean = chunk[voice_mask]
                        if len(chunk_clean) > 32000:
                            emb = encoder.embed_utterance(chunk_clean)
                            
                            # Calculate confidence based on voice ratio
                            voice_ratio = np.sum(voice_mask) / len(voice_mask)
                            chunk_embeddings.append(emb)
                            chunk_confidences.append(voice_ratio)
                
                print(f"    Processed {min(i+chunk_size, len(wav))//16000}/{int(duration)}s", end='\r')
                gc.collect()
            
            print()  # New line
            
            if not chunk_embeddings:
                print("  ✗ No voice detected in audio")
                return None, None, 0
            
            # Use weighted average based on voice quality
            if len(chunk_confidences) > 0:
                weights = np.array(chunk_confidences)
                weights = weights / np.sum(weights)  # Normalize
                test_embedding = np.average(chunk_embeddings, axis=0, weights=weights)
                print(f"  ✓ Generated high-quality embedding from {len(chunk_embeddings)} voice chunks")
            else:
                test_embedding = np.mean(chunk_embeddings, axis=0)
                print(f"  ✓ Generated embedding from {len(chunk_embeddings)} voice chunks")
            
            del wav
            gc.collect()
            
        # STANDARD PROCESSING: Short audio (<=1 minute)
        else:
            print("Using standard processing...")
            wav = preprocess_wav(audio_path)
            test_embedding = encoder.embed_utterance(wav)
        
        print(f"✓ Embedding generated (dimension: {test_embedding.shape[0]})")
        
        # Load database
        db = SpeakerDatabasePG()
        speakers = db.get_all_speakers()
        
        if not speakers:
            print("✗ No speakers in database. Please register speakers first.")
            return None, None, 0
        
        print(f"✓ Comparing against {len(speakers)} registered speakers...")
        
        # Calculate similarities
        similarities = []
        for code, info in speakers.items():
            similarity = np.dot(test_embedding, info['embedding']) / (
                np.linalg.norm(test_embedding) * np.linalg.norm(info['embedding'])
            )
            similarities.append((code, info['name'], similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # Get best match
        best_code, best_name, best_score = similarities[0]
        
        print()
        if best_score >= threshold:
            print(f"✓ VERIFIED: Speaker identified as '{best_name}' (Code: {best_code})")
            print(f"  Confidence: {best_score*100:.2f}%")
            return best_code, best_name, best_score
        else:
            print(f"✗ VERIFICATION FAILED: No matching speaker found")
            print(f"  (Highest match was {best_score*100:.2f}%, threshold is {threshold*100:.2f}%)")
            return None, None, best_score
            
    except Exception as e:
        print(f"✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0


def verify_speaker_against_specific(audio_path, speaker_code, threshold=0.75):
    """
    Verify if audio matches a specific speaker
    
    Args:
        audio_path: Path to the audio file to verify
        speaker_code: The specific speaker code to verify against
        threshold: Minimum similarity threshold
    
    Returns:
        bool: True if verified, False otherwise
    """
    try:
        audio_file = Path(audio_path)
        if not audio_file.exists():
            print(f"✗ Audio file not found: {audio_path}")
            return False
        
        # Check audio duration and automatically choose processing method
        info = sf.info(audio_path)
        duration = info.duration
        
        # AUTOMATIC OPTIMIZATION: Long audio (>1 minute) - Process ENTIRE audio
        if duration > 60:
            print(f"Processing entire audio in chunks...")
            
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
            
            del wav
            gc.collect()
            
        # STANDARD PROCESSING: Short audio (<=1 minute)
        else:
            wav = preprocess_wav(audio_path)
            test_embedding = encoder.embed_utterance(wav)
        
        # Get specific speaker from database
        db = SpeakerDatabasePG()
        speaker_info = db.get_speaker(speaker_code)
        
        if not speaker_info:
            print(f"✗ Speaker code '{speaker_code}' not found in database")
            return False
        
        # Calculate similarity
        similarity = np.dot(test_embedding, speaker_info['embedding']) / (
            np.linalg.norm(test_embedding) * np.linalg.norm(speaker_info['embedding'])
        )
        
        print(f"\n{'='*60}")
        print(f"Verifying against: {speaker_info['name']} (Code: {speaker_code})")
        print(f"Similarity: {similarity*100:.2f}%")
        print(f"Threshold: {threshold*100:.2f}%")
        
        if similarity >= threshold:
            print(f"✓ VERIFIED: Audio matches {speaker_info['name']}")
            print(f"{'='*60}")
            return True
        else:
            print(f"✗ NOT VERIFIED: Audio does not match {speaker_info['name']}")
            print(f"{'='*60}")
            return False
            
    except Exception as e:
        print(f"✗ Error during verification: {e}")
        return False


def scan_entire_audio(audio_path, segment_duration=30, threshold=0.75):
    """
    Scan entire audio file and detect ALL speakers who appear
    Useful for conversations with multiple speakers at different times
    
    Args:
        audio_path: Path to audio file
        segment_duration: Duration of each segment to analyze (seconds)
        threshold: Minimum confidence to consider a match
    
    Returns:
        dict: {speaker_code: {'name': str, 'segments': list, 'total_duration': float, 'confidence_avg': float}}
    """
    try:
        audio_file = Path(audio_path)
        if not audio_file.exists():
            print(f"✗ Audio file not found: {audio_path}")
            return {}
        
        print(f"\n{'='*60}")
        print(f"Scanning entire audio: {audio_path}")
        print(f"Segment duration: {segment_duration}s")
        print(f"Threshold: {threshold*100:.2f}%")
        print(f"{'='*60}")
        
        # Get audio info
        info = sf.info(audio_path)
        total_duration = info.duration
        print(f"Total audio duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        
        # Load database
        db = SpeakerDatabasePG()
        speakers_db = db.get_all_speakers()
        
        if not speakers_db:
            print("✗ No speakers in database")
            return {}
        
        print(f"Loaded {len(speakers_db)} registered speakers")
        
        # Process audio in segments
        num_segments = int(np.ceil(total_duration / segment_duration))
        print(f"Processing {num_segments} segments...\n")
        
        detected_speakers = {}
        
        for segment_idx in range(num_segments):
            start_time = segment_idx * segment_duration
            
            # Load segment
            wav, _ = librosa.load(audio_path, sr=16000, offset=start_time, duration=segment_duration)
            
            # Skip if too short
            if len(wav) < 16000:  # Less than 1 second
                continue
            
            # Quick silence check
            energy = np.mean(np.abs(wav))
            if energy < 0.01:  # Mostly silence
                print(f"  Segment {segment_idx+1}/{num_segments} ({start_time:.0f}s): [Silence]")
                continue
            
            # Generate embedding
            embedding = encoder.embed_utterance(wav)
            
            # Find best match
            best_code = None
            best_name = None
            best_score = 0
            
            for code, info in speakers_db.items():
                similarity = np.dot(embedding, info['embedding']) / (
                    np.linalg.norm(embedding) * np.linalg.norm(info['embedding'])
                )
                
                if similarity > best_score:
                    best_score = similarity
                    best_code = code
                    best_name = info['name']
            
            # Record if above threshold
            if best_score >= threshold:
                if best_code not in detected_speakers:
                    detected_speakers[best_code] = {
                        'name': best_name,
                        'segments': [],
                        'confidences': []
                    }
                
                detected_speakers[best_code]['segments'].append({
                    'start': start_time,
                    'end': min(start_time + segment_duration, total_duration),
                    'confidence': best_score
                })
                detected_speakers[best_code]['confidences'].append(best_score)
                
                print(f"  Segment {segment_idx+1}/{num_segments} ({start_time:.0f}s): {best_name} ({best_score*100:.1f}%)")
            else:
                print(f"  Segment {segment_idx+1}/{num_segments} ({start_time:.0f}s): Unknown ({best_score*100:.1f}%)")
            
            gc.collect()
        
        # Calculate statistics
        for code, data in detected_speakers.items():
            total_time = sum(seg['end'] - seg['start'] for seg in data['segments'])
            avg_confidence = np.mean(data['confidences'])
            
            data['total_duration'] = total_time
            data['confidence_avg'] = avg_confidence
            data['percentage'] = (total_time / total_duration) * 100
        
        # Display results
        print(f"\n{'='*60}")
        print("SCAN RESULTS:")
        print(f"{'='*60}")
        
        if detected_speakers:
            for code, data in detected_speakers.items():
                print(f"\n✓ {data['name']} (Code: {code})")
                print(f"  - Speaking time: {data['total_duration']:.1f}s ({data['percentage']:.1f}% of audio)")
                print(f"  - Average confidence: {data['confidence_avg']*100:.1f}%")
                print(f"  - Segments: {len(data['segments'])}")
                print(f"  - Appears at: {', '.join([f'{s['start']:.0f}s' for s in data['segments'][:5]])}")
                if len(data['segments']) > 5:
                    print(f"    ... and {len(data['segments']) - 5} more")
        else:
            print("✗ No known speakers detected in this audio")
        
        print(f"{'='*60}")
        
        return detected_speakers
        
    except Exception as e:
        print(f"✗ Error during scan: {e}")
        import traceback
        traceback.print_exc()
        return {}


def batch_verify(audio_dir, threshold=0.75):
    """Verify all audio files in a directory"""
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        print(f"✗ Directory not found: {audio_dir}")
        return
    
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
    
    if not audio_files:
        print("✗ No audio files found in directory")
        return
    
    print(f"Found {len(audio_files)} audio files to verify\n")
    
    results = []
    for audio_file in audio_files:
        code, name, score = verify_speaker(str(audio_file), threshold)
        results.append((audio_file.name, name, score))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    for filename, name, score in results:
        status = "✓" if score >= threshold else "✗"
        speaker = name if name else "Unknown"
        print(f"{status} {filename:<30} -> {speaker:<20} ({score*100:.2f}%)")
    print("="*60)


def interactive_verification():
    """Interactive verification mode"""
    db = SpeakerDatabasePG()
    
    while True:
        print("\n" + "="*60)
        print("Speaker Verification System - PostgreSQL")
        print("="*60)
        print("1. Verify speaker (identify from all)")
        print("2. Verify against specific speaker")
        print("3. Batch verify directory")
        print("4. Scan entire audio (find all speakers)")
        print("5. List all speakers")
        print("6. Exit")
        print("="*60)
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            audio_path = input("\nEnter audio file path: ").strip()
            if audio_path:
                threshold_input = input("Enter threshold (0-1, default 0.75): ").strip()
                threshold = float(threshold_input) if threshold_input else 0.75
                verify_speaker(audio_path, threshold)
            else:
                print("✗ Audio path is required")
        
        elif choice == "2":
            audio_path = input("\nEnter audio file path: ").strip()
            speaker_code = input("Enter speaker code: ").strip()
            if audio_path and speaker_code:
                threshold_input = input("Enter threshold (0-1, default 0.75): ").strip()
                threshold = float(threshold_input) if threshold_input else 0.75
                verify_speaker_against_specific(audio_path, speaker_code, threshold)
            else:
                print("✗ All fields are required")
        
        elif choice == "3":
            audio_dir = input("\nEnter directory path: ").strip()
            if audio_dir:
                threshold_input = input("Enter threshold (0-1, default 0.75): ").strip()
                threshold = float(threshold_input) if threshold_input else 0.75
                batch_verify(audio_dir, threshold)
            else:
                print("✗ Directory path is required")
        
        elif choice == "4":
            audio_path = input("\nEnter audio file path: ").strip()
            if audio_path:
                segment_input = input("Enter segment duration in seconds (default 30): ").strip()
                segment_duration = int(segment_input) if segment_input else 30
                threshold_input = input("Enter threshold (0-1, default 0.75): ").strip()
                threshold = float(threshold_input) if threshold_input else 0.75
                scan_entire_audio(audio_path, segment_duration, threshold)
            else:
                print("✗ Audio path is required")
        
        elif choice == "5":
            db.list_speakers()
        
        elif choice == "6":
            print("\nGoodbye!")
            break
        
        else:
            print("✗ Invalid option")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - run interactive mode
        interactive_verification()
    elif len(sys.argv) == 2:
        # Verify against all speakers
        audio_path = sys.argv[1]
        verify_speaker(audio_path)
    elif len(sys.argv) == 3:
        # Verify against specific speaker
        audio_path = sys.argv[1]
        speaker_code = sys.argv[2]
        verify_speaker_against_specific(audio_path, speaker_code)
    else:
        print("Usage:")
        print("  Interactive mode: python speaker_verification_pg.py")
        print("  Identify speaker: python speaker_verification_pg.py <audio_path>")
        print("  Verify specific: python speaker_verification_pg.py <audio_path> <speaker_code>")
        print("\nExamples:")
        print("  python speaker_verification_pg.py test.mp3")
        print("  python speaker_verification_pg.py test.mp3 john_doe")
