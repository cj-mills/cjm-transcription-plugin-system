import os
import numpy as np
from cjm_transcription_plugin_system.core import AudioData

def test_audio_data_serialization():
    # 1. Create dummy audio (1 second of silence at 16khz)
    samples = np.zeros(16000, dtype=np.float32)
    audio = AudioData(samples=samples, sample_rate=16000)
    
    # 2. Trigger the "Zero-Copy" logic
    file_path = audio.to_temp_file()
    
    print(f"Audio saved to: {file_path}")
    
    # 3. Verify file exists
    assert os.path.exists(file_path)
    assert file_path.endswith(".wav")
    
    # 4. Cleanup
    os.remove(file_path)
    print("✅ AudioData serialization passed")

if __name__ == "__main__":
    test_audio_data_serialization()