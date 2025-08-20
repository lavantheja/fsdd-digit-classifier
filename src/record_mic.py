# record_mic.py
import sounddevice as sd, soundfile as sf

print("Recording 1 second... speak a digit now!")
y = sd.rec(int(16000*1.0), samplerate=16000, channels=1, dtype="float32")
sd.wait()
sf.write("samples/mic_test.wav", y.squeeze(), 16000)
print("Saved to samples/mic_test.wav")
