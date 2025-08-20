**Watch the video here(https://drive.google.com/file/d/1-oQYccLDItMywfJzhqVbXv-3IIaMT9bB/view?usp=drive_links).**



**Overview**  
This project builds a spoken digit classification system that recognizes digits 0 through 9 from short audio recordings. The core objective is to take an audio file or microphone input and output the most likely digit spoken. It is built on top of the Free Spoken Digit Dataset (FSDD) which is hosted on Hugging Face under the dataset name mteb/free-spoken-digit-dataset.

The system is a small end to end pipeline covering dataset handling, feature extraction, model training, evaluation, and inference in both batch and live microphone modes. It is implemented in Python with scikit-learn, librosa, and sounddevice for audio processing and modeling.

**Dataset**  
The dataset used is mteb/free-spoken-digit-dataset. It contains recordings of spoken digits by different speakers. The dataset is downloaded and stored locally in Parquet format. The project scripts expect the following folder structure for dataset files:

data/fsdd_parquet/train-00000-of-00001.parquet  
data/fsdd_parquet/test-00000-of-00001.parquet

The dataset is then accessed using the Hugging Face datasets library and converted into NumPy arrays and optional WAV files when required.

**Installation Instructions**
1. Clone or download this repository.
    
2. Create a Python virtual environment in the project folder:  
    python -m venv .venv  
    On Windows, activate it with: ..venv\Scripts\Activate.ps1  
    On Linux or macOS, activate with: source .venv/bin/activate

3. Upgrade pip and install all dependencies:  
    pip install --upgrade pip  
    pip install -r requirements.txt  
    pip install resampy sounddevice
    

The extra two packages resampy and sounddevice are required because librosa depends on resampy for resampling audio and sounddevice is used for microphone input and output.

**Project Structure**  
src/ contains the main source code. It includes:

- train.py: trains a classification model from the dataset
- eval.py: evaluates a trained model on a dataset split
- predict_wav.py: predicts the digit for a given WAV file
- live_mic.py: runs live microphone inference in real time
- record_mic.py: records audio from the microphone and saves it as a WAV file
- data.py: utilities for loading the dataset and converting it to arrays
- features.py: feature extraction code (MFCC, log-mel, etc.)
- models.py: defines and builds machine learning pipelines
- utils.py: helper functions such as joblib load/save and path checks

**scripts/ contains helper scripts:**
- export_samples.py: exports a few WAV samples per digit from the dataset for quick testing
- inspect_parquet.py: inspects the contents of a Parquet dataset file
- check_paths.py: verifies environment and dataset paths

models/ holds saved trained models in joblib format.  
artifacts/ stores evaluation metrics such as cross validation results in JSON.  
samples/ stores exported or recorded WAV audio files for testing and inference.

**Running the Project Step by Step**

1. Export dataset samples into WAV files  
    This step converts a few test split samples from Parquet into WAV for quick debugging.  
    Example command:  
    python scripts/export_samples.py --split test --per_class 1  
    This creates files like samples/0_0.wav through samples/9_0.wav.
    
2. Train a model  
    Run training on the train split.  
    python src/train.py --split train  
    This saves a trained model to models/baseline.joblib and metrics to artifacts/cv_metrics.json.
    
3. Evaluate a model  
    Run evaluation on the test split using the trained model.  
    python src/eval.py --model_path models/baseline.joblib --split test  
    Example output might show accuracy close to 0.97 on the test set.

4. Predict from a WAV file  
    Once the model is trained, you can test it with individual audio files.  
    Command:  
    python src/predict_wav.py --model_path models/baseline.joblib --wav_path samples/0_0.wav  
    Example output:  
    Predicted digit: 0  
    Top-3 probs: 0:1.000, 8:0.000, 3:0.000  
    Latency ms — load: 9.5 feat: 2570.2 infer: 1.4 total: 2581.1

This shows the model predicted digit 0 with very high confidence and reports the latency breakdown for feature extraction and inference.


5. Live microphone inference  
    Run live microphone classification with the trained model.  
    python src/live_mic.py --model_path models/baseline.joblib --samplerate 16000 --block_dur 0.9  
    The program keeps listening to microphone audio in 0.9 second blocks. For each block it prints the predicted digit and top probabilities. Stop it with Ctrl+C.  
    Performance depends on microphone quality, background noise, and how clearly digits are spoken.
    
6. Record a WAV from microphone  
    To record and save your own digit samples:  
    python src/record_mic.py --out_path samples/mic_digit.wav --duration 2.0 --samplerate 16000  
    This records 2 seconds of audio at 16kHz and saves it for later testing with predict_wav.py.


**Optional Utilities**  
inspect_parquet.py lets you peek inside dataset parquet files to confirm audio arrays and labels are correct.  
check_paths.py helps debug file path issues by confirming expected directories exist and are readable.

**Metrics and Results**  
Cross-validation metrics are stored in artifacts/cv_metrics.json. Example:  
fold_acc = [0.67, 0.83, 0.54, 0.70, 0.76]  
mean = 0.70, std = 0.09
Test metrics are stored in artifacts/test_metrics.json. Example:  
accuracy = 0.97
Holdout metrics may also be logged in artifacts/holdout_metrics.json. Example:  
accuracy = 1.0
These results show the model generalizes well despite variance across folds.

**Limitations and Next Steps**

- Current system is designed for clean dataset recordings at 8kHz. Real-world microphone tests may show lower accuracy due to noise, echo, or mismatch in sampling rate.
- Noise handling or data augmentation could improve robustness. Adding background noise or speed/pitch perturbations during training is one possible next step.
- Streaming recognition is currently block-based. Continuous streaming could reduce delay and improve responsiveness.
- The model only supports English digits 0–9. Extending to multilingual digits or spoken commands would require new datasets.
- Latency is dominated by feature extraction. Faster feature extraction libraries or lightweight deep learning backends could reduce delays.

**Example Workflow Summary**

1. Export sample WAVs from dataset.
2. Train a baseline model.
3. Evaluate test accuracy.
4. Predict digits from static WAV files.
5. Test live microphone inference.
6. Record and test your own audio.

### 1. Set up environment

`# from project root python -m venv .venv .venv\Scripts\Activate.ps1  pip install --upgrade pip pip install -r requirements.txt`

### 2. Export sample WAVs from dataset

Generate test WAVs for quick sanity checks.

`python scripts/export_samples.py --split test --per_class 3`

Expected: files like `samples/0_0.wav`, `samples/1_1.wav` etc.

### 3. Train the model

`python src/train.py --algo logreg --C 2.0 --model_out models/baseline.joblib`

This saves the trained model to `models/baseline.joblib` and logs metrics in `artifacts/`.

### 4. Evaluate the model

`python src/eval.py --model_path models/baseline.joblib`

Outputs test set accuracy and comparison with CV metrics.

### 5. Predict digit from a WAV file

`python src/predict_wav.py --model_path models/baseline.joblib --wav_path samples/0_0.wav`

Output: predicted digit + probabilities + latency breakdown.

### 6. Live microphone inference (real-time test)

`python src/live_mic.py --model_path models/baseline.joblib --samplerate 16000 --block_dur 0.9`

The system keeps listening in ~1 second blocks.  
Stop with **Ctrl+C**.

### 7. Record your own audio and predict

Step A: Record a 2-second sample from your microphone

`python src/record_mic.py --out_path samples/mic_test.wav --duration 2.0 --samplerate 16000`

Step B: Run prediction on your recording
