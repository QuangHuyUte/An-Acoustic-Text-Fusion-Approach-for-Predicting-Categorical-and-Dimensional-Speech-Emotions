# IEMOCAP voice test set for the web demo

This folder contains 40 short IEMOCAP-style test utterances copied from the local prepared IEMOCAP audio cache.

Structure:
- neutral/: 10 WAV files + 10 JSON label files
- happy/: 10 WAV files + 10 JSON label files
- sad/: 10 WAV files + 10 JSON label files
- angry/: 10 WAV files + 10 JSON label files
- manifest.csv: one-row-per-audio summary

How to test in the web demo:
1. Open the web demo.
2. Use Upload audio and choose a WAV file from one of these emotion folders.
3. Do not paste the reference transcript as script.
4. Run Analyze. The backend should use local ASR to recognize text automatically.
5. Compare the predicted emotion/VAD with the JSON sidecar or manifest.csv.

Important: transcript_reference_only is included only for checking whether ASR is reasonable. It should not be used as manual input during live testing.
