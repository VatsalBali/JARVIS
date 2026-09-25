# Training the "Oracle" wake word

Until this is done, ORACLE wakes on openWakeWord's pretrained **"Hey Jarvis"** model. The rest of the voice pipeline is the same either way.

## 1. Train the model

openWakeWord trains custom words entirely from synthetic speech, so you don't record anything yourself.

1. Open the openWakeWord repository: <https://github.com/dscripka/openWakeWord>.
2. In its README, find **Training New Models** and open the linked Google Colab notebook. The simple Colab version is enough to start; the full `automatic_model_training` notebook in `notebooks/` gives better accuracy.
3. Set the target phrase to `oracle`. Also try a phonetic spelling like `or uh cull` if the first model misses a lot.
4. In Colab, set **Runtime → Change runtime type → GPU**, then run all cells. Expect roughly an hour.
5. Download the resulting **`.onnx`** file (not `.tflite`).

## 2. Install it

Rename the file to `oracle.onnx` and put it here:

```
%LOCALAPPDATA%\ORACLE\wakeword\oracle.onnx
```

Restart ORACLE. The orb's asleep caption changes to *Say "Oracle"*. To use a model somewhere else, set `ORACLE_WAKE_MODEL` to its path.

## 3. Tune it

Every wake, and every near miss that didn't reach the required consecutive frames, is logged to:

```
%LOCALAPPDATA%\ORACLE\wake_log.csv
```

- **False wakes** (the README target is under about one a day): raise the threshold. Set it to `0.6` with `core.set_setting("wake_threshold", "0.6")`.
- **Missed wakes from across the room**: lower it to `0.4`, or retrain with more training steps.
- `WAKE_PATIENCE` in `voice_engine.py` (consecutive 80 ms frames above the threshold, default 2) is the other lever.

## Voice ID threshold

Voice ID compares each request with your enrolled voiceprint (tray → **Learn my voice…**, or say "learn my voice"). The default similarity threshold is `0.75`. In testing, the same voice scored about 0.97 and a different voice about 0.58. If ORACLE often fails to recognise you, lower it with `core.set_setting("voice_id_threshold", "0.7")`. Re-enrolling in the room you normally use also helps.
