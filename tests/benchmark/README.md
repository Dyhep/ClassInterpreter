# 识别准确率基准测试

1. `python synth.py lecture.txt out/` — 用离线 TTS（sherpa-onnx piper 声音）合成讲课录音，生成三种条件：
   `quiet`（安静）、`room`（小房间外放）、`classroom`（远距离 + 同学说话 + 混响）。
2. `python score.py out/ '{"名称":[{设置覆盖}, "parakeet"]}' quiet,room,classroom` — 逐词比对，输出错误率和漏词率。

2026-10 的结果见 `app/asr.py` 里 `MODEL_CFG` 上方的注释。
