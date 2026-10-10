# 课堂同传（ClassInterpreter）

上课、开会时实时把英文语音转成文字并翻译成中文，下课一键生成中文课堂笔记。Windows 10/11 桌面版。

## 功能
- **实时英文字幕 + 中文翻译**（双语 / 只看中文 / 只看英文，字号可调）
- **声音来源**：麦克风（现场上课）或电脑声音（网课、Zoom、视频）
- **导入录音文件**（wav / mp3 / flac / ogg），比实时快很多
- **课堂总结**：主题、主要内容、术语表（中英对照）、作业与考试提醒、复习建议
- **自动保存**：每节课一个文件夹（`文档\课堂同传记录`），含双语转写 `.md` 和总结 `.md`
- **窗口置顶**，看网课时浮在视频上面

## 两种模式
| | 本地离线（默认，免费） | 填 Claude API key 后 |
|---|---|---|
| 语音识别 | 本机 | 本机 |
| 翻译 | 本机 OPUS-MT | Claude（更自然，术语更准） |
| 总结 | 本机 Qwen2.5-1.5B（首次需下载约 1.1GB，一小时课约 5–10 分钟） | Claude（几十秒） |
| 费用 | 0 | 约 $0.15 / 小时课 |

## 技术组成
- 语音识别：[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) + Silero VAD + Moonshine base（轻量）/ Parakeet-TDT 0.6B（高精度，可选下载）
- 翻译：Helsinki-NLP/opus-mt-en-zh，用 CTranslate2 int8 运行
- 本地总结：[llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` + Qwen2.5-1.5B-Instruct Q4_K_M
- 界面：pywebview（Edge WebView2）
- 打包：PyInstaller + Inno Setup，由 GitHub Actions 自动构建

## 下载安装
到本仓库的 **Releases** 页面下载 `ClassInterpreter-Setup-x.y.z.exe`，双击安装。
第一次运行如果 Windows 提示“已保护你的电脑”，点 **更多信息 → 仍要运行**（因为没有付费代码签名）。

## 发布新版本
```
git tag v0.1.0 && git push origin v0.1.0
```
GitHub Actions 会自动：下载/转换模型 → 端到端测试 → 打包 → 发布到 Releases。

## 本地开发
```
pip install -r requirements.txt
pip install transformers huggingface_hub torch   # 只有准备模型时需要
python build/prepare_models.py dist_assets
set CI_MODELS_DIR=%cd%\dist_assets\models
set LLAMA_SERVER=%cd%\dist_assets\bin\llama\llama-server.exe
python app/main.py
```
