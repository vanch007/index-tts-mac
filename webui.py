import json
import os
import sys
import threading
import time
import gc

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import openpyxl  # Required for Excel file support

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 for inference if available")
parser.add_argument("--use_deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'

# Autodetect device and FP16 support to match API server behavior
import torch
if torch.cuda.is_available():
    device = "cuda"
    use_fp16 = cmd_args.fp16
elif hasattr(torch, "mps") and torch.backends.mps.is_available():
    device = "mps"
    use_fp16 = False  # 在MPS上禁用FP16以减少内存使用
else:
    device = "cpu"
    use_fp16 = False

# 延迟初始化模型，在第一次使用时加载
tts = None
model_loaded = False

def load_model():
    global tts, model_loaded
    if not model_loaded:
        from indextts.infer_v2 import IndexTTS2
        print("Loading model for the first time...")
        tts = IndexTTS2(model_dir=cmd_args.model_dir,
                        cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                        use_fp16=use_fp16,
                        device=device,
                        use_deepspeed=cmd_args.use_deepspeed,
                        use_cuda_kernel=cmd_args.cuda_kernel,
                        )
        model_loaded = True
        print("Model loaded successfully.")
    return tts

# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    example_cases = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None
        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0)]
                             )


def import_excel_data(excel_file):
    """
    从Excel文件导入批量任务数据
    """
    if not excel_file:
        return []
    
    try:
        # 使用pandas读取Excel文件
        df = pd.read_excel(excel_file.name)
        
        # 检查列数是否匹配
        if len(df.columns) != 14:
            gr.Warning(i18n("Excel文件列数不正确，应为14列"))
            return []
        
        # 转换为数组格式
        data = df.values.tolist()
        return data
    except Exception as e:
        gr.Warning(f"{i18n('导入Excel文件时出错')}: {str(e)}")
        return []


def gen_batch(batch_data, progress=gr.Progress()):
    # 确保模型已加载
    load_model()
    
    # 创建按日期和时间命名的文件夹
    batch_folder_name = f"batch-{time.strftime('%Y%m%d-%H%M', time.localtime())}"
    batch_folder_path = os.path.join("outputs", batch_folder_name)
    os.makedirs(batch_folder_path, exist_ok=True)

    output_files = []
    # 定义默认的kwargs参数
    kwargs = {
        "do_sample": False,
        "top_p": 1.0,
        "top_k": 0,
        "temperature": 1.0,
        "length_penalty": 1.0,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "max_mel_tokens": 800,
    }
    for idx, row in enumerate(batch_data):
        if len(row) != 14:
            continue  # Skip rows that don't have the correct number of columns
        
        emo_control_method = EMO_CHOICES.index(row[1]) if row[1] in EMO_CHOICES else 0
        prompt_audio = row[0]
        input_text_single = row[2]
        emo_upload = row[3]
        emo_weight = row[4]
        emo_text = row[5]
        vec = row[6:14]
        
        # 每个文件的输出路径
        output_path = os.path.join(batch_folder_path, f"file_{idx + 1}_{int(time.time())}.wav")
        output = tts.infer(spk_audio_prompt=prompt_audio, text=input_text_single,
                           output_path=output_path,
                           emo_audio_prompt=emo_upload, emo_alpha=emo_weight,
                           emo_vector=vec if sum(vec) > 0 else None,
                           use_emo_text=bool(emo_text), emo_text=emo_text,
                           verbose=cmd_args.verbose,
                           max_text_tokens_per_segment=int(cmd_args.gui_seg_tokens),
                           **kwargs)
        output_files.append(output_path)  # 返回音频文件路径
        
        # 更彻底的内存清理和资源释放
        tts.clear_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # 如果只有一个文件，直接返回该文件路径
    if len(output_files) == 1:
        return output_files[0]
    # 如果有多个文件，合并它们
    elif len(output_files) > 1:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for file in output_files:
            audio = AudioSegment.from_wav(file)
            combined += audio
        combined_path = os.path.join(batch_folder_path, f"combined_{int(time.time())}.wav")
        combined.export(combined_path, format="wav")
        return combined_path
    else:
        return None  # 返回音频文件路径列表


def gen_single(emo_control_method, prompt_audio, input_text_single, emo_upload, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text, emo_random, 
               saved_voice_top, use_saved_voice_checkbox_top,
               emo_reference_file, use_emo_reference_checkbox,
               max_text_tokens_per_segment=120,
               *args, progress=gr.Progress()):
    # 确保模型已加载
    load_model()
    
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    
    # 优先使用顶部已保存的音色
    if use_saved_voice_checkbox_top and saved_voice_top:
        prompt_audio = os.path.join("voices", f"{saved_voice_top}.pkl")
    
    # 处理情感参考音频
    if emo_control_method == 1 and use_emo_reference_checkbox and emo_reference_file:
        emo_upload = os.path.join("情感参考", emo_reference_file)
    
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:
        emo_upload = None
        emo_weight = 1.0
    if emo_control_method == 1:
        emo_weight = emo_weight
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec_sum = sum([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8])
        if vec_sum > 1.5:
            gr.Warning(i18n("情感向量之和不能超过1.5，请调整后重试。"))
            return
    else:
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt_audio, text=input_text_single,
                       output_path=output_path,
                       emo_audio_prompt=emo_upload, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                       **kwargs)
    
    # 更彻底的内存清理和资源释放
    tts.clear_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

with gr.Blocks(title="IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
    <p align="center">
    <a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
    </p>
    ''')
    with gr.Tab(i18n("音频生成")):
        with gr.Row():
            # 添加音色下拉选项框
            with gr.Column():
                # 初始化音色列表
                try:
                    os.makedirs('voices', exist_ok=True)
                    voice_files = [f for f in os.listdir('voices') if f.endswith('.pkl')]
                    voice_names = [os.path.splitext(f)[0] for f in voice_files]
                except Exception as e:
                    print(f"初始化音色列表时出错: {e}")
                    voice_names = []
                
                saved_voices_top = gr.Dropdown(choices=voice_names, label=i18n("已保存音色"))
                refresh_voices_button_top = gr.Button(i18n("刷新音色列表"))
                use_saved_voice_checkbox_top = gr.Checkbox(label=i18n("使用已保存音色"), value=False)
            prompt_audio = gr.Audio(label=i18n("音色参考音频"), key="prompt_audio", sources=["upload", "microphone"], type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label=i18n("文本"), key="input_text_single", placeholder=i18n("请输入目标文本"), info=f"{i18n('当前模型版本')}...")
                gen_button = gr.Button(i18n("生成语音"), key="gen_button", interactive=True)
            output_audio = gr.Audio(label=i18n("生成结果"), visible=True, key="output_audio")
        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES,
                    type="index",
                    value=EMO_CHOICES[0],label=i18n("情感控制方式"))
        # 音色保存选项
        with gr.Group():
            with gr.Row():
                voice_name = gr.Textbox(label=i18n("音色名称"), placeholder=i18n("请输入音色名称，留空则使用音频文件名"))
                save_voice_button = gr.Button(i18n("保存当前音色"))
                save_status = gr.Textbox(label=i18n("保存状态"), interactive=False)
        
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                # 添加情感参考音频下拉选项
                try:
                    os.makedirs('情感参考', exist_ok=True)
                    emo_audio_files = [f for f in os.listdir('情感参考') if f.endswith(('.wav', '.mp3', '.flac'))]
                    emo_audio_paths = [os.path.join('情感参考', f) for f in emo_audio_files]
                except Exception as e:
                    print(f"初始化情感参考音频列表时出错: {e}")
                    emo_audio_files = []
                    emo_audio_paths = []
                
                emo_reference_dropdown = gr.Dropdown(choices=emo_audio_files, label=i18n("选择情感参考音频"))
                refresh_emo_reference_button = gr.Button(i18n("刷新情感参考列表"))
                use_emo_reference_checkbox = gr.Checkbox(label=i18n("使用预设情感参考"), value=False)
                
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

            with gr.Row():
                emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.6, value=0.8, step=0.01)

        # 情感随机采样
        with gr.Row():
            emo_random = gr.Checkbox(label=i18n("情感随机采样"),value=False,visible=False)

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            with gr.Row():
                emo_text = gr.Textbox(label=i18n("情感描述文本"), placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"), value="", info=i18n("例如：高兴，愤怒，悲伤等"))

        with gr.Accordion(i18n("高级生成参数设置"), open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    gr.Markdown(f"**{i18n('性能优化提示')}**: {i18n('默认参数已优化以提高推理速度。如需更高音频质量，可适当调整参数，但会增加生成时间。')}")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=False, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=0, step=1)
                        num_beams = gr.Slider(label="num_beams", value=1, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=1.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=1.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=800, minimum=50, maximum=8000, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(60, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=1024, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]
        
        if len(example_cases) > 0:
            gr.Examples(
                examples=example_cases,
                examples_per_page=20,
                inputs=[prompt_audio,
                        emo_control_method,
                        input_text_single,
                        emo_upload,
                        emo_weight,
                        emo_text,
                        vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8]
            )

    with gr.Tab(i18n("批量任务")):
        with gr.Row():
            batch_table = gr.Dataframe(
                headers=[i18n("音色参考音频"), i18n("情感控制方式"), i18n("文本"), i18n("上传情感参考音频"), i18n("情感权重"), i18n("情感描述文本"), i18n("喜"), i18n("怒"), i18n("哀"), i18n("惧"), i18n("厌恶"), i18n("低落"), i18n("惊喜"), i18n("平静")],
                datatype=["str", "str", "str", "str", "number", "str", "number", "number", "number", "number", "number", "number", "number", "number"],
                row_count=5,
                col_count=(14, "fixed"),
                type="array",
                key="batch_table"
            )
        with gr.Row():
            with gr.Column():
                import_excel_button = gr.UploadButton(
                    label=i18n("导入Excel文件"), 
                    file_types=[".xlsx", ".xls"],
                    file_count="single"
                )
                batch_gen_button = gr.Button(i18n("批量生成语音"), key="batch_gen_button", interactive=True)
            with gr.Column():
                gr.Markdown(i18n("**使用说明**：\n1. 创建一个Excel文件，包含14列数据，列名分别为：音色参考音频、情感控制方式、文本、上传情感参考音频、情感权重、情感描述文本、喜、怒、哀、惧、厌恶、低落、惊喜、平静\n2. 点击\"导入Excel文件\"按钮上传Excel文件\n3. 确认数据无误后点击\"批量生成语音\"按钮\n\n您也可以参考项目根目录下的 batch_tasks.xlsx 文件格式"))
        with gr.Row():
            batch_output = gr.Audio(label=i18n("批量生成结果"), type="filepath", key="batch_output")

    def on_input_text_change(text, max_text_tokens_per_segment):
        # 确保模型已加载
        load_model()
        
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview: gr.update(value=df),
            }
    def on_method_select(emo_control_method):
        if emo_control_method == 1:
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )
        elif emo_control_method == 2:
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False)
                    )
        elif emo_control_method == 3:
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        else:
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    def save_voice(spk_audio_path, voice_name):
        # 确保模型已加载
        load_model()
        
        if not spk_audio_path:
            return i18n("请先上传音色参考音频")
        
        try:
            voice_path = tts.save_speaker_voice(spk_audio_path, voice_name)
            return f"{i18n('音色保存成功')}: {voice_path}"
        except Exception as e:
            return f"{i18n('音色保存失败')}: {str(e)}"

    def refresh_voices():
        os.makedirs('voices', exist_ok=True)
        voice_files = [f for f in os.listdir('voices') if f.endswith('.pkl')]
        voice_names = [os.path.splitext(f)[0] for f in voice_files]
        return gr.update(choices=voice_names)

    def use_saved_voice_checkbox_change(use_saved):
        return gr.update(visible=use_saved), gr.update(visible=not use_saved)

    def refresh_voices_top():
        os.makedirs('voices', exist_ok=True)
        voice_files = [f for f in os.listdir('voices') if f.endswith('.pkl')]
        voice_names = [os.path.splitext(f)[0] for f in voice_files]
        return gr.update(choices=voice_names)
    
    def use_saved_voice_checkbox_top_change(use_saved):
        return gr.update(visible=use_saved), gr.update(visible=not use_saved)

    def refresh_emo_reference_list():
        try:
            os.makedirs('情感参考', exist_ok=True)
            emo_audio_files = [f for f in os.listdir('情感参考') if f.endswith(('.wav', '.mp3', '.flac'))]
        except Exception as e:
            print(f"刷新情感参考音频列表时出错: {e}")
            emo_audio_files = []
        return gr.update(choices=emo_audio_files)
    
    def use_emo_reference_checkbox_change(use_emo_ref):
        return gr.update(visible=use_emo_ref), gr.update(visible=not use_emo_ref)

    emo_control_method.select(on_method_select,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emo_random,
                 emotion_vector_group,
                 emo_text_group]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )
    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )
    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    save_voice_button.click(
        save_voice,
        inputs=[prompt_audio, voice_name],
        outputs=[save_status]
    )

    refresh_voices_button_top.click(
        refresh_voices_top,
        outputs=[saved_voices_top]
    )

    use_saved_voice_checkbox_top.change(
        use_saved_voice_checkbox_top_change,
        inputs=[use_saved_voice_checkbox_top],
        outputs=[saved_voices_top, prompt_audio]
    )
    
    refresh_emo_reference_button.click(
        refresh_emo_reference_list,
        outputs=[emo_reference_dropdown]
    )
    
    use_emo_reference_checkbox.change(
        use_emo_reference_checkbox_change,
        inputs=[use_emo_reference_checkbox],
        outputs=[emo_reference_dropdown, emo_upload]
    )

    gen_button.click(gen_single,
                     inputs=[emo_control_method, prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text, emo_random, 
                             saved_voices_top, use_saved_voice_checkbox_top,
                             emo_reference_dropdown, use_emo_reference_checkbox,
                             max_text_tokens_per_segment,
                             *advanced_params,
                     ],
                     outputs=[output_audio])

    batch_gen_button.click(gen_batch,
                           inputs=[batch_table],
                           outputs=[batch_output])

    # 添加导入Excel文件的事件处理
    import_excel_button.upload(import_excel_data,
                               inputs=[import_excel_button],
                               outputs=[batch_table])

if __name__ == "__main__":
    # 初始化顶部音色下拉框
    try:
        os.makedirs('voices', exist_ok=True)
        voice_files = [f for f in os.listdir('voices') if f.endswith('.pkl')]
        voice_names = [os.path.splitext(f)[0] for f in voice_files]
        # 更新顶部和底部的音色下拉框
        # 注意：在Gradio中，不能直接设置choices属性，需要通过更新函数来设置
        pass  # 实际的初始化将在应用启动后通过刷新按钮完成
    except Exception as e:
        print(f"初始化音色列表时出错: {e}")
    
    # 初始化情感参考音频下拉框
    try:
        os.makedirs('情感参考', exist_ok=True)
        emo_audio_files = [f for f in os.listdir('情感参考') if f.endswith(('.wav', '.mp3', '.flac'))]
        # 实际的初始化将在应用启动后通过刷新按钮完成
        pass
    except Exception as e:
        print(f"初始化情感参考音频列表时出错: {e}")
    
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)