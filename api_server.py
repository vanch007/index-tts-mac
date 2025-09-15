import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from contextlib import asynccontextmanager
import torch
import psutil
import sys
import gc

from indextts.infer_v2 import IndexTTS2

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            p = psutil.Process(proc.info['pid'])
            for conn in p.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"Port {port} is being used by process {proc.info['pid']} ({proc.info['name']}). Terminating it.")
                    p.kill()
                    p.wait()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def get_process_memory_usage():
    """获取当前Python进程的总内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # MB

class TTSRequest(BaseModel):
    text: str
    spk_audio_prompt: str
    output_path: str = "outputs/api_gen.wav"
    emo_audio_prompt: str | None = None
    emo_alpha: float = 1.0
    use_emo_text: bool = False
    emo_text: str | None = None
    use_random: bool = False
    interval_silence: int = 200
    verbose: bool = False
    max_text_tokens_per_segment: int = 60

models = {}
failure_count = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model
    print("Loading model...")
    
    # Autodetect device and FP16 support
    if torch.cuda.is_available():
        device = "cuda"
        use_fp16 = args.fp16
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        device = "mps"
        use_fp16 = False  # 在MPS上禁用FP16以减少内存使用
    else:
        device = "cpu"
        use_fp16 = False

    print(f"Using device: {device}, FP16: {use_fp16}")

    models['tts'] = IndexTTS2(
        cfg_path="checkpoints/config.yaml", 
        model_dir="checkpoints", 
        use_fp16=use_fp16,
        device=device
    )
    print("Model loaded.")
    yield
    # Clean up the models
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "IndexTTS API is running."}

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Generate speech from text using a speaker prompt.
    """
    global failure_count
    tts_model = models.get('tts')
    if not tts_model:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # Ensure the speaker audio prompt exists
    if not os.path.exists(request.spk_audio_prompt):
        raise HTTPException(status_code=400, detail=f"Speaker prompt audio file not found: {request.spk_audio_prompt}")

    try:
        print(f">> Process memory usage before inference: {get_process_memory_usage():.2f} MB")
        
        # 检查当前进程内存使用情况，如果超过30GB则重启服务
        current_memory = get_process_memory_usage()
        if current_memory > 30 * 1024:  # 30GB = 30 * 1024 MB
            print(f">> Process memory usage ({current_memory:.2f} MB) exceeds 30GB. Restarting server to free memory...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(request.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Perform inference
        tts_model.infer(
            spk_audio_prompt=request.spk_audio_prompt,
            text=request.text,
            output_path=request.output_path,
            emo_audio_prompt=request.emo_audio_prompt,
            emo_alpha=request.emo_alpha,
            use_emo_text=request.use_emo_text,
            emo_text=request.emo_text,
            use_random=request.use_random,
            interval_silence=request.interval_silence,
            verbose=request.verbose,
            max_text_tokens_per_segment=request.max_text_tokens_per_segment,
        )

        if os.path.exists(request.output_path):
            failure_count = 0
            print(f">> Process memory usage after inference: {get_process_memory_usage():.2f} MB")
            return FileResponse(request.output_path, media_type="audio/wav", filename=os.path.basename(request.output_path))
        else:
            failure_count += 1
            if failure_count >= 3:
                print("Three consecutive errors occurred. Restarting server...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
            raise HTTPException(status_code=500, detail="Failed to generate audio file.")

    except Exception as e:
        failure_count += 1
        print(f"An error occurred during TTS inference: {e}")
        import traceback
        traceback.print_exc()
        if failure_count >= 3:
            print("Three consecutive errors occurred. Restarting server...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理PyTorch缓存，释放GPU/CPU内存
        if tts_model:
            if tts_model.device == "mps":
                torch.mps.empty_cache()
            elif tts_model.device == "cuda":
                torch.cuda.empty_cache()
            # 清理模型内部缓存
            tts_model.clear_cache()
            
            # 强制垃圾回收
            gc.collect()
            
        print(f">> Process memory usage after cleanup: {get_process_memory_usage():.2f} MB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IndexTTS API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the API server on.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API server on.")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")  # 默认禁用FP16
    args = parser.parse_args()
    kill_process_on_port(args.port)
    print("Starting API server...")
    uvicorn.run(app, host=args.host, port=args.port)