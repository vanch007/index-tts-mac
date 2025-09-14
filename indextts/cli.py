import os
import sys
import warnings
import soundfile as sf
import torch
# Suppress warnings from tensorflow and other libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
def main():
    import argparse
    parser = argparse.ArgumentParser(description="IndexTTS2 Command Line")
    parser.add_argument("text", type=str, help="Text to be synthesized")
    parser.add_argument("--voice", "-v", type=str, required=True, help="Path to the audio prompt file (wav format)")
    parser.add_argument("--output_path", "-o", type=str, default="gen.wav", help="Path to the output wav file")
    parser.add_argument("--language", "-l", type=str, default="zh", help="Language of the text (zh/en)")
    parser.add_argument("--emo_text", type=str, help="Emotion description text")
    parser.add_argument("--emo_audio", type=str, help="Emotion reference audio file path")
    parser.add_argument("-c", "--config", type=str, default="checkpoints/config.yaml", help="Path to the config file. Default is 'checkpoints/config.yaml'")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Path to the model directory. Default is 'checkpoints'")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 for inference if available")
    parser.add_argument("-f", "--force", action="store_true", default=False, help="Force to overwrite the output file if it exists")
    parser.add_argument("-d", "--device", type=str, default=None, help="Device to run the model on (cpu, cuda, mps)." )
    args = parser.parse_args()
    if len(args.text.strip()) == 0:
        print("ERROR: Text is empty.")
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.voice):
        print(f"Audio prompt file {args.voice} does not exist.")
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist.")
        parser.print_help()
        sys.exit(1)

    output_path = args.output_path
    if os.path.exists(output_path):
        if not args.force:
            print(f"ERROR: Output file {output_path} already exists. Use --force to overwrite.")
            parser.print_help()
            sys.exit(1)
        else:
            os.remove(output_path)
    
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install it first.")
        sys.exit(1)

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda:0"
        elif torch.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
            args.fp16 = False # Disable FP16 on CPU
            print("WARNING: Running on CPU may be slow.")

    from indextts.infer_v2 import IndexTTS2
    tts = IndexTTS2(device=args.device)
    
    # Prepare reference audio
    spk_audio_prompt, _ = sf.read(args.voice)
    spk_audio_prompt = torch.from_numpy(spk_audio_prompt).unsqueeze(0).to(args.device)
    
    # Prepare emotion reference if provided
    emo_audio_prompt = None
    if args.emo_audio:
        emo_audio_prompt, _ = sf.read(args.emo_audio)
        emo_audio_prompt = torch.from_numpy(emo_audio_prompt).unsqueeze(0).to(args.device)

    # Synthesize
    audio = tts.infer(
        text=args.text.strip(),
        spk_audio_prompt=spk_audio_prompt,
        emo_audio_prompt=emo_audio_prompt,
        use_emo_text=args.emo_text is not None,
        emo_text=args.emo_text,
        language=args.language,
        use_fp16=args.fp16
    )

    # Save output
    sf.write(args.output_path, audio.cpu().numpy(), tts.sampling_rate)
    print(f"Audio saved to {args.output_path}")

if __name__ == "__main__":
    main()