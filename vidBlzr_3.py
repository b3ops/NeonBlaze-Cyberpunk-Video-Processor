import random
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision.io import read_video
from tqdm import tqdm
import time
import warnings
import shutil
import re

warnings.filterwarnings("ignore")

def sanitize_filename(filename):
    """Replace special characters in filename with underscores."""
    return re.sub(r'[^\w\-\.]', '_', filename)

def get_video_files(input_dir):
    """Get list of video files in dir."""
    extensions = ('.mp4', '.avi', '.mov', '.mkv')
    return [f for f in os.listdir(input_dir) if f.lower().endswith(extensions)]

def load_font(font_size, font_path=None):
    """Load font once per video, with fallback."""
    if font_path is None:
        font_paths = [
            '/mnt/c/Windows/Fonts/arial.ttf',
            os.path.expanduser('~/progs/fonts/arial.ttf'),
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
        ]
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
    
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
            print(f"Loaded font: {os.path.basename(font_path)} ({font_size}px)")
            return font, font_size, font_path
        except Exception as e:
            print(f"Font load failed ({e}); using default.")
    print("No TTF font found—install with: sudo apt update && sudo apt install fonts-dejavu-core && sudo fc-cache -fv")
    font = ImageFont.load_default()
    return font, font_size, None

def add_watermark(img, text, font, font_size, font_path=None, line_spacing=10, color='cyan'):
    """Add bottom-right watermark with PIL—word-wrap + anti-clip buffers."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    margin = int(min(w, h) * 0.03)
    max_width = w - 4 * margin
    available_height = int(h * 0.12)
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]
        if test_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    if not lines:
        lines = [text]
    
    line_height = font_size + line_spacing
    total_height = len(lines) * line_height - line_spacing
    if total_height > available_height:
        scale_factor = available_height / total_height
        new_font_size = max(48, int(font_size * scale_factor * 0.85))
        print(f"Quote too tall; scaling font to {new_font_size}px")
        if font_path:
            try:
                font = ImageFont.truetype(font_path, new_font_size)
                font_size = new_font_size
            except:
                font = ImageFont.load_default()
        line_height = new_font_size + line_spacing
        lines = []
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]
            if test_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        total_height = len(lines) * line_height - line_spacing
    
    start_y = h - total_height - 3 * margin
    current_y = max(0, start_y)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = max(margin, w - text_w - 3 * margin)
        draw.text((x, current_y), line, fill=color, font=font)
        current_y += line_height
    
    return img

def apply_neon_effect(batch_tensor, device):
    """Apply cyberpunk neon effect: edge glow, vibrant colors, dark background."""
    r, g, b = batch_tensor[:, 0, :, :], batch_tensor[:, 1, :, :], batch_tensor[:, 2, :, :]
    max_val, _ = torch.max(batch_tensor, dim=1, keepdim=True)
    min_val, _ = torch.min(batch_tensor, dim=1, keepdim=True)
    delta = max_val - min_val
    delta = torch.where(delta < 1e-5, torch.ones_like(delta), delta)
    s = delta / (max_val + 1e-5)
    s = s * 1.2
    s = torch.clamp(s, 0, 1)
    batch_tensor = (batch_tensor - min_val) * s + min_val
    batch_tensor = torch.clamp(batch_tensor, 0, 1)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    gray = batch_tensor.mean(dim=1, keepdim=True)
    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    edges = torch.sqrt(edge_x**2 + edge_y**2)
    edges = torch.clamp(edges, 0, 1) * 2.0

    neon_color = torch.tensor([1.0, 0.2, 0.8], device=device).view(1, 3, 1, 1)
    neon_mask = edges * neon_color
    batch_tensor = batch_tensor * 0.6 + neon_mask * 0.4
    batch_tensor = torch.clamp(batch_tensor, 0, 1)

    dark_mask = 1 - edges * 0.5
    batch_tensor = batch_tensor * dark_mask
    batch_tensor = torch.clamp(batch_tensor, 0, 1)

    return batch_tensor

def process_video(video_path, args, device, quote_lines):
    """Process single video: neon effect + watermark, optimized for speed."""
    try:
        frames, _, meta = read_video(video_path, pts_unit='sec')
        if frames.numel() == 0:
            print(f"Skipping {video_path}: No frames.")
            return
    except Exception as e:
        print(f"Skipping {video_path}: {e}")
        return

    orig_h, orig_w = frames.shape[1:3]
    fps = float(meta.get('video_fps', 30.0))  # Convert to float first
    fps = int(fps) if fps.is_integer() else fps  # Use int if whole number
    print(f"Meta: {meta}, FPS type: {type(fps)}, FPS value: {fps}")
    num_frames = frames.shape[0]

    print(f"Processing {video_path}: {num_frames} frames, {orig_w}x{orig_h} @ {fps:.2f}fps")

    target_h = 2160 if args.four_k else args.resize_height
    if target_h > 0:
        scale = target_h / orig_h
        target_w = int(orig_w * scale)
        target_w = (target_w + 1) // 2 * 2  # Ensure even width for libx264
        print(f"Resizing to {target_w}x{target_h}")
    else:
        target_h, target_w = orig_h, orig_w
        target_w = (target_w + 1) // 2 * 2

    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    if (target_h, target_w) != (orig_h, orig_w):
        frames = F.interpolate(frames, size=(target_h, target_w), mode='bilinear', align_corners=False)

    watermark_text = args.watermark_text
    font_size = max(48, int(target_h / 21))
    font, font_size, font_path = load_font(font_size)
    if args.watermark and quote_lines:
        watermark_text = random.choice(quote_lines)

    processed_frames = []
    pbar = tqdm(total=num_frames, desc="Frames")
    for start in range(0, num_frames, args.frame_batch_size):
        end = min(start + args.frame_batch_size, num_frames)
        batch = frames[start:end].to(device)
        with torch.no_grad():
            batch = apply_neon_effect(batch, device)
        batch = batch.cpu() * 255.0
        batch = batch.byte()

        for i in range(batch.shape[0]):
            frame_img = Image.fromarray(batch[i].permute(1, 2, 0).numpy())
            if args.watermark:
                frame_img = add_watermark(frame_img, watermark_text, font, font_size, font_path, color=args.watermark_color)
            processed_frames.append(torch.from_numpy(np.array(frame_img)).permute(2, 0, 1))

        pbar.update(end - start)
        del batch
        torch.cuda.empty_cache()

    pbar.close()
    processed_frames = torch.stack(processed_frames)
    del frames
    torch.cuda.empty_cache()

    base_name = os.path.basename(video_path).rsplit('.', 1)[0]
    sanitized_base_name = sanitize_filename(base_name)
    out_path = os.path.join(args.output_dir, f"blazed_{sanitized_base_name}.mp4")
    frame_dir = os.path.join(args.output_dir, f"blazed_{sanitized_base_name}_frames")
    os.makedirs(frame_dir, exist_ok=True)

    try:
        print(f"Writing video with codec libx264, fps={fps}, type={type(fps)}")
        cmd = (
            f"ffmpeg -y -framerate {fps} -i \"{frame_dir}/frame_%04d.png\" "
            f"-c:v libx264 -preset ultrafast -crf {args.crf} -pix_fmt yuv420p \"{out_path}\""
        )
        for i, frame in enumerate(processed_frames):
            frame_img = Image.fromarray(frame.permute(1, 2, 0).numpy())
            frame_img.save(os.path.join(frame_dir, f"frame_{i:04d}.png"))

        print(f"Running: {cmd}")
        result = os.system(cmd)
        if result == 0:
            print(f"Saved {out_path}")
            shutil.rmtree(frame_dir)
            print(f"Cleaned up {frame_dir}")
        else:
            print(f"FFmpeg failed with exit code {result}; frames saved to {frame_dir}")
    except Exception as e:
        print(f"Video encoding failed ({e}); frames saved to {frame_dir}")
    finally:
        del processed_frames
        torch.cuda.empty_cache()

    return

def blaze_videos(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Blazing videos on {device} (RTX 3050 safe). Frame batch: {args.frame_batch_size}")

    if args.four_k:
        args.resize_height = 2160
        args.frame_batch_size = min(args.frame_batch_size, 2)
        print("4K Mode: Height=2160, batch capped at 2 for VRAM.")

    start_time = time.perf_counter()  # Use perf_counter for accurate timing
    quote_lines = []
    if args.watermark:
        quote_file = os.path.join(os.getcwd(), 'cyberpunk_quotes.txt')
        try:
            with open(quote_file, 'r', encoding='utf-8') as f:
                quote_lines = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(quote_lines)} cyberpunk quotes!")
        except FileNotFoundError:
            print(f"No {quote_file}; using fixed text: {args.watermark_text}")

    video_files = get_video_files(args.input_dir)
    if not video_files:
        print("No videos found.")
        return

    if args.max_videos:
        video_files = video_files[:args.max_videos]
        print(f"Limiting to {args.max_videos} videos.")

    os.makedirs(args.output_dir, exist_ok=True)

    for vfile in video_files:
        process_video(os.path.join(args.input_dir, vfile), args, device, quote_lines)

    elapsed = time.perf_counter() - start_time
    print(f"Blazed {len(video_files)} videos in {elapsed:.2f}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Video Neon Blazer—Cyberpunk Vibes")
    parser.add_argument('--input_dir', required=True, help="Folder of input videos")
    parser.add_argument('--output_dir', default='./blazed_vids', help="Output folder")
    parser.add_argument('--resize_height', type=int, default=0, help="Resize to this height (0=original)")
    parser.add_argument('--frame_batch_size', type=int, default=8, help="Frames per GPU batch (RTX 3050: 8@1080p, 2@4K)")
    parser.add_argument('--crf', type=int, default=23, help="FFmpeg CRF quality (0-51, lower=better quality)")
    parser.add_argument('--4k', dest='four_k', action='store_true', help="4K mode: height=2160, halve batch")
    parser.add_argument('--watermark', action='store_true', help="Add bottom-right watermark")
    parser.add_argument('--watermark_text', default="Blazed by Grok", help="Fixed watermark text")
    parser.add_argument('--watermark_color', default="cyan", help="Watermark color (e.g., cyan, magenta, blue)")
    parser.add_argument('--max_videos', type=int, help="Max videos to process")
    args = parser.parse_args()
    blaze_videos(args)