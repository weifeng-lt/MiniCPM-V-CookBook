"""
MiniCPMO C++ 后端 HTTP 服务器（统一版：支持单工/双工模式切换）
通过 HTTP 接口包装 C++ llama-server 的功能，对外提供与 Python 版本一致的 API

模式说明：
- 单工模式 (duplex_mode=False): 使用"延迟一拍"机制，每个 round 有独立目录
- 双工模式 (duplex_mode=True): 直接转发 prefill，全局 WAV 计数器，支持并行处理
"""
import os
import sys
import base64
import json
import asyncio
import io
import librosa
import numpy as np
import soundfile as sf
from PIL import Image
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import subprocess
import signal
import time
import httpx
import socket
import requests
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import uuid
import shutil

# ====================== 配置 ======================
# 注意: Python Token2Wav 现在由 C++ 程序直接通过 subprocess 调用
# C++ t2w_thread 会启动 Python 服务进程处理 audio tokens 并生成 WAV 文件
# HTTP server 只需要读取 tts_wav 目录下的 WAV 文件即可（和之前一样）
# C++ 服务器配置
CPP_SERVER_HOST = "127.0.0.1"
# C++ 端口动态计算：Python 端口 + 10000，例如 8100 -> 18100
CPP_SERVER_PORT = None  # 在 lifespan 中根据 Python 端口设置
CPP_SERVER_URL = None   # 在 lifespan 中根据 Python 端口设置

# 模型配置 - 必须通过环境变量或命令行参数指定
# 🔧 [本地部署] 必须设置 LLAMACPP_ROOT 和 MODEL_DIR
LLAMACPP_ROOT = os.environ.get("LLAMACPP_ROOT", "")  # 必须指定
DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "")  # 必须指定
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "")  # 如果不指定，自动从 MODEL_DIR 查找
# GPU 设备：macOS 使用 Metal（设为空字符串），Linux/CUDA 使用设备ID
DEFAULT_GPU_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
DEFAULT_CTX_SIZE = int(os.environ.get("CTX_SIZE", "8192"))
DEFAULT_N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "99"))

# 固定音色文件（用于 voice cloning）
FIXED_TIMBRE_PATH = os.environ.get("REF_AUDIO", "")  # 默认在启动时从 LLAMACPP_ROOT 推导

# 视觉编码器后端: "metal"(默认，GPU) 或 "coreml"(ANE加速，macOS专用)
VISION_BACKEND = os.environ.get("VISION_BACKEND", "metal")

# Token2Wav device: "gpu:1"(默认，GPU加速) 或 "cpu"(节省GPU显存，适合16GB内存机型)
TOKEN2WAV_DEVICE = os.environ.get("TOKEN2WAV_DEVICE", "gpu:1")


def auto_detect_llm_model(model_dir: str) -> str:
    """自动从模型目录检测 LLM GGUF 文件
    
    优先级：Q4_K_M > Q8_0 > F16 > 其他 .gguf 文件
    """
    if not model_dir or not os.path.isdir(model_dir):
        return ""
    
    # 按优先级排序的模式
    priority_patterns = [
        "*Q4_K_M*.gguf",
        "*Q4_K_S*.gguf", 
        "*Q8_0*.gguf",
        "*Q5_K_M*.gguf",
        "*F16*.gguf",
    ]
    
    import glob
    
    # 按优先级查找
    for pattern in priority_patterns:
        matches = glob.glob(os.path.join(model_dir, pattern))
        # 排除子目录中的文件，只取根目录的
        root_matches = [m for m in matches if os.path.dirname(m) == model_dir]
        if root_matches:
            # 返回文件名（不含路径）
            return os.path.basename(sorted(root_matches)[0])
    
    # 如果优先模式都没找到，查找任意 .gguf 文件
    all_gguf = glob.glob(os.path.join(model_dir, "*.gguf"))
    if all_gguf:
        # 排除明显不是 LLM 的文件（如 audio, vision, tts）
        llm_candidates = [f for f in all_gguf 
                         if not any(x in os.path.basename(f).lower() 
                                   for x in ['audio', 'vision', 'tts', 'projector'])]
        if llm_candidates:
            return os.path.basename(sorted(llm_candidates)[0])
    
    return ""

# 临时文件目录
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_streaming_prefill")

# C++ llama-server 输出目录 (tools/omni/output)
# 🔧 [多实例支持] 默认值，运行时可通过 --output-dir 参数覆盖
DEFAULT_CPP_OUTPUT_DIR = os.path.join(LLAMACPP_ROOT, "tools/omni/output")
CPP_OUTPUT_DIR = DEFAULT_CPP_OUTPUT_DIR  # 运行时会被替换为实际值

# 服务注册配置（默认为本机 IP:8025 端口，设为空则不注册）
# 🔧 [本地联调] 动态获取本机 IP，注册到后端服务 8025 端口（避免 macOS 8021 端口冲突）
def _get_default_register_url():
    """获取默认注册地址（本机 IP:8025）"""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
        return f"http://{local_ip}:8025"
    except:
        return "http://127.0.0.1:8025"

REGISTER_URL = os.environ.get("REGISTER_URL", _get_default_register_url())

# ====================== 全局状态 ======================
cpp_server_process: Optional[subprocess.Popen] = None
current_msg_type: Optional[int] = None  # 1=audio, 2=video/omni
current_duplex_mode: bool = False  # 是否启用双工模式
current_high_quality_mode: bool = False  # 是否启用高清模式（max_slice_nums=2）
current_high_fps_mode: bool = False  # 是否启用高刷模式（1秒5帧stack）
current_active_session_id: Optional[str] = None
current_request_counter: int = 0
current_round_number: int = 0
session_lock = threading.Lock()
model_state_initialized: bool = False
pending_prefill_data: Optional[dict] = None
is_breaking: bool = False  # break 标志：为 True 时中间层停止向前端发送数据
health_server_thread: Optional[threading.Thread] = None

# 🔧 [System Prompt 缓存] 用于极速恢复时判断是否需要重新 prefill
current_system_prompt_prefix: Optional[str] = None
current_system_prompt_suffix: Optional[str] = None

# 🔧 [高刷模式] 子图缓存：按 image_audio_id 分组存储（frame_index 1-4 的子图）
# key: image_audio_id, value: {frame_index: PIL.Image}
# 注意：主图（frame_index=0）立即处理，不缓存
high_fps_subimage_cache: Dict[int, Dict[int, Image.Image]] = {}
high_fps_cache_lock = threading.Lock()
# 🔧 [高刷模式] 待处理音频缓存：当音频先于子图到达时暂存
# key: image_audio_id, value: (audio_np, sr, audio_path)
high_fps_pending_audio: Dict[int, tuple] = {}
high_fps_audio_lock = threading.Lock()

# 🔧 [双工模式] 全局 WAV 发送计数器（跨 generate 调用保持状态）
global_sent_wav_count: int = 0
# 全局文本行计数器（跨 generate 调用保持状态，用于累积解析 llm_text.txt）
global_parsed_line_count: int = 0
# 全局文本列表（跨 generate 调用保持状态）
global_parsed_texts: list = []
# 全局文本发送索引（已发送的文本数量，用于顺序消费文本）
global_text_send_idx: int = 0
# 🔧 [修复回文] 全局已发送 WAV 文件名集合（跨 generate 调用保持状态）
global_sent_wav_files: set = set()

# WAV 发送时序日志
WAV_TIMING_LOG_PATH = os.path.join(os.path.dirname(__file__), "wav_timing.log")
wav_timing_log_file: Optional[Any] = None  # 全局日志文件句柄
last_wav_send_time: Optional[float] = None  # 上一次 WAV 发送时间

# HTTP 客户端
http_client: Optional[httpx.AsyncClient] = None

# ====================== 显存监控配置 ======================
GPU_MEMORY_THRESHOLD_MB = 2000  # 显存剩余低于此值时触发重启 (MB)
# 🔧 [本地部署] 默认禁用显存检查和自动重启功能（生产环境可通过环境变量启用）
# 设置 GPU_MEMORY_CHECK=1 启用显存监控和自动重启
import platform
GPU_CHECK_ENABLED = os.environ.get("GPU_MEMORY_CHECK", "0") == "1"
cpp_restart_lock = threading.Lock()  # 重启锁，防止并发重启
cpp_restarting = False  # 🔧 [修复] 正在重启标志，防止重启期间接收新请求


def get_gpu_memory_info() -> dict:
    """获取 GPU 显存信息
    
    Returns:
        dict: {
            'total_mb': 总显存 (MB),
            'used_mb': 已用显存 (MB),
            'free_mb': 剩余显存 (MB),
            'utilization': 使用率 (0-100)
        }
        如果获取失败返回 None
    """
    try:
        import subprocess
        # 使用 nvidia-smi 获取显存信息
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", 
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                total = int(parts[0].strip())
                used = int(parts[1].strip())
                free = int(parts[2].strip())
                return {
                    'total_mb': total,
                    'used_mb': used,
                    'free_mb': free,
                    'utilization': round(used / total * 100, 1) if total > 0 else 0
                }
    except Exception as e:
        print(f"[显存监控] 获取显存信息失败: {e}", flush=True)
    return None


def check_gpu_memory_and_restart_if_needed() -> bool:
    """检查 GPU 显存，如果剩余不足则重启 C++ 服务器
    
    Returns:
        bool: True 如果执行了重启，False 如果不需要重启
    """
    global cpp_server_process, model_state_initialized
    
    if not GPU_CHECK_ENABLED:
        return False
    
    mem_info = get_gpu_memory_info()
    if mem_info is None:
        return False
    
    free_mb = mem_info['free_mb']
    print(f"[显存监控] 剩余显存: {free_mb} MB (阈值: {GPU_MEMORY_THRESHOLD_MB} MB)", flush=True)
    
    if free_mb < GPU_MEMORY_THRESHOLD_MB:
        print(f"[显存监控] ⚠️ 显存不足 ({free_mb} MB < {GPU_MEMORY_THRESHOLD_MB} MB)，准备重启 C++ 服务器...", flush=True)
        
        with cpp_restart_lock:
            # 再次检查，避免重复重启
            mem_info = get_gpu_memory_info()
            if mem_info and mem_info['free_mb'] >= GPU_MEMORY_THRESHOLD_MB:
                print(f"[显存监控] 显存已恢复，取消重启", flush=True)
                return False
            
            try:
                restart_cpp_server()
                return True
            except Exception as e:
                print(f"[显存监控] 重启失败: {e}", flush=True)
                return False
    
    return False


def restart_cpp_server():
    """重启 C++ llama-server（保持相同配置）"""
    global cpp_server_process, model_state_initialized, current_msg_type
    global current_round_number, global_sent_wav_count, global_parsed_line_count
    global global_parsed_texts, global_text_send_idx, global_sent_wav_files
    global current_duplex_mode, cpp_restarting
    
    print("=" * 60, flush=True)
    print("[重启] 开始重启 C++ llama-server...", flush=True)
    print("=" * 60, flush=True)
    
    # 🔧 [修复] 设置重启标志，阻止新请求
    cpp_restarting = True
    
    # 保存当前模式（重启后需要恢复）
    saved_duplex_mode = current_duplex_mode
    saved_msg_type = current_msg_type if current_msg_type else 2  # 默认 omni 模式
    
    # 1. 停止当前 C++ 服务器
    stop_cpp_server()
    
    # 2. 等待进程完全退出
    time.sleep(2)
    
    # 3. 清理 output 目录
    reset_output_dir()
    
    # 4. 重置状态
    model_state_initialized = False
    current_msg_type = None
    current_round_number = 0
    global_sent_wav_count = 0
    global_parsed_line_count = 0
    global_parsed_texts = []
    global_text_send_idx = 0
    global_sent_wav_files = set()
    
    # 5. 重新启动 C++ 服务器
    start_cpp_server(
        model_dir=app.state.model_dir,
        gpu_devices=app.state.gpu_devices,
        port=CPP_SERVER_PORT
    )
    
    print("[重启] C++ llama-server 重启完成", flush=True)
    
    # 6. 🔧 重新初始化 omni context（解决重启后 "omni context not initialized" 的问题）
    try:
        print("[重启] 重新初始化 omni context...", flush=True)
        
        model_dir = app.state.model_dir
        # TTS 模型在 tts/ 目录，Token2Wav 模型在 token2wav-gguf/ 目录
        tts_bin_dir = os.path.join(model_dir, "tts")
        
        cpp_request = {
            "media_type": saved_msg_type,      # 恢复之前的模式
            "use_tts": True,
            "duplex_mode": saved_duplex_mode,
            "model_dir": model_dir,
            "tts_bin_dir": tts_bin_dir,
            "tts_gpu_layers": 100,
            "token2wav_device": TOKEN2WAV_DEVICE,
            "output_dir": CPP_OUTPUT_DIR,
        }
        
        # 视觉编码器后端
        cpp_request["vision_backend"] = VISION_BACKEND
        
        # 使用固定音色文件
        if os.path.exists(FIXED_TIMBRE_PATH):
            cpp_request["voice_audio"] = FIXED_TIMBRE_PATH
        
        # 使用同步 requests（因为当前在后台线程中）
        resp = requests.post(
            f"{CPP_SERVER_URL}/v1/stream/omni_init",
            json=cpp_request,
            timeout=60.0
        )
        
        if resp.status_code == 200:
            model_state_initialized = True
            current_msg_type = saved_msg_type
            current_duplex_mode = saved_duplex_mode
            print(f"[重启] omni context 初始化成功: {resp.json()}", flush=True)
        else:
            print(f"[重启] omni context 初始化失败: {resp.text}", flush=True)
    except Exception as e:
        print(f"[重启] omni context 初始化异常: {e}", flush=True)
    finally:
        # 🔧 [修复] 无论成功失败，都清除重启标志
        cpp_restarting = False
    
    print("=" * 60, flush=True)


def stack_images(images: List[Image.Image]) -> Image.Image:
    """将多张图片 stack 成一张
    
    Stack 策略（根据图片数量）：
    - 1张：直接返回
    - 2张：横向拼接 (1x2)
    - 3张：2x2 布局，右下角空白
    - 4张：2x2 布局
    
    Args:
        images: PIL Image 列表
        
    Returns:
        拼接后的单张 PIL Image
    """
    if len(images) == 0:
        raise ValueError("images 列表不能为空")
    if len(images) == 1:
        return images[0]
    
    # 获取单张图片尺寸（假设所有图片尺寸相同）
    w, h = images[0].size
    
    if len(images) == 2:
        # 横向拼接 1x2
        result = Image.new('RGB', (w * 2, h))
        result.paste(images[0], (0, 0))
        result.paste(images[1], (w, 0))
    elif len(images) == 3:
        # 2x2 布局，右下角空白（黑色）
        result = Image.new('RGB', (w * 2, h * 2), (0, 0, 0))
        result.paste(images[0], (0, 0))
        result.paste(images[1], (w, 0))
        result.paste(images[2], (0, h))
    else:  # 4张或更多（取前4张）
        # 2x2 布局
        result = Image.new('RGB', (w * 2, h * 2))
        result.paste(images[0], (0, 0))
        result.paste(images[1], (w, 0))
        result.paste(images[2], (0, h))
        if len(images) >= 4:
            result.paste(images[3], (w, h))
    
    return result


class HealthCheckHandler(BaseHTTPRequestHandler):
    """独立的健康检查和打断HTTP处理器，运行在单独线程中，不受主线程推理任务阻塞
    
    支持的接口：
    - GET /health - 健康检查
    - POST /omni/break - 打断当前生成（快速响应，不阻塞）
    - POST /omni/stop - 停止会话（快速响应，不阻塞）
    """
    
    def log_message(self, format, *args):
        """禁用默认日志输出，避免干扰主程序日志"""
        pass
    
    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = json.dumps({
                "status": "healthy",
                "message": "服务正常 (C++ backend)",
                "backend": "cpp"
            })
            self.wfile.write(response.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """处理 POST 请求 - 打断和停止"""
        global is_breaking, CPP_SERVER_URL
        
        if self.path == "/omni/break":
            # 快速打断 - 在独立线程中设置 break 标志并调用 C++ break 接口
            print("======= [独立线程] 收到快速打断指令 =======", flush=True)
            
            # 【关键】立即设置 break 标志，让 generate_stream 停止向前端发送数据
            is_breaking = True
            print("[独立线程] is_breaking 已设置为 True，中间层将停止发送数据", flush=True)
            
            # 调用 C++ 服务器的 break 接口
            cpp_break_success = False
            if CPP_SERVER_URL:
                try:
                    break_resp = requests.post(
                        f"{CPP_SERVER_URL}/v1/stream/break",
                        json={"reason": "user_interrupt_from_health_thread"},
                        timeout=5.0
                    )
                    if break_resp.status_code == 200:
                        print(f"[独立线程] C++ 生成已中止: {break_resp.json()}", flush=True)
                        cpp_break_success = True
                    else:
                        print(f"[独立线程] C++ break 调用失败: {break_resp.status_code}", flush=True)
                except Exception as e:
                    print(f"[独立线程] C++ break 调用异常: {e}", flush=True)
            
            response = json.dumps({
                "success": True,
                "message": "当前轮对话已打断",
                "state": "break",
                "cpp_break": cpp_break_success
            })
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response.encode())
            
        elif self.path == "/omni/stop":
            # 快速停止 - 设置 break 标志并调用 C++ break 接口
            print("======= [独立线程] 收到快速停止指令 =======", flush=True)
            
            # 设置 break 标志
            is_breaking = True
            print("[独立线程] is_breaking 已设置为 True (stop)", flush=True)
            
            # 调用 C++ 服务器的 break 接口
            cpp_break_success = False
            if CPP_SERVER_URL:
                try:
                    break_resp = requests.post(
                        f"{CPP_SERVER_URL}/v1/stream/break",
                        json={"reason": "session_stop_from_health_thread"},
                        timeout=5.0
                    )
                    if break_resp.status_code == 200:
                        print(f"[独立线程] C++ 生成已中止 (stop): {break_resp.json()}", flush=True)
                        cpp_break_success = True
                    else:
                        print(f"[独立线程] C++ break 调用失败 (stop): {break_resp.status_code}", flush=True)
                except Exception as e:
                    print(f"[独立线程] C++ break 调用异常 (stop): {e}", flush=True)
            
            response = json.dumps({
                "success": True,
                "message": "会话已停止",
                "state": "session_stop",
                "cpp_break": cpp_break_success
            })
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response.encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        """处理CORS预检请求"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()


def start_health_server(port: int):
    """在独立线程中启动健康检查和打断服务器
    
    该服务器运行在独立线程中，不受主线程推理任务阻塞。
    支持快速响应打断请求，即使模型正在生成中也能立即处理。
    
    支持的接口：
    - GET  /health     - 健康检查
    - POST /omni/break - 快速打断（推理期间可用）
    - POST /omni/stop  - 快速停止（推理期间可用）
    """
    health_port = port + 1
    server = HTTPServer(("0.0.0.0", health_port), HealthCheckHandler)
    print(f"独立健康检查/打断服务器已启动: http://0.0.0.0:{health_port}", flush=True)
    print(f"  - GET  /health     - 健康检查", flush=True)
    print(f"  - POST /omni/break - 快速打断", flush=True)
    print(f"  - POST /omni/stop  - 快速停止", flush=True)
    server.serve_forever()


def get_local_ip():
    """获取本机 IP 地址"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 连接到一个不会真的通信的公网地址
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def register_service_node(port: int, duplex_mode: bool):
    """注册服务节点到调度中心（如果配置了 REGISTER_URL）"""
    if not REGISTER_URL:
        print("跳过服务注册（未配置 REGISTER_URL）", flush=True)
        return
    
    try:
        url = f"{REGISTER_URL}/api/inference/register"
        local_ip = get_local_ip()
        # 根据 duplex_mode 设置 model_type
        model_type = "duplex" if duplex_mode else "simplex"
        data = {
            "ip": local_ip,
            "port": port,
            "model_port": port,
            "model_type": model_type,
            "session_type": "release",  # 标记为 C++ 后端
            "service_name": "o45-cpp",
        }
        print(f"正在注册服务节点: url={url}, data={data}", flush=True)
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 200:
            print(f"服务节点注册成功: {response.text}", flush=True)
        else:
            print(f"服务节点注册失败: HTTP {response.status_code}, 响应: {response.text}", flush=True)
    except Exception as e:
        import traceback
        print(f"服务节点注册异常: {e}", flush=True)
        traceback.print_exc()


def reset_output_dir():
    """启动时重置 output 目录（rm -rf + mkdir）"""
    if os.path.exists(CPP_OUTPUT_DIR):
        try:
            shutil.rmtree(CPP_OUTPUT_DIR)
            print(f"[启动清理] 已删除 output 目录: {CPP_OUTPUT_DIR}", flush=True)
        except Exception as e:
            print(f"[启动清理] 删除 output 目录失败: {e}", flush=True)
    
    try:
        os.makedirs(CPP_OUTPUT_DIR, exist_ok=True)
        print(f"[启动清理] 已创建 output 目录: {CPP_OUTPUT_DIR}", flush=True)
    except Exception as e:
        print(f"[启动清理] 创建 output 目录失败: {e}", flush=True)


def clear_output_subfolders():
    """清空 output 目录下每个子文件夹的内容，但保留一级子文件夹本身"""
    if not os.path.exists(CPP_OUTPUT_DIR):
        print(f"[清空输出] output 目录不存在: {CPP_OUTPUT_DIR}", flush=True)
        return
    
    cleared_count = 0
    for item in os.listdir(CPP_OUTPUT_DIR):
        item_path = os.path.join(CPP_OUTPUT_DIR, item)
        if os.path.isdir(item_path):
            # 清空子文件夹内容
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                try:
                    if os.path.isdir(sub_item_path):
                        shutil.rmtree(sub_item_path)
                    else:
                        os.remove(sub_item_path)
                    cleared_count += 1
                except Exception as e:
                    print(f"[清空输出] 删除失败 {sub_item_path}: {e}", flush=True)
    
    print(f"[清空输出] 已清空 {CPP_OUTPUT_DIR} 下的子文件夹内容 (删除 {cleared_count} 项)", flush=True)


def start_cpp_server(model_dir: str, gpu_devices: str, port: int):
    """启动 C++ llama-server"""
    global cpp_server_process
    
    # 构建启动命令
    llamacpp_root = LLAMACPP_ROOT
    
    # 查找 llama-server 可执行文件
    # 优先使用 build/bin，回退到 preset 目录
    import platform
    server_bin = None
    candidates = [
        os.path.join(llamacpp_root, "build/bin/llama-server"),              # Linux/macOS default
        os.path.join(llamacpp_root, "build/bin/Release/llama-server.exe"),  # Windows MSVC
        os.path.join(llamacpp_root, "build/bin/llama-server.exe"),          # Windows other
    ]
    if platform.system() == "Darwin":
        candidates.append(os.path.join(llamacpp_root, "build-arm64-apple-clang-release/bin/llama-server"))
    elif platform.system() != "Windows":
        candidates.append(os.path.join(llamacpp_root, "build-x64-linux-cuda-release/bin/llama-server"))
    for c in candidates:
        if os.path.exists(c):
            server_bin = c
            break
    if server_bin is None:
        server_bin = candidates[0]  # fallback for error message
    
    # model_dir 可以是绝对路径或相对路径
    if os.path.isabs(model_dir):
        model_path = os.path.join(model_dir, DEFAULT_LLM_MODEL)
    else:
        model_path = os.path.join(llamacpp_root, model_dir, DEFAULT_LLM_MODEL)
    
    if not os.path.exists(server_bin):
        raise RuntimeError(f"C++ server binary not found: {server_bin}")
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")
    
    # 设置环境变量
    env = os.environ.copy()
    
    # 🔧 [跨平台] 根据系统设置不同的环境变量
    if platform.system() == "Darwin":  # macOS
        # macOS 使用 Metal，不需要 CUDA 环境变量
        # 设置 DYLD_LIBRARY_PATH 以找到动态库
        dyld_paths = [
            os.path.dirname(server_bin),  # 使用实际的 server_bin 目录
            env.get('DYLD_LIBRARY_PATH', '')
        ]
        env["DYLD_LIBRARY_PATH"] = ":".join(p for p in dyld_paths if p)
        print(f"Platform: macOS (Metal)", flush=True)
        print(f"DYLD_LIBRARY_PATH={env.get('DYLD_LIBRARY_PATH', '')[:200]}", flush=True)
    else:  # Linux with CUDA
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices
        # 使用 CUDA 库路径
        cuda_env_path = os.environ.get("CUDA_LIB_PATH", "/usr/local/cuda/lib64")
        cuda_lib_paths = [
            cuda_env_path,
            llamacpp_root + "/build/bin",  # libggml-cuda.so, libomni.so 等
            "/usr/lib/x86_64-linux-gnu",
            env.get('LD_LIBRARY_PATH', '')
        ]
        env["LD_LIBRARY_PATH"] = ":".join(p for p in cuda_lib_paths if p)
        print(f"Platform: Linux (CUDA)", flush=True)
        print(f"CUDA_VISIBLE_DEVICES={gpu_devices}", flush=True)
        print(f"LD_LIBRARY_PATH={env['LD_LIBRARY_PATH'][:300]}", flush=True)
    
    # 启动时指定 --model，omni_init 会复用已加载的模型
    cmd = [
        server_bin,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--model", model_path,
        "--ctx-size", str(DEFAULT_CTX_SIZE),
        "--n-gpu-layers", str(DEFAULT_N_GPU_LAYERS),
        "--repeat-penalty", "1.05",
        "--temp", "0.7",
    ]
    
    print(f"启动 C++ llama-server: {' '.join(cmd)}", flush=True)
    
    # 启动进程
    cpp_server_process = subprocess.Popen(
        cmd,
        env=env,
        cwd=llamacpp_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )
    
    # 启动日志读取线程
    def log_reader():
        try:
            for line in cpp_server_process.stdout:
                print(f"[CPP] {line.rstrip()}", flush=True)
        except Exception as e:
            print(f"[CPP log_reader] 异常: {e}", flush=True)
    
    log_thread = threading.Thread(target=log_reader, daemon=True)
    log_thread.start()
    
    # 等待服务器启动
    max_wait = 180
    for i in range(max_wait):
        try:
            resp = requests.get(f"http://{CPP_SERVER_HOST}:{port}/health", timeout=2)
            if resp.status_code == 200:
                print(f"C++ llama-server 启动成功 (等待 {i+1} 秒)", flush=True)
                return True
        except:
            pass
        time.sleep(1)
    
    raise RuntimeError(f"C++ llama-server 启动超时 ({max_wait}秒)")


def stop_cpp_server():
    """停止 C++ llama-server"""
    global cpp_server_process
    if cpp_server_process:
        print("停止 C++ llama-server...", flush=True)
        cpp_server_process.terminate()
        try:
            cpp_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cpp_server_process.kill()
        cpp_server_process = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global http_client, health_server_thread, CPP_SERVER_PORT, CPP_SERVER_URL
    
    # 动态计算 C++ 端口：Python 端口 + 10000
    CPP_SERVER_PORT = app.state.port + 10000
    CPP_SERVER_URL = f"http://{CPP_SERVER_HOST}:{CPP_SERVER_PORT}"
    print(f"C++ 服务器端口: {CPP_SERVER_PORT} (Python 端口 {app.state.port} + 10000)", flush=True)
    print(f"显存监控: {'启用' if GPU_CHECK_ENABLED else '禁用'} (设置 GPU_MEMORY_CHECK=1 启用)", flush=True)
    
    # 启动健康检查服务器
    health_server_thread = threading.Thread(
        target=start_health_server,
        args=(app.state.port,),
        daemon=True
    )
    health_server_thread.start()
    
    # 创建临时目录（启动时先清空旧数据再创建）
    import shutil
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 启动时清理 output 目录
    reset_output_dir()
    
    # 启动 C++ 服务器
    print("正在启动 C++ llama-server...", flush=True)
    try:
        start_cpp_server(
            model_dir=app.state.model_dir,
            gpu_devices=app.state.gpu_devices,
            port=CPP_SERVER_PORT
        )
    except Exception as e:
        print(f"C++ 服务器启动失败: {e}", flush=True)
        raise
    
    # 创建 HTTP 客户端
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    
    # 🔧 [预初始化] Server 启动时就初始化所有模块（LLM+TTS+APM+Python T2W）
    # 这样用户调用 /omni/init_sys_prompt 时就不需要等待 ~12s 了
    print("正在预初始化 omni context（TTS + APM + Python T2W）...", flush=True)
    try:
        model_dir = app.state.model_dir
        # TTS 模型在 tts/ 目录
        tts_bin_dir = os.path.join(model_dir, "tts")
        
        pre_init_request = {
            "media_type": 2,  # 🔧 [修复] 使用 omni 模式预初始化，这样 VPM 也会被加载
            "use_tts": True,
            "duplex_mode": app.state.default_duplex_mode,
            "model_dir": model_dir,
            "tts_bin_dir": tts_bin_dir,
            "tts_gpu_layers": 100,
            "token2wav_device": TOKEN2WAV_DEVICE,
            "output_dir": CPP_OUTPUT_DIR,
        }
        
        # 视觉编码器后端
        pre_init_request["vision_backend"] = VISION_BACKEND
        
        # 使用固定音色文件进行预初始化
        if os.path.exists(FIXED_TIMBRE_PATH):
            pre_init_request["voice_audio"] = FIXED_TIMBRE_PATH
        
        pre_init_resp = await http_client.post(
            f"{CPP_SERVER_URL}/v1/stream/omni_init",
            json=pre_init_request,
            timeout=120.0  # 预初始化可能需要较长时间
        )
        
        if pre_init_resp.status_code == 200:
            global model_state_initialized, current_duplex_mode, current_msg_type
            model_state_initialized = True
            current_duplex_mode = app.state.default_duplex_mode
            current_msg_type = 2  # 🔧 [修复] omni 模式，支持 audio 和视频
            print(f"预初始化成功: {pre_init_resp.json()}", flush=True)
        else:
            print(f"预初始化失败（不影响后续使用）: {pre_init_resp.text}", flush=True)
    except Exception as e:
        print(f"预初始化异常（不影响后续使用）: {e}", flush=True)
    
    print("MiniCPMO C++ HTTP 服务器初始化完成", flush=True)
    
    # 注册服务节点（使用默认模式）
    try:
        register_service_node(port=app.state.port, duplex_mode=app.state.default_duplex_mode)
    except Exception as e:
        print(f"服务节点注册失败: {e}", flush=True)
    
    try:
        yield
    finally:
        # 关闭 HTTP 客户端
        if http_client:
            await http_client.aclose()
        # 停止 C++ 服务器
        stop_cpp_server()


app = FastAPI(title="MiniCPMO C++ HTTP Server (Unified)", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================== 请求模型 ======================
class InitSysPromptRequest(BaseModel):
    media_type: Optional[str] = None  # "audio" 或 "omni"
    duplex_mode: Optional[bool] = None  # 是否启用双工模式（None 表示使用默认值）
    high_quality_mode: Optional[bool] = False  # 🔧 [高清模式] 启用图片切片 (max_slice_nums=2)
    high_fps_mode: Optional[bool] = False  # 🔧 [高刷模式] 1秒5帧 stack
    language: Optional[str] = "zh"  # 🔧 [语言切换] "zh" 中文, "en" 英文
    system_prompt_prefix: Optional[str] = None  # 🔧 [自定义系统提示词前缀] 插入在参考音频之前
    system_prompt_suffix: Optional[str] = None  # 🔧 [自定义系统提示词后缀] 插入在参考音频之后

class StreamingPrefillRequest(BaseModel):
    audio: Optional[str] = None  # base64编码的音频
    image: Optional[str] = None  # base64编码的图片
    text: Optional[str] = None  # 🔧 [文本输入] 用户文本输入（支持半双工模式下的文字输入）
    # 🔧 [高刷模式] 图片按 image_audio_id 分组
    image_audio_id: Optional[int] = None  # 图片音频关联ID（用于标记同一组音频和图片）
    frame_index: Optional[int] = None  # 当前帧索引 (0=主图, 1-4=子图用于stack)
    max_slice_nums: Optional[int] = None
    session_id: Optional[str] = None
    is_last_chunk: bool = False


# ====================== API 端点 ======================
@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "message": "服务正常 (C++ backend)",
        "backend": "cpp",
        "duplex_mode": current_duplex_mode
    }


@app.post("/omni/stop")
async def omni_stop(session_id: Optional[str] = None):
    """会话停止（中止当前生成，但保留 KV cache 和会话状态）"""
    global current_active_session_id, current_request_counter, current_round_number
    global model_state_initialized, pending_prefill_data, is_breaking
    global wav_timing_log_file, last_wav_send_time
    global global_sent_wav_count, global_parsed_line_count, global_parsed_texts, global_text_send_idx, global_sent_wav_files
    
    print("======= 收到会话停止指令 =======", flush=True)
    
    stopped_session_id = current_active_session_id
    
    # 调用 C++ 服务器的 break 接口，中止生成但不清空 KV cache
    try:
        break_resp = await http_client.post(
            f"{CPP_SERVER_URL}/v1/stream/break",
            json={}
        )
        if break_resp.status_code == 200:
            print(f"[omni_stop] C++ 生成已中止: {break_resp.json()}", flush=True)
        else:
            print(f"[omni_stop] C++ break 调用失败: {break_resp.status_code} - {break_resp.text}", flush=True)
    except Exception as e:
        print(f"[omni_stop] C++ break 调用异常: {e}", flush=True)
    
    # 设置 break 标志，让 generate_stream 停止发送数据
    is_breaking = True
    
    # 关闭并写入 WAV 时序日志总结
    if wav_timing_log_file:
        try:
            wav_timing_log_file.write(f"{'-'*120}\n")
            wav_timing_log_file.write(f"[会话停止] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
            wav_timing_log_file.close()
            print(f"[📊 WAV 时序日志已写入] {WAV_TIMING_LOG_PATH}", flush=True)
        except:
            pass
        wav_timing_log_file = None
    last_wav_send_time = None
    
    # 重置计数器和缓存
    with session_lock:
        current_active_session_id = None
        current_request_counter = 0
        current_round_number = 0
        pending_prefill_data = None
        # 双工模式需要重置的全局状态
        global_sent_wav_count = 0
        global_parsed_line_count = 0
        global_parsed_texts = []
        global_text_send_idx = 0
        global_sent_wav_files = set()
    
    print(f"会话已暂停: {stopped_session_id} (会话状态保留，可继续对话)", flush=True)
    print("======= 生成已中止，会话和 KV cache 保留，可直接继续 prefill =======", flush=True)
    
    return {
        "success": True,
        "message": "生成已中止，会话保留，可直接继续对话",
        "state": "generation_stopped",
        "session_id": stopped_session_id,
        "kv_cache_preserved": True
    }


@app.post("/omni/break")
async def omni_break():
    """单轮打断（只打断当前轮decode，不重置会话）"""
    global is_breaking
    
    if not model_state_initialized:
        raise HTTPException(status_code=503, detail="模型未初始化")
    
    try:
        print("======= 收到单轮打断指令 =======", flush=True)
        
        # 【关键】立即设置 break 标志，让 generate_stream 停止向前端发送数据
        is_breaking = True
        print("[omni_break] is_breaking 已设置为 True，中间层将停止发送数据", flush=True)
        
        # 调用 C++ 服务器的 break 接口，中止当前生成
        try:
            break_resp = await http_client.post(
                f"{CPP_SERVER_URL}/v1/stream/break",
                json={}
            )
            if break_resp.status_code == 200:
                print(f"[omni_break] C++ 生成已中止: {break_resp.json()}", flush=True)
            else:
                print(f"[omni_break] C++ break 调用失败: {break_resp.status_code} - {break_resp.text}", flush=True)
        except Exception as e:
            print(f"[omni_break] C++ break 调用异常: {e}", flush=True)
        
        print("======= 当前轮对话已打断（会话状态保留）=======", flush=True)
        return {"success": True, "message": "当前轮对话已打断", "state": "break"}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"打断失败: {str(e)}")


@app.post("/omni/init_sys_prompt")
async def init_sys_prompt(request: InitSysPromptRequest):
    """初始化系统提示
    
    C++ 版本：server 启动时已加载所有模型，此接口只做状态初始化
    - 首次调用：初始化 omni 上下文（加载 TTS/APM/VPM 等模块，复用已有 LLM）
    - 后续调用：快速恢复会话状态
    
    支持模式切换：
    - duplex_mode=True: 双工模式，直接转发 prefill，全局 WAV 计数器
    - duplex_mode=False: 单工模式，使用"延迟一拍"机制
    """
    global current_msg_type, current_duplex_mode, current_active_session_id, current_request_counter
    global current_round_number, model_state_initialized, pending_prefill_data
    global current_high_quality_mode, current_high_fps_mode
    global global_sent_wav_count, global_parsed_line_count, global_parsed_texts, global_text_send_idx, global_sent_wav_files
    global wav_timing_log_file, last_wav_send_time
    global is_breaking
    
    # 🔧 [修复] 重置 is_breaking 标志，防止上一次 stop 后残留的状态影响新会话
    # 场景：调度中心调用 /omni/stop 设置 is_breaking=True，之后新用户开始会话
    #       如果不在 init_sys_prompt 中重置，新用户的 streaming_generate 会检测到残留的 is_breaking=True
    if is_breaking:
        print("[init_sys_prompt] 检测到残留的 is_breaking=True，重置为 False", flush=True)
        is_breaking = False
    
    # 🔧 [修复] 检查是否正在重启，防止重启期间的请求导致冲突
    if cpp_restarting:
        print("[init_sys_prompt] 服务正在重启中，请稍后重试", flush=True)
        raise HTTPException(status_code=503, detail="服务正在重启中，请稍后重试")
    
    try:
        # 清空 output 子目录（每次 init 时清空上一次的输出）
        clear_output_subfolders()
        
        # 设置 duplex_mode（优先使用请求参数，否则使用默认值）
        if request.duplex_mode is not None:
            duplex_mode = request.duplex_mode
        else:
            duplex_mode = app.state.default_duplex_mode
        
        # 🔧 [修复] 不在 init_sys_prompt 时调用 reset 清空 KV cache
        # 原因：预初始化时已经 prefill 了 system prompt，清空 KV cache 会导致上下文丢失
        # LLM 线程在 decode 时会自己清理用户对话部分（保留 n_keep = system prompt）
        
        # 生成新的会话ID
        new_session_id = str(uuid.uuid4())[:8]
        
        # 设置 msg_type
        if request.media_type:
            if request.media_type.lower() == "audio":
                msg_type = 1
            elif request.media_type.lower() in ["video", "omni"]:
                msg_type = 2
            else:
                raise HTTPException(status_code=400, detail=f"不支持的media_type: {request.media_type}")
        else:
            msg_type = 2  # 默认 omni 模式
        
        # 🔧 [高清模式] 设置 max_slice_nums
        high_quality_mode = request.high_quality_mode if request.high_quality_mode is not None else False
        
        # 🔧 [高刷模式] 设置 1秒5帧 stack
        high_fps_mode = request.high_fps_mode if request.high_fps_mode is not None else False
        
        # 🔧 [语言切换] 设置语言 ("zh" 或 "en")
        language = request.language if request.language is not None else "zh"
        
        is_audio_mode = (msg_type == 1)
        mode_name = "audio" if is_audio_mode else "omni"
        duplex_name = "双工" if duplex_mode else "单工"
        quality_name = "高清" if high_quality_mode else "普通"
        fps_name = "高刷" if high_fps_mode else "标准帧率"
        
        # 检测 duplex_mode 是否变化（用于警告日志）
        # 注意：每个 server 实例的 duplex_mode 在启动时确定，运行时不应改变
        duplex_mode_changed = model_state_initialized and (current_duplex_mode != duplex_mode)
        if duplex_mode_changed:
            print(f"[警告] duplex_mode 从 {current_duplex_mode} 变为 {duplex_mode}，但 server 已初始化，此次请求的 duplex_mode 将被忽略", flush=True)
            duplex_mode = current_duplex_mode  # 保持原有模式
        
        # 🔧 [修复] 检测 media_type 是否变化（audio <-> omni）
        # 不同模式需要不同的 system prompt
        media_type_changed = model_state_initialized and (current_msg_type != msg_type)
        if media_type_changed:
            print(f"[模式切换] media_type 从 {current_msg_type} 变为 {msg_type}，调用 update_session_config", flush=True)
        
        current_msg_type = msg_type
        current_duplex_mode = duplex_mode
        
        # 🔧 [优化] 首次时调用 omni_init，media_type 变化时调用 update_session_config
        if not model_state_initialized:
            # model_dir 可以是绝对路径或相对路径（相对于 llamacpp 根目录）
            model_dir = app.state.model_dir
            # TTS 模型在 tts/ 目录
            tts_bin_dir = os.path.join(model_dir, "tts")
            
            cpp_request = {
                "media_type": msg_type,      # 1=audio, 2=omni
                "use_tts": True,             # 启用 TTS 语音合成
                "duplex_mode": duplex_mode,  # 双工/单工模式
                "model_dir": model_dir,
                "tts_bin_dir": tts_bin_dir,
                "tts_gpu_layers": 100,
                "token2wav_device": TOKEN2WAV_DEVICE,
                "output_dir": CPP_OUTPUT_DIR,  # 🔧 [多实例支持] 传递配置的输出目录
                "language": language,        # 🔧 [语言切换] "zh" 或 "en"
            }
            
            # 视觉编码器后端
            cpp_request["vision_backend"] = VISION_BACKEND
            
            # 🔧 [高清模式] 设置 max_slice_nums
            if high_quality_mode:
                cpp_request["max_slice_nums"] = 2  # 高清模式：切图
                print(f"[高清模式] 启用图片切片 max_slice_nums=2", flush=True)
            
            # 保存模式状态
            current_high_quality_mode = high_quality_mode
            current_high_fps_mode = high_fps_mode
            
            print(f"[模式设置] 双工={duplex_mode}, 高清={high_quality_mode}, 高刷={high_fps_mode}", flush=True)
            
            # 使用固定音色文件
            if os.path.exists(FIXED_TIMBRE_PATH):
                cpp_request["voice_audio"] = FIXED_TIMBRE_PATH
                print(f"使用音色文件: {FIXED_TIMBRE_PATH}", flush=True)

            # 🔧 [自定义系统提示词] 如果提供，传递给 C++
            if request.system_prompt_prefix:
                cpp_request["system_prompt_prefix"] = request.system_prompt_prefix
                print(f"[自定义系统提示词前缀] {request.system_prompt_prefix}", flush=True)
            if request.system_prompt_suffix:
                cpp_request["system_prompt_suffix"] = request.system_prompt_suffix
                print(f"[自定义系统提示词后缀] {request.system_prompt_suffix}", flush=True)

            print(f"初始化，调用 C++ omni_init: {json.dumps(cpp_request, ensure_ascii=False)}", flush=True)
            
            resp = await http_client.post(
                f"{CPP_SERVER_URL}/v1/stream/omni_init",
                json=cpp_request
            )
            
            if resp.status_code != 200:
                error_text = resp.text
                print(f"C++ omni_init 失败: {error_text}", flush=True)
                raise HTTPException(status_code=500, detail=f"C++ omni_init 失败: {error_text}")
            
            cpp_result = resp.json()
            print(f"C++ omni_init 成功: {cpp_result}", flush=True)
            model_state_initialized = True

            # 🔧 [System Prompt 缓存] 保存当前 system prompt 用于后续比较
            global current_system_prompt_prefix, current_system_prompt_suffix
            current_system_prompt_prefix = request.system_prompt_prefix
            current_system_prompt_suffix = request.system_prompt_suffix
            print(f"[System Prompt 缓存] prefix={current_system_prompt_prefix is not None}, suffix={current_system_prompt_suffix is not None}", flush=True)

            fast_resume = False
            init_message = f"初始化完成（{mode_name}模式，{duplex_name}，{quality_name}画质，{fps_name}）"
        elif media_type_changed:
            # 🔧 [优化] media_type 变化，调用 update_session_config（不重新加载模型）
            current_high_quality_mode = high_quality_mode
            current_high_fps_mode = high_fps_mode
            
            update_request = {
                "media_type": msg_type,
                "duplex_mode": duplex_mode,
                "language": language,  # 🔧 [语言切换]
            }

            # 使用固定音色文件重新 prefill system prompt
            if os.path.exists(FIXED_TIMBRE_PATH):
                update_request["voice_audio"] = FIXED_TIMBRE_PATH

            # 🔧 [自定义系统提示词] 如果提供，传递给 C++
            if request.system_prompt_prefix:
                update_request["system_prompt_prefix"] = request.system_prompt_prefix
                print(f"[自定义系统提示词前缀] {request.system_prompt_prefix}", flush=True)
            if request.system_prompt_suffix:
                update_request["system_prompt_suffix"] = request.system_prompt_suffix
                print(f"[自定义系统提示词后缀] {request.system_prompt_suffix}", flush=True)

            print(f"[模式切换] 调用 C++ update_session_config: {json.dumps(update_request, ensure_ascii=False)}", flush=True)
            
            resp = await http_client.post(
                f"{CPP_SERVER_URL}/v1/stream/update_session_config",
                json=update_request,
                timeout=30.0
            )
            
            if resp.status_code != 200:
                error_text = resp.text
                print(f"C++ update_session_config 失败: {error_text}", flush=True)
                raise HTTPException(status_code=500, detail=f"C++ update_session_config 失败: {error_text}")
            
            cpp_result = resp.json()
            print(f"C++ update_session_config 成功: {cpp_result}", flush=True)
            fast_resume = False

            # 🔧 [System Prompt 缓存] 更新缓存
            current_system_prompt_prefix = request.system_prompt_prefix
            current_system_prompt_suffix = request.system_prompt_suffix

            init_message = f"模式切换完成（{mode_name}模式，{duplex_name}，{quality_name}画质，{fps_name}）"
        else:
            # 已初始化且模式未变，但仍需通知 C++ 重置状态
            current_high_quality_mode = high_quality_mode
            current_high_fps_mode = high_fps_mode

            # 🔧 [极速恢复优化] 检查 system prompt 是否变化
            system_prompt_changed = (
                request.system_prompt_prefix != current_system_prompt_prefix or
                request.system_prompt_suffix != current_system_prompt_suffix
            )

            if system_prompt_changed:
                # System prompt 变化，需要重新 prefill
                print(f"[极速恢复] System prompt 变化，需要重新 prefill", flush=True)
                print(f"  原 prefix: {current_system_prompt_prefix}", flush=True)
                print(f"  新 prefix: {request.system_prompt_prefix}", flush=True)
                print(f"  原 suffix: {current_system_prompt_suffix}", flush=True)
                print(f"  新 suffix: {request.system_prompt_suffix}", flush=True)

                update_request = {
                    "media_type": msg_type,
                    "duplex_mode": duplex_mode,
                    "language": language,
                }

                if os.path.exists(FIXED_TIMBRE_PATH):
                    update_request["voice_audio"] = FIXED_TIMBRE_PATH

                if request.system_prompt_prefix:
                    update_request["system_prompt_prefix"] = request.system_prompt_prefix
                if request.system_prompt_suffix:
                    update_request["system_prompt_suffix"] = request.system_prompt_suffix

                print(f"[极速恢复] 调用 C++ update_session_config: {json.dumps(update_request, ensure_ascii=False)}", flush=True)

                resp = await http_client.post(
                    f"{CPP_SERVER_URL}/v1/stream/update_session_config",
                    json=update_request,
                    timeout=30.0
                )

                if resp.status_code != 200:
                    error_text = resp.text
                    print(f"[极速恢复] C++ update_session_config 失败: {error_text}", flush=True)
                    raise HTTPException(status_code=500, detail=f"C++ update_session_config 失败: {error_text}")

                cpp_result = resp.json()
                print(f"[极速恢复] C++ update_session_config 成功: {cpp_result}", flush=True)
            else:
                # System prompt 未变化，使用轻量级 reset API
                print(f"[极速恢复] System prompt 未变化，使用轻量级 reset", flush=True)

                reset_request = {
                    "duplex_mode": duplex_mode,
                }

                print(f"[极速恢复] 调用 C++ reset: {json.dumps(reset_request, ensure_ascii=False)}", flush=True)

                resp = await http_client.post(
                    f"{CPP_SERVER_URL}/v1/stream/reset",
                    json=reset_request,
                    timeout=10.0
                )

                if resp.status_code != 200:
                    error_text = resp.text
                    print(f"[极速恢复] C++ reset 失败: {error_text}", flush=True)
                    # reset 失败回退到 update_session_config
                    print(f"[极速恢复] 回退到 update_session_config", flush=True)

                    update_request = {
                        "media_type": msg_type,
                        "duplex_mode": duplex_mode,
                        "language": language,
                    }
                    if os.path.exists(FIXED_TIMBRE_PATH):
                        update_request["voice_audio"] = FIXED_TIMBRE_PATH

                    resp = await http_client.post(
                        f"{CPP_SERVER_URL}/v1/stream/update_session_config",
                        json=update_request,
                        timeout=30.0
                    )
                    if resp.status_code != 200:
                        raise HTTPException(status_code=500, detail=f"C++ fallback 失败: {resp.text}")
                    cpp_result = resp.json()
                    print(f"[极速恢复] C++ update_session_config 成功: {cpp_result}", flush=True)
                else:
                    cpp_result = resp.json()
                    print(f"[极速恢复] C++ reset 成功: {cpp_result}", flush=True)

            fast_resume = True

            # 🔧 [System Prompt 缓存] 更新缓存
            current_system_prompt_prefix = request.system_prompt_prefix
            current_system_prompt_suffix = request.system_prompt_suffix

            init_message = f"初始化成功（{mode_name}模式，{duplex_name}，{quality_name}画质，{fps_name}，快速恢复）"

        # 关闭之前的日志文件（如果有）
        if wav_timing_log_file:
            try:
                wav_timing_log_file.write(f"{'-'*120}\n")
                wav_timing_log_file.write(f"[新会话初始化，关闭旧日志] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
                wav_timing_log_file.close()
            except:
                pass
            wav_timing_log_file = None
        last_wav_send_time = None
        
        # 更新会话状态
        with session_lock:
            current_active_session_id = new_session_id
            current_request_counter = 0
            current_round_number = 0
            pending_prefill_data = None
            # 双工模式需要重置的全局状态
            global_sent_wav_count = 0
            global_parsed_line_count = 0
            global_parsed_texts = []
            global_text_send_idx = 0
            global_sent_wav_files = set()
        
        # 🔧 [高刷模式] 清理图片缓存
        with high_fps_cache_lock:
            high_fps_subimage_cache.clear()
            print(f"[init_sys_prompt] 已清理高刷模式图片缓存", flush=True)
        
        return {
            "success": True,
            "message": init_message,
            "msg_type": msg_type,
            "duplex_mode": duplex_mode,
            "session_id": new_session_id,
            "fast_resume": fast_resume
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")


@app.post("/omni/streaming_prefill")
async def streaming_prefill(request: StreamingPrefillRequest):
    """流式预填充
    
    根据 duplex_mode 使用不同的处理逻辑：
    - 单工模式：使用"延迟一拍"机制
    - 双工模式：直接转发给 C++ /v1/stream/prefill
    """
    global pending_prefill_data, current_request_counter
    
    # 🔧 [修复] 检查是否正在重启
    if cpp_restarting:
        raise HTTPException(status_code=503, detail="服务正在重启中，请稍后重试")
    
    if not current_active_session_id:
        raise HTTPException(
            status_code=400,
            detail="未找到活跃会话，请先调用 /omni/init_sys_prompt 初始化会话"
        )
    
    prefill_start_time = time.time()
    
    # ========== 性能统计变量 ==========
    timing_stats = {}
    
    try:
        # 1. 解码音频
        t0 = time.time()
        audio_np = None
        sr = 16000
        if request.audio:
            try:
                audio_bytes = base64.b64decode(request.audio)
                # 先用 soundfile 读取，获取原始采样率
                audio_np, file_sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
                
                # 如果是立体声，转为单声道
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.mean(axis=1)
                
                # 如果采样率不是 16kHz，使用 librosa 重采样
                if file_sr != 16000:
                    audio_np = librosa.resample(audio_np, orig_sr=file_sr, target_sr=16000)
                
                audio_np = audio_np.astype(np.float32)
                sr = 16000
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"音频数据解码失败: {str(e)}")
        timing_stats['audio_decode'] = (time.time() - t0) * 1000
        
        # 2. 解码图片
        t0 = time.time()
        pil_image = None
        if request.image:
            try:
                image_bytes = base64.b64decode(request.image)
                pil_image = Image.open(io.BytesIO(image_bytes))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"图片数据解码失败: {str(e)}")
        timing_stats['image_decode'] = (time.time() - t0) * 1000
        
        # 🔧 [高刷模式] 新的图片/音频分离处理逻辑
        # 逻辑：
        # 1. 主图（frame_index=0）立即 prefill
        # 2. 子图（frame_index=1-4）缓存，等满4张或收到音频时触发
        # 3. 音频到达时：取出缓存的子图，stack 后 prefill（图片在前）
        
        # 🔧 [高清+高刷] 标记是否为主图（用于决定高清切片）
        is_main_image = False
        
        if current_high_fps_mode and request.image_audio_id is not None:
            frame_idx = request.frame_index if request.frame_index is not None else 0
            
            # 情况1：只有图片，没有音频
            if pil_image is not None and audio_np is None:
                if frame_idx == 0:
                    # 主图：立即 prefill，不缓存
                    print(f"[高刷模式] 主图到达 image_audio_id={request.image_audio_id}，立即 prefill", flush=True)
                    pil_images = [pil_image]
                    audio_np = None  # 明确没有音频
                    is_main_image = True  # 🔧 [高清+高刷] 标记为主图
                    # 继续后面的 prefill 流程
                else:
                    # 子图（frame_index 1-4）：缓存
                    with high_fps_cache_lock:
                        if request.image_audio_id not in high_fps_subimage_cache:
                            high_fps_subimage_cache[request.image_audio_id] = {}
                        high_fps_subimage_cache[request.image_audio_id][frame_idx] = pil_image
                        cached_count = len(high_fps_subimage_cache[request.image_audio_id])
                        # 检查是否收齐4张子图（frame 1,2,3,4）
                        all_subframes_ready = all(
                            i in high_fps_subimage_cache[request.image_audio_id] 
                            for i in [1, 2, 3, 4]
                        )
                    
                    print(f"[高刷模式] 子图缓存 image_audio_id={request.image_audio_id}, frame={frame_idx}, 已缓存{cached_count}帧", flush=True)
                    
                    if all_subframes_ready:
                        # 收齐4张子图，检查是否有待处理的音频
                        pending_audio = None
                        with high_fps_audio_lock:
                            if request.image_audio_id in high_fps_pending_audio:
                                pending_audio = high_fps_pending_audio.pop(request.image_audio_id)
                        
                        if pending_audio is not None:
                            # 有待处理的音频，取出子图，stack，然后 prefill
                            audio_np, sr, _ = pending_audio
                            with high_fps_cache_lock:
                                cached_frames = high_fps_subimage_cache.pop(request.image_audio_id, {})
                            sorted_frames = sorted(cached_frames.items(), key=lambda x: x[0])
                            subimages = [img for _, img in sorted_frames]
                            stacked_image = stack_images(subimages)
                            pil_images = [stacked_image]
                            print(f"[高刷模式] 子图收齐+待处理音频，stack {len(subimages)} 帧，prefill", flush=True)
                            # 继续后面的 prefill 流程
                        else:
                            # 没有待处理的音频，只是缓存完成
                            return {
                                "success": True,
                                "message": f"子图已缓存完毕，等待音频 (image_audio_id={request.image_audio_id})",
                                "cached_frames": cached_count,
                                "mode": "high_fps_cache_ready"
                            }
                    else:
                        # 还没收齐，只返回缓存状态
                        return {
                            "success": True,
                            "message": f"子图已缓存 (image_audio_id={request.image_audio_id}, frame={frame_idx})",
                            "cached_frames": cached_count,
                            "mode": "high_fps_cache"
                        }
            
            # 情况2：有音频（可能同时有图片）
            elif audio_np is not None:
                # 从缓存取出子图
                with high_fps_cache_lock:
                    cached_frames = high_fps_subimage_cache.pop(request.image_audio_id, {})
                
                if len(cached_frames) > 0:
                    # 有缓存的子图，stack 后 prefill
                    sorted_frames = sorted(cached_frames.items(), key=lambda x: x[0])
                    subimages = [img for _, img in sorted_frames]
                    stacked_image = stack_images(subimages)
                    pil_images = [stacked_image]
                    print(f"[高刷模式] 音频到达，取出 {len(subimages)} 帧子图 stack，prefill", flush=True)
                    
                    # 如果当前请求也带图片（不应该发生，但做个保护）
                    if pil_image is not None:
                        pil_images.append(pil_image)
                else:
                    # 没有缓存的子图，检查子图是否还没到齐
                    # 缓存音频，等子图到齐
                    with high_fps_audio_lock:
                        high_fps_pending_audio[request.image_audio_id] = (audio_np, sr, None)
                    print(f"[高刷模式] 音频到达但无子图缓存，暂存音频等待子图 image_audio_id={request.image_audio_id}", flush=True)
                    return {
                        "success": True,
                        "message": f"音频已暂存，等待子图 (image_audio_id={request.image_audio_id})",
                        "mode": "high_fps_audio_pending"
                    }
        else:
            # 非高刷模式或没有 image_audio_id：使用原有逻辑
            pil_images = [pil_image] if pil_image is not None else []
        
        if audio_np is None and len(pil_images) == 0:
            raise HTTPException(status_code=400, detail="必须提供音频或图片至少一项")
        
        audio_duration = len(audio_np) / sr if audio_np is not None else 0.0
        omni_mode = (current_msg_type == 2)
        
        # ========== 根据模式选择不同的处理逻辑 ==========
        if current_duplex_mode:
            # ========== 双工模式：直接转发给 C++ ==========
            return await _streaming_prefill_duplex(
                request, audio_np, pil_images, sr, audio_duration,
                omni_mode, timing_stats, prefill_start_time, request.text
            )
        elif current_high_fps_mode and current_msg_type == 2:
            # ========== 高刷单工模式：直接 prefill，不延迟 ==========
            # 高刷模式只对 omni 模式（有图片）有意义，audio 模式走普通单工路径
            # 高刷模式通过 image_audio_id 保证配对，不需要"延迟一拍"
            # 主图立即 prefill，音频+stack图也立即 prefill
            return await _streaming_prefill_highfps_direct(
                request, audio_np, pil_images, sr, audio_duration,
                omni_mode, timing_stats, prefill_start_time,
                is_main_image=is_main_image,  # 🔧 [高清+高刷] 传入主图标记
                user_text=request.text  # 🔧 [文本输入] 传入用户文本
            )
        else:
            # ========== 普通单工模式：使用"延迟一拍"机制 ==========
            return await _streaming_prefill_simplex(
                request, audio_np, pil_images, sr, audio_duration,
                omni_mode, timing_stats, prefill_start_time, request.text
            )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"预填充失败: {str(e)}")


async def _streaming_prefill_duplex(
    request, audio_np, pil_images, sr, audio_duration,
    omni_mode, timing_stats, prefill_start_time, user_text=None
):
    """双工模式的 streaming_prefill 实现：直接转发给 C++"""
    global current_request_counter
    
    # 增加请求计数，计算 cnt（从 0 开始）
    with session_lock:
        cnt = current_request_counter
        current_request_counter += 1
    
    # 3. 保存音频到临时文件
    t0 = time.time()
    temp_audio_path = ""
    if audio_np is not None and len(audio_np) > 0:
        # 音频太短时进行 padding（最少 0.1s = 1600 samples）
        MIN_AUDIO_SAMPLES = 1600
        if len(audio_np) < MIN_AUDIO_SAMPLES:
            original_len = len(audio_np)
            padding_len = MIN_AUDIO_SAMPLES - original_len
            audio_np = np.pad(audio_np, (0, padding_len), mode='constant', constant_values=0)
        
        temp_audio_path = os.path.join(TEMP_DIR, f"prefill_{current_active_session_id}_{cnt}.wav")
        audio_to_save = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
        sf.write(temp_audio_path, audio_to_save, 16000, format='WAV', subtype='PCM_16')
    timing_stats['audio_save'] = (time.time() - t0) * 1000
    
    # 4. 处理图片并保存到临时文件
    t0 = time.time()
    temp_image_paths = []
    
    if len(pil_images) > 0:
        if current_high_fps_mode and len(pil_images) > 1:
            # 高刷模式：第1张是主图，后面的 stack 成一张
            main_image = pil_images[0]
            rest_images = pil_images[1:]
            
            main_path = os.path.join(TEMP_DIR, f"prefill_{current_active_session_id}_{cnt}_main.png")
            main_image.save(main_path, format='PNG')
            temp_image_paths.append(main_path)
            
            if len(rest_images) > 0:
                stacked_image = stack_images(rest_images)
                stack_path = os.path.join(TEMP_DIR, f"prefill_{current_active_session_id}_{cnt}_stack.png")
                stacked_image.save(stack_path, format='PNG')
                temp_image_paths.append(stack_path)
                print(f"[高刷模式] 处理 {len(pil_images)} 帧，主图1张 + stack {len(rest_images)} 帧成1张", flush=True)
        else:
            # 普通模式：单张图
            img_path = os.path.join(TEMP_DIR, f"prefill_{current_active_session_id}_{cnt}.png")
            pil_images[0].save(img_path, format='PNG')
            temp_image_paths.append(img_path)
    
    timing_stats['image_save'] = (time.time() - t0) * 1000
    
    # 5. 调用 C++ prefill
    t0 = time.time()
    cpp_success = True
    
    if len(temp_image_paths) == 0:
        # 只有音频，没有图片
        cpp_request = {
            "audio_path_prefix": temp_audio_path,
            "img_path_prefix": "",
            "cnt": cnt,
            "user_text": user_text if user_text else ""
        }
        resp = await http_client.post(
            f"{CPP_SERVER_URL}/v1/stream/prefill",
            json=cpp_request,
            timeout=30.0
        )
        cpp_success = (resp.status_code == 200)
    else:
        # 有图片：第一张图和音频一起发，后续图片单独发
        for i, img_path in enumerate(temp_image_paths):
            cpp_request = {
                "audio_path_prefix": temp_audio_path if i == 0 else "",
                "img_path_prefix": img_path,
                "cnt": cnt + i,
                "user_text": user_text if (i == 0 and user_text) else ""
            }
            resp = await http_client.post(
                f"{CPP_SERVER_URL}/v1/stream/prefill",
                json=cpp_request,
                timeout=30.0
            )
            if resp.status_code != 200:
                cpp_success = False
                break
        
        # 更新 counter
        with session_lock:
            current_request_counter += len(temp_image_paths) - 1
    
    timing_stats['cpp_http'] = (time.time() - t0) * 1000
    
    total_prefill_time = (time.time() - prefill_start_time) * 1000
    timing_stats['total'] = total_prefill_time
    
    # 打印性能统计
    num_images = len(temp_image_paths)
    has_image = f"✓({num_images}张)" if num_images > 0 else "✗"
    if cpp_success:
        print(f"[Prefill #{cnt}] ✓ {total_prefill_time:.0f}ms (音频:{audio_duration:.2f}s 图片:{has_image}) [双工]", flush=True)
    else:
        print(f"[Prefill #{cnt}] ✗ C++ prefill 失败 [双工]", flush=True)
    
    # 🔧 清理临时文件（C++ 已读取完毕，不再需要）
    try:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        for img_path in temp_image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
    except Exception:
        pass
    
    return {
        "success": cpp_success,
        "session_id": current_active_session_id,
        "cnt": cnt,
        "audio_duration_seconds": float(audio_duration),
        "timing": timing_stats,
        "backend": "cpp_duplex"
    }


async def _streaming_prefill_highfps_direct(
    request, audio_np, pil_images, sr, audio_duration,
    omni_mode, timing_stats, prefill_start_time,
    is_main_image: bool = False,  # 🔧 [高清+高刷] 标记是否为主图（决定是否使用高清切片）
    user_text=None  # 🔧 [文本输入] 用户文本输入
):
    """高刷单工模式的 streaming_prefill 实现：直接 prefill，不延迟
    
    高刷模式通过 image_audio_id 保证数据配对，不需要"延迟一拍"机制。
    - 主图到达：立即 prefill 主图（无音频），如果开启高清则 max_slice_nums=2
    - 音频+stack图到达：立即 prefill stack图+音频，max_slice_nums=1（不切片）
    """
    global current_request_counter
    
    # 增加请求计数
    with session_lock:
        cnt = current_request_counter
        current_request_counter += 1
    
    # 保存音频到临时文件
    t0 = time.time()
    temp_audio_path = ""
    if audio_np is not None and len(audio_np) > 0:
        # 音频太短时进行 padding（最少 0.1s = 1600 samples）
        MIN_AUDIO_SAMPLES = 1600
        if len(audio_np) < MIN_AUDIO_SAMPLES:
            padding_len = MIN_AUDIO_SAMPLES - len(audio_np)
            audio_np = np.pad(audio_np, (0, padding_len), mode='constant', constant_values=0)
        
        temp_audio_path = os.path.join(TEMP_DIR, f"prefill_{current_active_session_id}_{cnt}.wav")
        audio_to_save = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
        sf.write(temp_audio_path, audio_to_save, 16000, format='WAV', subtype='PCM_16')
    timing_stats['audio_save'] = (time.time() - t0) * 1000
    
    # 保存图片到临时文件
    t0 = time.time()
    temp_image_paths = []
    
    if len(pil_images) > 0:
        for i, img in enumerate(pil_images):
            img_path = os.path.join(TEMP_DIR, f"prefill_{current_active_session_id}_{cnt}_{i}.png")
            img.save(img_path, format='PNG')
            temp_image_paths.append(img_path)
    
    timing_stats['image_save'] = (time.time() - t0) * 1000
    
    # 调用 C++ prefill
    t0 = time.time()
    cpp_success = True
    
    # 🔧 [高清+高刷] 根据 is_main_image 决定 max_slice_nums
    # 主图且高清开启：max_slice_nums=2（切片）
    # Stacked 图或普通模式：max_slice_nums=1（不切片）
    if is_main_image and current_high_quality_mode:
        slice_nums = 2  # 主图使用高清切片
        slice_desc = "高清"
    else:
        slice_nums = 1  # Stacked 图不切片
        slice_desc = "普通"
    
    if len(temp_image_paths) == 0:
        # 只有音频，没有图片（不太可能在高刷模式下发生）
        cpp_request = {
            "audio_path_prefix": temp_audio_path,
            "img_path_prefix": "",
            "cnt": cnt,
            "user_text": user_text if user_text else ""
        }
        resp = await http_client.post(
            f"{CPP_SERVER_URL}/v1/stream/prefill",
            json=cpp_request,
            timeout=30.0
        )
        cpp_success = (resp.status_code == 200)
    else:
        # 有图片：图片和音频一起发
        for i, img_path in enumerate(temp_image_paths):
            cpp_request = {
                "audio_path_prefix": temp_audio_path if i == 0 else "",
                "img_path_prefix": img_path,
                "cnt": cnt + i,
                "max_slice_nums": slice_nums,  # 🔧 [高清+高刷] 传入切片参数
                "user_text": user_text if (i == 0 and user_text) else ""
            }
            resp = await http_client.post(
                f"{CPP_SERVER_URL}/v1/stream/prefill",
                json=cpp_request,
                timeout=30.0
            )
            if resp.status_code != 200:
                cpp_success = False
                break
        
        # 更新 counter
        with session_lock:
            current_request_counter += len(temp_image_paths) - 1
    
    timing_stats['cpp_http'] = (time.time() - t0) * 1000
    
    total_prefill_time = (time.time() - prefill_start_time) * 1000
    timing_stats['total'] = total_prefill_time
    
    # 打印性能统计
    num_images = len(temp_image_paths)
    has_image = f"✓({num_images}张)" if num_images > 0 else "✗"
    has_audio = f"{audio_duration:.2f}s" if audio_duration > 0 else "无"
    img_type = "主图" if is_main_image else "Stacked图"
    if cpp_success:
        print(f"[Prefill #{cnt}] ✓ {total_prefill_time:.0f}ms (音频:{has_audio} 图片:{has_image} {img_type}/{slice_desc}) [高刷单工]", flush=True)
    else:
        print(f"[Prefill #{cnt}] ✗ C++ prefill 失败 (status={resp.status_code if 'resp' in dir() else 'N/A'}) [高刷单工]", flush=True)
    
    # 🔧 清理临时文件（C++ 已读取完毕，不再需要）
    try:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        for img_path in temp_image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
    except Exception:
        pass
    
    return {
        "success": cpp_success,
        "session_id": current_active_session_id,
        "cnt": cnt,
        "audio_duration_seconds": float(audio_duration),
        "timing": timing_stats,
        "backend": "cpp_highfps_simplex"
    }


async def _streaming_prefill_simplex(
    request, audio_np, pil_images, sr, audio_duration,
    omni_mode, timing_stats, prefill_start_time, user_text=None
):
    """普通单工模式的 streaming_prefill 实现：使用"延迟一拍"机制"""
    global pending_prefill_data, current_request_counter
    
    # 增加请求计数
    with session_lock:
        current_request_counter += 1
        request_idx = current_request_counter
    
    # 【延迟一拍】先处理上一次缓存的数据
    model_prefill_start = time.time()
    if pending_prefill_data is not None:
        prev_data = pending_prefill_data
        prev_images = prev_data.get("images", [])
        prev_audio = prev_data["audio_np"]
        has_prev_audio = prev_audio is not None and len(prev_audio) > 0
        has_prev_images = len(prev_images) > 0
        
        # 🔧 [高刷修复] 只要有音频或图片，就处理上一次的缓存数据
        if has_prev_audio or has_prev_images:
            prev_cnt = prev_data["cnt"]
            
            # 处理音频
            temp_audio_path = ""
            if has_prev_audio:
                temp_audio_path = os.path.join(TEMP_DIR, f"prefill_{current_active_session_id}_{prev_cnt}.wav")
                audio_to_save = np.clip(prev_audio, -1.0, 1.0).astype(np.float32)
                sf.write(temp_audio_path, audio_to_save, 16000, format='WAV', subtype='PCM_16')
            
            # 处理图片列表
            temp_image_paths = []
            
            if has_prev_images:
                if current_high_fps_mode and len(prev_images) > 1:
                    main_image = prev_images[0]
                    rest_images = prev_images[1:]
                    
                    main_path = os.path.join(TEMP_DIR, f"prefill_{current_active_session_id}_{prev_cnt}_main.png")
                    main_image.save(main_path, format='PNG')
                    temp_image_paths.append(main_path)
                    
                    if len(rest_images) > 0:
                        stacked_image = stack_images(rest_images)
                        stack_path = os.path.join(TEMP_DIR, f"prefill_{current_active_session_id}_{prev_cnt}_stack.png")
                        stacked_image.save(stack_path, format='PNG')
                        temp_image_paths.append(stack_path)
                else:
                    for i, img in enumerate(prev_images):
                        img_path = os.path.join(TEMP_DIR, f"prefill_{current_active_session_id}_{prev_cnt}_{i}.png")
                        img.save(img_path, format='PNG')
                        temp_image_paths.append(img_path)
            
            # 🔧 [文本输入] 获取上一次缓存的用户文本（确保是字符串，不是 None）
            prev_user_text = prev_data.get("user_text") or ""

            # 调用 C++ prefill
            if len(temp_image_paths) == 0:
                cpp_request = {
                    "audio_path_prefix": temp_audio_path,
                    "img_path_prefix": "",
                    "cnt": prev_cnt,
                    "user_text": prev_user_text
                }
                await http_client.post(f"{CPP_SERVER_URL}/v1/stream/prefill", json=cpp_request)
            else:
                for i, img_path in enumerate(temp_image_paths):
                    cpp_request = {
                        "audio_path_prefix": temp_audio_path if i == 0 else "",
                        "img_path_prefix": img_path,
                        "cnt": prev_cnt + i,
                        "user_text": prev_user_text if i == 0 else ""
                    }
                    await http_client.post(f"{CPP_SERVER_URL}/v1/stream/prefill", json=cpp_request)
            
            print(f"[延迟一拍] 处理了上一次缓存的 prefill 数据 (cnt={prev_cnt}) [单工]", flush=True)
    else:
        print(f"[延迟一拍] 首次 prefill，无缓存数据 [单工]", flush=True)
    
    # 计算当前数据的 cnt
    current_cnt = request_idx - 1
    
    # 缓存当前数据
    pending_prefill_data = {
        "audio_np": audio_np,
        "images": pil_images,
        "omni_mode": omni_mode,
        "audio_duration": audio_duration,
        "request_idx": request_idx,
        "cnt": current_cnt,
        "user_text": user_text or "",  # 🔧 [文本输入] 缓存用户文本（确保不是 None）
    }
    num_imgs = len(pil_images)
    print(f"[延迟一拍] 当前数据已缓存 (音频: {audio_duration:.2f}s, 图片: {num_imgs}张, cnt={current_cnt}) [单工]", flush=True)
    print(f"[🔔 提醒] 缓存数据等待 streaming_generate 调用处理 [单工]", flush=True)
    
    model_prefill_time = (time.time() - model_prefill_start) * 1000
    total_prefill_time = (time.time() - prefill_start_time) * 1000
    
    return {
        "success": True,
        "session_id": current_active_session_id,
        "request_idx": request_idx,
        "audio_duration_seconds": float(audio_duration),
        "timing": {
            "audio_decode_ms": round(timing_stats.get('audio_decode', 0), 1),
            "image_decode_ms": round(timing_stats.get('image_decode', 0), 1),
            "model_prefill_ms": round(model_prefill_time, 1),
            "total_ms": round(total_prefill_time, 1),
            "rtf": round(total_prefill_time / 1000 / audio_duration, 2) if audio_duration > 0 else None
        },
        "backend": "cpp_simplex"
    }


@app.post("/omni/streaming_generate")
async def streaming_generate():
    """流式生成
    
    根据 duplex_mode 使用不同的处理逻辑：
    - 单工模式：每个 round 有独立目录
    - 双工模式：全局 WAV 计数器，使用 SSE 流式读取
    """
    global pending_prefill_data, current_round_number, is_breaking
    
    # 🔧 [修复] 检查是否正在重启
    if cpp_restarting:
        raise HTTPException(status_code=503, detail="服务正在重启中，请稍后重试")
    
    if not current_active_session_id:
        raise HTTPException(
            status_code=400,
            detail="未找到活跃会话，请先调用 /omni/init_sys_prompt 初始化会话"
        )
    
    # 【重置 break 标志】开始新一轮生成时，重置 is_breaking
    is_breaking = False
    
    generate_request_time = time.time()
    print(f"[Generate] 开始生成 (Round #{current_round_number}, duplex_mode={current_duplex_mode})", flush=True)
    
    # 根据模式选择不同的实现
    if current_duplex_mode:
        return await _streaming_generate_duplex(generate_request_time)
    else:
        return await _streaming_generate_simplex(generate_request_time)


async def _streaming_generate_simplex(generate_request_time):
    """单工模式的 streaming_generate 实现"""
    global pending_prefill_data, current_round_number, is_breaking
    
    # 🔧 [诊断] 记录 generate 调用时的状态
    has_pending = pending_prefill_data is not None
    pending_cnt = pending_prefill_data.get("cnt", -1) if has_pending else -1
    print(f"[streaming_generate] 开始, pending_data={has_pending}, pending_cnt={pending_cnt}, round={current_round_number} [单工]", flush=True)
    
    # 【延迟一拍】处理缓存的最后一片数据
    if pending_prefill_data is not None:
        try:
            print("[streaming_generate] 处理缓存的最后一片数据 (is_last_chunk=True)... [单工]", flush=True)
            last_data = pending_prefill_data
            
            audio_np = last_data["audio_np"]
            if audio_np is not None and len(audio_np) > 0:
                MIN_AUDIO_SAMPLES = 1600
                if len(audio_np) < MIN_AUDIO_SAMPLES:
                    original_len = len(audio_np)
                    padding_len = MIN_AUDIO_SAMPLES - original_len
                    audio_np = np.pad(audio_np, (0, padding_len), mode='constant', constant_values=0)
                    print(f"[音频Padding] {original_len} -> {MIN_AUDIO_SAMPLES} samples", flush=True)
                
                last_cnt = last_data["cnt"]
                temp_audio_path = os.path.join(TEMP_DIR, f"prefill_{current_active_session_id}_{last_cnt}.wav")
                audio_to_save = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
                sf.write(temp_audio_path, audio_to_save, 16000, format='WAV', subtype='PCM_16')
                
                temp_image_path = ""
                images = last_data.get("images", [])
                if len(images) > 0:
                    temp_image_path = os.path.join(TEMP_DIR, f"prefill_{current_active_session_id}_{last_cnt}.png")
                    images[0].save(temp_image_path, format='PNG')
                
                # 🔧 [文本输入] 获取缓存的用户文本（确保是字符串，不是 None）
                user_text = last_data.get("user_text") or ""
                cpp_request = {
                    "audio_path_prefix": temp_audio_path,
                    "img_path_prefix": temp_image_path,
                    "cnt": last_cnt,
                    "user_text": user_text
                }

                resp = await http_client.post(
                    f"{CPP_SERVER_URL}/v1/stream/prefill",
                    json=cpp_request
                )
                
                if resp.status_code != 200:
                    print(f"C++ 最后一片 prefill 失败: {resp.text}", flush=True)
                else:
                    print(f"[streaming_generate] 最后一片 prefill 成功 (cnt={last_cnt}) [单工]", flush=True)
            
            pending_prefill_data = None
            print("[streaming_generate] 最后一片已处理 [单工]", flush=True)
            
        except Exception as e:
            print(f"[streaming_generate] 处理最后一片失败: {e}", flush=True)
            pending_prefill_data = None
    
    # 输出目录（单工模式：每个 round 有独立目录）
    output_dir = os.path.join(TEMP_DIR, f"session_{current_active_session_id}", f"round_{current_round_number:04d}", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    async def generate_stream():
        global current_round_number
        import re
        
        generate_start_time = time.time()
        first_chunk_time = None
        first_text_time = None
        chunk_durations = []
        sent_chunk_count = 0
        last_text_len = 0
        sr = 24000
        
        def sort_wav_files(files):
            def extract_num(f):
                match = re.search(r'wav_(\d+)\.wav', f)
                return int(match.group(1)) if match else 0
            return sorted(files, key=extract_num)
        
        try:
            cpp_request = {
                "debug_dir": output_dir,
                "stream": True,
                "round_idx": current_round_number
            }
            
            print(f"[streaming_generate] 调用 C++ decode: {json.dumps(cpp_request)} [单工]", flush=True)
            
            decode_task = asyncio.create_task(
                http_client.post(
                    f"{CPP_SERVER_URL}/v1/stream/decode",
                    json=cpp_request,
                    timeout=600.0
                )
            )
            
            cpp_output_base = CPP_OUTPUT_DIR
            round_dir = os.path.join(cpp_output_base, f"round_{current_round_number:03d}")
            tts_wav_dir = os.path.join(round_dir, "tts_wav")
            llm_debug_dir = os.path.join(round_dir, "llm_debug")
            
            print(f"[streaming_generate] 当前轮次: {current_round_number} [单工]", flush=True)
            print(f"  WAV 目录: {tts_wav_dir}", flush=True)
            
            max_wait = 1800
            check_interval = 0.01
            no_new_wav_count = 0
            max_no_new_wav = 1000
            decode_done = False
            chunk_texts = {}
            all_generated_text = []
            existing_wav_files = set()
            sent_wav_files = set()
            llm_chunk_idx = 0
            
            if os.path.exists(tts_wav_dir):
                existing_wav_files = set(f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav"))
            
            def read_chunk_text(llm_debug_dir, chunk_idx):
                chunk_dir = os.path.join(llm_debug_dir, f"chunk_{chunk_idx}")
                text_file = os.path.join(chunk_dir, "llm_text.txt")
                if os.path.exists(text_file):
                    try:
                        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                            return f.read().strip()
                    except:
                        pass
                return ""
            
            for _ in range(int(max_wait / check_interval)):
                await asyncio.sleep(check_interval)
                
                if is_breaking:
                    print(f"[streaming_generate] 检测到 break 标志，停止发送数据 [单工]", flush=True)
                    yield f"data: {json.dumps({'break': True, 'done': True, 'message': '用户打断'}, ensure_ascii=False)}\n\n"
                    break
                
                if decode_task.done() and not decode_done:
                    decode_done = True
                    try:
                        resp = decode_task.result()
                        if resp.status_code != 200:
                            print(f"[streaming_generate] C++ decode 返回错误: {resp.text}", flush=True)
                        else:
                            print(f"[streaming_generate] C++ decode 完成 [单工]", flush=True)
                    except Exception as e:
                        print(f"[streaming_generate] C++ decode 异常: {e}", flush=True)
                
                if os.path.exists(tts_wav_dir):
                    wav_files = [f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav")]
                    wav_files = sort_wav_files(wav_files)
                    
                    new_wav_files = [f for f in wav_files if f not in existing_wav_files and f not in sent_wav_files]
                    
                    for wav_file in new_wav_files:
                        wav_path = os.path.join(tts_wav_dir, wav_file)
                        
                        if not os.path.exists(wav_path):
                            await asyncio.sleep(0.05)
                            if not os.path.exists(wav_path):
                                continue
                        
                        match = re.search(r'wav_(\d+)\.wav', wav_file)
                        chunk_idx = int(match.group(1)) if match else sent_chunk_count
                        
                        try:
                            await asyncio.sleep(0.01)
                            
                            audio_data, audio_sr = sf.read(wav_path)
                            
                            if len(audio_data) == 0:
                                sent_wav_files.add(wav_file)
                                continue
                            
                            if first_chunk_time is None:
                                first_chunk_time = (time.time() - generate_start_time) * 1000
                                print(f"[⏱️ Generate 音频首响] {first_chunk_time:.1f}ms [单工]", flush=True)
                            
                            if audio_data.dtype != np.int16:
                                audio_data = (audio_data * 32767).astype(np.int16)
                            wav_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
                            
                            chunk_duration = len(audio_data) / audio_sr
                            chunk_durations.append(chunk_duration)
                            
                            if chunk_idx not in chunk_texts and os.path.exists(llm_debug_dir):
                                chunk_text = read_chunk_text(llm_debug_dir, llm_chunk_idx)
                                if chunk_text:
                                    chunk_texts[chunk_idx] = chunk_text
                                    all_generated_text.append(chunk_text)
                                    llm_chunk_idx += 1
                                    if first_text_time is None:
                                        first_text_time = (time.time() - generate_start_time) * 1000
                                        print(f"[⏱️ Generate 文本首响] {first_text_time:.1f}ms [单工]", flush=True)
                            
                            chunk_data = {
                                "chunk_idx": sent_chunk_count,
                                "chunk_data": {
                                    "wav": wav_base64,
                                    "sample_rate": int(audio_sr)
                                }
                            }
                            
                            if chunk_idx in chunk_texts:
                                chunk_data["chunk_data"]["text"] = chunk_texts[chunk_idx]
                                last_text_len += len(chunk_texts[chunk_idx])
                                print(f"[Chunk #{chunk_idx}] 发送 {wav_file} ({chunk_duration:.3f}s) + 文本 [单工]", flush=True)
                            else:
                                print(f"[Chunk #{chunk_idx}] 发送 {wav_file} ({chunk_duration:.3f}s) [单工]", flush=True)
                            
                            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                            
                            sent_wav_files.add(wav_file)
                            sent_chunk_count += 1
                            
                        except FileNotFoundError:
                            print(f"[Chunk #{chunk_idx}] 文件尚未就绪，稍后重试 [单工]", flush=True)
                        except Exception as e:
                            print(f"[Chunk #{chunk_idx}] 读取失败: {e} [单工]", flush=True)
                            sent_wav_files.add(wav_file)
                    
                    # 检查结束标记
                    done_flag_path = os.path.join(tts_wav_dir, "generation_done.flag")
                    if os.path.exists(done_flag_path):
                        try:
                            with open(done_flag_path, 'r') as f:
                                last_wav_idx = int(f.read().strip())
                            last_wav_file = f"wav_{last_wav_idx}.wav"
                            if last_wav_file in sent_wav_files or last_wav_file in existing_wav_files:
                                print(f"[streaming_generate] 所有 wav 已发送，立即结束 [单工]", flush=True)
                                break
                        except:
                            pass
                    
                    current_new_count = len([f for f in wav_files if f not in existing_wav_files])
                    if current_new_count == len(sent_wav_files):
                        no_new_wav_count += 1
                        if decode_done and no_new_wav_count >= 30000:
                            print(f"[streaming_generate] 超时退出 [单工]", flush=True)
                            break
                    else:
                        no_new_wav_count = 0
                
                if decode_done and sent_chunk_count == 0:
                    no_new_wav_count += 1
                    if no_new_wav_count >= max_no_new_wav:
                        print(f"[streaming_generate] decode完成但无wav输出，超时退出 [单工]", flush=True)
                        break
            
            if not decode_task.done():
                print("[streaming_generate] 等待 C++ decode 完成... [单工]", flush=True)
                try:
                    await asyncio.wait_for(decode_task, timeout=30.0)
                except asyncio.TimeoutError:
                    print("[streaming_generate] C++ decode 超时 [单工]", flush=True)
            
            if all_generated_text:
                full_text = "".join(all_generated_text)
                print(f"\n[📝 完整生成文本] {full_text}\n", flush=True)
            
            total_generate_time = (time.time() - generate_start_time) * 1000
            total_audio_duration = sum(chunk_durations) if chunk_durations else 0
            overall_rtf = total_generate_time / 1000 / total_audio_duration if total_audio_duration > 0 else 0
            
            print(f"\n{'='*60}", flush=True)
            print(f"[⏱️ Generate 性能总结] [单工]", flush=True)
            print(f"  音频首响: {first_chunk_time:.1f}ms" if first_chunk_time else "  音频首响: N/A", flush=True)
            print(f"  总生成时间: {total_generate_time:.1f}ms", flush=True)
            print(f"  总音频时长: {total_audio_duration:.2f}s", flush=True)
            print(f"  整体 RTF: {overall_rtf:.2f}x {'✅' if overall_rtf < 1.0 else '⚠️'}", flush=True)
            print(f"  发送 Chunk 数量: {sent_chunk_count}", flush=True)
            print(f"{'='*60}\n", flush=True)
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[streaming_generate] 异常: {e}\n{error_detail} [单工]", flush=True)
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        
        with session_lock:
            current_round_number += 1
            # 🔧 [修复多轮 user prompt] 每轮结束后重置 prefill 计数器
            # 这样下一轮的 prefill 会从 cnt=0 开始，C++ 端能正确识别新一轮的开始
            global current_request_counter
            current_request_counter = 0
        
        # 推理结束后，在后台检查显存并在需要时重启
        def background_memory_check():
            time.sleep(1.0)  # 等待一会儿让当前请求完全结束
            if check_gpu_memory_and_restart_if_needed():
                print("[单工] 显存不足，已在后台触发重启", flush=True)
        
        threading.Thread(target=background_memory_check, daemon=True).start()
        
        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


async def _streaming_generate_duplex(generate_request_time):
    """双工模式的 streaming_generate 实现"""
    global current_round_number, is_breaking
    global global_sent_wav_count, global_parsed_line_count, global_parsed_texts, global_text_send_idx, global_sent_wav_files
    global wav_timing_log_file, last_wav_send_time
    
    output_dir = os.path.join(TEMP_DIR, f"session_{current_active_session_id}", f"round_{current_round_number:04d}", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    async def generate_stream():
        global current_round_number, global_sent_wav_count, global_parsed_line_count
        global global_parsed_texts, global_text_send_idx, is_breaking, global_sent_wav_files
        global wav_timing_log_file, last_wav_send_time
        import re
        
        generate_start_time = time.time()
        setup_time = (generate_start_time - generate_request_time) * 1000
        if setup_time > 10:
            print(f"[Generate] ⚠️ 请求处理延迟: {setup_time:.0f}ms [双工]", flush=True)
        first_chunk_time = None
        first_text_time = None
        chunk_durations = []
        sent_chunk_count = global_sent_wav_count
        last_text_len = 0
        is_listen = True
        
        def sort_wav_files(files):
            def extract_num(f):
                match = re.search(r'wav_(\d+)\.wav', f)
                return int(match.group(1)) if match else 0
            return sorted(files, key=extract_num)
        
        try:
            cpp_request = {
                "debug_dir": "./tools/omni/output",
                "stream": True
            }
            
            print(f"[streaming_generate] 调用 C++ decode: {json.dumps(cpp_request)} [双工]", flush=True)
            
            # 🔧 [多实例支持] 使用配置的输出目录
            cpp_output_base = CPP_OUTPUT_DIR
            tts_wav_dir = os.path.join(cpp_output_base, "tts_wav")
            llm_debug_dir = os.path.join(cpp_output_base, "llm_debug")
            
            all_generated_text = []
            end_of_turn = False
            
            def parse_llm_text_file():
                global global_parsed_line_count, global_parsed_texts
                text_file = os.path.join(llm_debug_dir, "llm_text.txt")
                new_count = 0
                if os.path.exists(text_file):
                    try:
                        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                        
                        for line in lines[global_parsed_line_count:]:
                            line = line.strip()
                            if not line:
                                continue
                            match = re.match(r'\[chunk_\d+\]\s*(.*)', line)
                            if match:
                                text = match.group(1).strip()
                                if text:
                                    global_parsed_texts.append(text)
                                    new_count += 1
                            else:
                                global_parsed_texts.append(line)
                                new_count += 1
                        
                        global_parsed_line_count = len(lines)
                    except Exception as e:
                        print(f"[Parse LLM Text] 解析失败: {e} [双工]", flush=True)
                return new_count
            
            def init_wav_timing_log():
                global wav_timing_log_file
                if wav_timing_log_file is None:
                    wav_timing_log_file = open(WAV_TIMING_LOG_PATH, 'a', encoding='utf-8')
                    wav_timing_log_file.write(f"\n{'='*80}\n")
                    wav_timing_log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}] 新会话开始 (双工模式)\n")
                    wav_timing_log_file.write(f"{'='*80}\n")
                    wav_timing_log_file.flush()
            
            print(f"[streaming_generate] 开始监控 [双工]:", flush=True)
            print(f"  WAV目录: {tts_wav_dir}", flush=True)
            
            wav_queue = asyncio.Queue()
            stop_wav_scanner = asyncio.Event()
            
            async def wav_scanner_coroutine():
                global last_wav_send_time, wav_timing_log_file, global_parsed_texts, global_text_send_idx, global_sent_wav_files
                nonlocal sent_chunk_count, first_chunk_time, first_text_time, last_text_len
                scan_interval = 0.05
                
                while not stop_wav_scanner.is_set():
                    try:
                        if not os.path.exists(tts_wav_dir):
                            await asyncio.sleep(scan_interval)
                            continue
                        
                        new_count = parse_llm_text_file()
                        if new_count > 0:
                            new_texts = global_parsed_texts[-new_count:]
                            all_generated_text.extend(new_texts)
                            if first_text_time is None and global_parsed_texts:
                                first_text_time = (time.time() - generate_start_time) * 1000
                                print(f"[⏱️ Generate 文本首响] {first_text_time:.1f}ms [双工]", flush=True)
                        
                        wav_files = [f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav")]
                        wav_files = sort_wav_files(wav_files)
                        
                        new_wav_files = [f for f in wav_files if f not in global_sent_wav_files]
                        
                        for wav_file in new_wav_files:
                            if wav_file in global_sent_wav_files:
                                continue
                            global_sent_wav_files.add(wav_file)
                            
                            wav_path = os.path.join(tts_wav_dir, wav_file)
                            match = re.search(r'wav_(\d+)\.wav', wav_file)
                            wav_idx = int(match.group(1)) if match else sent_chunk_count
                            
                            try:
                                await asyncio.sleep(0.02)
                                
                                file_mtime = os.path.getmtime(wav_path)
                                cpp_write_time = datetime.fromtimestamp(file_mtime)
                                
                                audio_data, audio_sr = sf.read(wav_path)
                                
                                if len(audio_data) == 0:
                                    continue
                                
                                if first_chunk_time is None:
                                    first_chunk_time = (time.time() - generate_start_time) * 1000
                                    print(f"[⏱️ Generate 音频首响] {first_chunk_time:.1f}ms [双工]", flush=True)
                                
                                if audio_data.dtype != np.int16:
                                    audio_data = (audio_data * 32767).astype(np.int16)
                                wav_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
                                
                                chunk_duration = len(audio_data) / audio_sr
                                chunk_durations.append(chunk_duration)
                                
                                chunk_text = ""
                                if global_text_send_idx < len(global_parsed_texts):
                                    chunk_text = global_parsed_texts[global_text_send_idx]
                                    global_text_send_idx += 1
                                
                                chunk_data = {
                                    "chunk_idx": sent_chunk_count,
                                    "chunk_data": {
                                        "wav": wav_base64,
                                        "sample_rate": int(audio_sr)
                                    }
                                }
                                
                                send_time = time.time()
                                send_datetime = datetime.fromtimestamp(send_time)
                                write_to_send_delay_ms = (send_time - file_mtime) * 1000
                                interval_from_last_ms = (send_time - last_wav_send_time) * 1000 if last_wav_send_time else 0
                                last_wav_send_time = send_time
                                
                                init_wav_timing_log()
                                wav_timing_log_file.write(
                                    f"{wav_file:<20} "
                                    f"{cpp_write_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:23]:<26} "
                                    f"{send_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:23]:<26} "
                                    f"{write_to_send_delay_ms:>10.1f}ms    "
                                    f"{interval_from_last_ms:>10.1f}ms    "
                                    f"{chunk_duration:>6.3f}s\n"
                                )
                                wav_timing_log_file.flush()
                                
                                if chunk_text:
                                    chunk_data["chunk_data"]["text"] = chunk_text
                                    last_text_len += len(chunk_text)
                                    print(f"[WAV #{sent_chunk_count}] 发送 {wav_file} ({chunk_duration:.3f}s) + 文本 | 延迟:{write_to_send_delay_ms:.0f}ms [双工]", flush=True)
                                else:
                                    print(f"[WAV #{sent_chunk_count}] 发送 {wav_file} ({chunk_duration:.3f}s) | 延迟:{write_to_send_delay_ms:.0f}ms [双工]", flush=True)
                                
                                await wav_queue.put(f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n")
                                sent_chunk_count += 1
                                
                            except Exception as e:
                                print(f"[WAV #{wav_idx}] 读取失败: {e} [双工]", flush=True)
                        
                        await asyncio.sleep(scan_interval)
                        
                    except Exception as e:
                        print(f"[WAV Scanner] 异常: {e} [双工]", flush=True)
                        await asyncio.sleep(scan_interval)
                
                print(f"[WAV Scanner] 停止，已发送 {sent_chunk_count} chunks [双工]", flush=True)
            
            wav_scanner_task = asyncio.create_task(wav_scanner_coroutine())
            
            http_start = time.time()
            async with http_client.stream(
                "POST",
                f"{CPP_SERVER_URL}/v1/stream/decode",
                json=cpp_request,
                timeout=600.0
            ) as response:
                http_connect_time = (time.time() - http_start) * 1000
                if http_connect_time > 50:
                    print(f"[Generate] ⚠️ HTTP连接延迟: {http_connect_time:.0f}ms [双工]", flush=True)
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(f"[streaming_generate] C++ decode 错误: {error_text.decode()} [双工]", flush=True)
                    stop_wav_scanner.set()
                    wav_scanner_task.cancel()
                    yield f"data: {json.dumps({'error': 'decode failed'}, ensure_ascii=False)}\n\n"
                    return
                
                buffer = ""
                should_exit = False
                
                sse_iterator = response.aiter_text().__aiter__()
                
                while not should_exit:
                    if is_breaking:
                        print(f"[streaming_generate] 检测到 break 标志，停止发送数据 [双工]", flush=True)
                        yield f"data: {json.dumps({'break': True, 'done': True, 'message': '用户打断'}, ensure_ascii=False)}\n\n"
                        should_exit = True
                        break
                    
                    try:
                        while True:
                            try:
                                wav_chunk = wav_queue.get_nowait()
                                yield wav_chunk
                            except asyncio.QueueEmpty:
                                break
                        
                        try:
                            chunk = await asyncio.wait_for(sse_iterator.__anext__(), timeout=0.1)
                            buffer += chunk
                        except asyncio.TimeoutError:
                            continue
                        except StopAsyncIteration:
                            break
                        
                        while "\n\n" in buffer or "\r\n\r\n" in buffer:
                            if "\r\n\r\n" in buffer:
                                event_str, buffer = buffer.split("\r\n\r\n", 1)
                            else:
                                event_str, buffer = buffer.split("\n\n", 1)
                            
                            for line in event_str.split("\n"):
                                line = line.strip()
                                if line.startswith("data: "):
                                    try:
                                        event_data = json.loads(line[6:])
                                        
                                        if 'is_listen' in event_data:
                                            new_is_listen = event_data['is_listen']
                                            if new_is_listen != is_listen:
                                                print(f"[streaming_generate] is_listen: {is_listen} -> {new_is_listen} [双工]", flush=True)
                                                is_listen = new_is_listen
                                        
                                        if 'end_of_turn' in event_data:
                                            end_of_turn = event_data['end_of_turn']
                                            if end_of_turn:
                                                print(f"[streaming_generate] end_of_turn=True [双工]", flush=True)
                                        
                                        if 'text' in event_data and event_data['text']:
                                            all_generated_text.append(event_data['text'])
                                            if first_text_time is None:
                                                first_text_time = (time.time() - generate_start_time) * 1000
                                                print(f"[streaming_generate] 文本首响: {first_text_time:.1f}ms [双工]", flush=True)
                                        
                                    except json.JSONDecodeError:
                                        pass
                            
                            while True:
                                try:
                                    wav_chunk = wav_queue.get_nowait()
                                    yield wav_chunk
                                except asyncio.QueueEmpty:
                                    break
                            
                            if is_listen:
                                # 🔧 [修复音频错位] is_listen=True 时，快速检查是否有残留音频
                                # 原问题：TTS 异步处理，可能还有未完成的音频
                                # 解决：非阻塞快速扫描，最多等待 300ms，有新音频就发送
                                quick_check_start = time.time()
                                quick_check_rounds = 0
                                while (time.time() - quick_check_start) < 0.05: # < 50ms
                                    quick_check_rounds += 1
                                    # 检查队列中是否有新的 wav
                                    found_new = False
                                    while True:
                                        try:
                                            wav_chunk = wav_queue.get_nowait()
                                            yield wav_chunk
                                            found_new = True
                                        except asyncio.QueueEmpty:
                                            break
                                    if not found_new and quick_check_rounds >= 2:
                                        break
                                    await asyncio.sleep(0.02)
                                
                                print(f"[streaming_generate] is_listen=True，已发送 {sent_chunk_count} chunks [双工]", flush=True)
                                yield f"data: {json.dumps({'is_listen': True, 'chunks_received': sent_chunk_count}, ensure_ascii=False)}\n\n"
                                should_exit = True
                                break
                            
                            if end_of_turn:
                                print(f"[streaming_generate] end_of_turn=True，已发送 {sent_chunk_count} chunks [双工]", flush=True)
                                should_exit = True
                                break
                    
                    except Exception as e:
                        print(f"[streaming_generate] 主循环异常: {e} [双工]", flush=True)
                        break
                
                print(f"[streaming_generate] SSE 流结束，等待 WAV 扫描完成... [双工]", flush=True)
                max_final_wait = 3.0
                no_new_wav_count = 0
                final_start = time.time()
                
                while (time.time() - final_start) < max_final_wait:
                    prev_count = sent_chunk_count
                    
                    while True:
                        try:
                            wav_chunk = wav_queue.get_nowait()
                            yield wav_chunk
                        except asyncio.QueueEmpty:
                            break
                    
                    if sent_chunk_count > prev_count:
                        no_new_wav_count = 0
                    else:
                        no_new_wav_count += 1
                        if no_new_wav_count >= 10:
                            print(f"[streaming_generate] 连续 {no_new_wav_count} 次无新 WAV，结束扫描 [双工]", flush=True)
                            break
                    
                    await asyncio.sleep(0.1)
            
            stop_wav_scanner.set()
            try:
                await asyncio.wait_for(wav_scanner_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                wav_scanner_task.cancel()
            
            if all_generated_text:
                full_text = "".join(all_generated_text)
                print(f"\n[📝 完整生成文本] {full_text}\n", flush=True)
            
            total_generate_time = (time.time() - generate_start_time) * 1000
            total_audio_duration = sum(chunk_durations) if chunk_durations else 0
            overall_rtf = total_generate_time / 1000 / total_audio_duration if total_audio_duration > 0 else 0
            
            print(f"\n{'='*60}", flush=True)
            print(f"[⏱️ Generate 性能总结] [双工]", flush=True)
            print(f"  音频首响: {first_chunk_time:.1f}ms" if first_chunk_time else "  音频首响: N/A", flush=True)
            print(f"  总生成时间: {total_generate_time:.1f}ms", flush=True)
            print(f"  总音频时长: {total_audio_duration:.2f}s", flush=True)
            print(f"  整体 RTF: {overall_rtf:.2f}x {'✅' if overall_rtf < 1.0 else '⚠️'}", flush=True)
            print(f"  发送 Chunk 数量: {sent_chunk_count}", flush=True)
            print(f"{'='*60}\n", flush=True)
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[streaming_generate] 异常: {e}\n{error_detail} [双工]", flush=True)
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        
        with session_lock:
            current_round_number += 1
            global_sent_wav_count = sent_chunk_count
            # 🔧 [修复多轮 user prompt] 每轮结束后重置 prefill 计数器
            # 这样下一轮的 prefill 会从 cnt=0 开始，C++ 端能正确识别新一轮的开始
            global current_request_counter
            current_request_counter = 0
        
        print(f"[Generate] 本轮结束，round_number={current_round_number}，已发送WAV={global_sent_wav_count} [双工]", flush=True)
        
        # 推理结束后，在后台检查显存并在需要时重启
        def background_memory_check():
            time.sleep(1.0)  # 等待一会儿让当前请求完全结束
            if check_gpu_memory_and_restart_if_needed():
                print("[双工] 显存不足，已在后台触发重启", flush=True)
        
        threading.Thread(target=background_memory_check, daemon=True).start()
        
        total_audio_duration = sum(chunk_durations) if chunk_durations else 0
        yield f"data: {json.dumps({'done': True, 'is_listen': is_listen, 'chunks_received': sent_chunk_count, 'audio_duration_seconds': total_audio_duration}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MiniCPMO C++ HTTP 服务器（统一版：支持单工/双工模式切换）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8060, help="服务器端口")
    parser.add_argument("--llamacpp-root", type=str, default=LLAMACPP_ROOT, 
                        help="llama.cpp-omni 根目录（必须指定，或设置 LLAMACPP_ROOT 环境变量）")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, 
                        help="GGUF 模型目录（必须指定，或设置 MODEL_DIR 环境变量）")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL,
                        help="LLM 模型文件名（可选，默认自动从 model-dir 检测）")
    parser.add_argument("--gpu-devices", type=str, default=DEFAULT_GPU_DEVICES, help="GPU 设备 (e.g., '0,1')")
    parser.add_argument("--duplex", action="store_true", help="默认使用双工模式")
    parser.add_argument("--simplex", action="store_true", help="默认使用单工模式（优先级高于 --duplex）")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="C++ 输出目录（默认基于端口号: ./tools/omni/output_<port>）")
    parser.add_argument("--vision-backend", type=str, default="metal", choices=["metal", "coreml"],
                        help="视觉编码器后端: metal(默认GPU) 或 coreml(ANE加速，macOS专用)")
    
    args = parser.parse_args()
    
    # ========== 参数验证 ==========
    # 1. 验证 LLAMACPP_ROOT
    llamacpp_root = args.llamacpp_root
    if not llamacpp_root:
        print("❌ 错误: 必须指定 --llamacpp-root 或设置 LLAMACPP_ROOT 环境变量", flush=True)
        print("   示例: python minicpmo_cpp_http_server.py --llamacpp-root /path/to/llama.cpp-omni --model-dir /path/to/gguf", flush=True)
        sys.exit(1)
    if not os.path.isdir(llamacpp_root):
        print(f"❌ 错误: LLAMACPP_ROOT 目录不存在: {llamacpp_root}", flush=True)
        sys.exit(1)
    # 更新全局变量
    LLAMACPP_ROOT = llamacpp_root
    
    # 2. 验证 MODEL_DIR
    model_dir = args.model_dir
    if not model_dir:
        print("❌ 错误: 必须指定 --model-dir 或设置 MODEL_DIR 环境变量", flush=True)
        print("   示例: python minicpmo_cpp_http_server.py --llamacpp-root /path/to/llama.cpp-omni --model-dir /path/to/gguf", flush=True)
        sys.exit(1)
    if not os.path.isdir(model_dir):
        print(f"❌ 错误: MODEL_DIR 目录不存在: {model_dir}", flush=True)
        sys.exit(1)
    
    # 3. 自动检测或验证 LLM 模型
    llm_model = args.llm_model
    if not llm_model:
        llm_model = auto_detect_llm_model(model_dir)
        if llm_model:
            print(f"✅ 自动检测到 LLM 模型: {llm_model}", flush=True)
        else:
            print(f"❌ 错误: 在 {model_dir} 中未找到 LLM GGUF 模型", flush=True)
            print("   请使用 --llm-model 手动指定，或确保目录中有 .gguf 文件", flush=True)
            sys.exit(1)
    else:
        llm_path = os.path.join(model_dir, llm_model)
        if not os.path.exists(llm_path):
            print(f"❌ 错误: LLM 模型文件不存在: {llm_path}", flush=True)
            sys.exit(1)
    
    # 更新全局变量
    globals()['LLAMACPP_ROOT'] = llamacpp_root
    globals()['DEFAULT_LLM_MODEL'] = llm_model
    
    # 4. 设置参考音频路径（如果未指定）
    if not globals()['FIXED_TIMBRE_PATH']:
        globals()['FIXED_TIMBRE_PATH'] = os.path.join(llamacpp_root, "tools/omni/assets/default_ref_audio.wav")
    FIXED_TIMBRE_PATH = globals()['FIXED_TIMBRE_PATH']
    
    # 5. 设置视觉编码器后端
    if args.vision_backend == "coreml":
        vision_coreml = os.path.join(model_dir, "vision", "coreml_minicpmo45_vit_all_f16.mlmodelc")
        if os.path.exists(vision_coreml):
            globals()['VISION_BACKEND'] = "coreml"
            print(f"✅ Vision backend: CoreML/ANE ({vision_coreml})", flush=True)
        else:
            print(f"⚠️  CoreML model not found at {vision_coreml}, falling back to Metal", flush=True)
            globals()['VISION_BACKEND'] = "metal"
    else:
        globals()['VISION_BACKEND'] = "metal"
        print(f"✅ Vision backend: Metal (GPU)", flush=True)
    VISION_BACKEND = globals()['VISION_BACKEND']
    
    # 确定默认模式：--simplex 优先级最高，否则看 --duplex
    if args.simplex:
        default_duplex_mode = False
    elif args.duplex:
        default_duplex_mode = True
    else:
        default_duplex_mode = False  # 默认单工
    
    # 🔧 [多实例支持] 设置输出目录（基于端口号，避免多实例冲突）
    # 🔧 [修复] 使用 globals() 更新全局变量，确保其他函数能访问到
    if args.output_dir:
        globals()['CPP_OUTPUT_DIR'] = args.output_dir
    else:
        # 默认基于端口号创建独立的输出目录
        globals()['CPP_OUTPUT_DIR'] = os.path.join(llamacpp_root, f"tools/omni/output_{args.port}")
    CPP_OUTPUT_DIR = globals()['CPP_OUTPUT_DIR']
    
    # 确保输出目录存在
    os.makedirs(CPP_OUTPUT_DIR, exist_ok=True)
    
    app.state.port = args.port
    app.state.model_dir = model_dir
    app.state.gpu_devices = args.gpu_devices
    app.state.default_duplex_mode = default_duplex_mode
    app.state.output_dir = CPP_OUTPUT_DIR  # 保存到 app.state
    
    mode_name = "双工" if default_duplex_mode else "单工"
    print(f"", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"MiniCPM-o C++ HTTP 服务器", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  HTTP 地址: http://{args.host}:{args.port}", flush=True)
    print(f"  健康检查: http://{args.host}:{args.port + 1}/health", flush=True)
    print(f"  默认模式: {mode_name}", flush=True)
    print(f"", flush=True)
    print(f"  LLAMACPP_ROOT: {llamacpp_root}", flush=True)
    print(f"  MODEL_DIR:     {model_dir}", flush=True)
    print(f"  LLM_MODEL:     {llm_model}", flush=True)
    print(f"  OUTPUT_DIR:    {CPP_OUTPUT_DIR}", flush=True)
    print(f"  REF_AUDIO:     {FIXED_TIMBRE_PATH}", flush=True)
    print(f"  VISION_BACKEND: {VISION_BACKEND}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"", flush=True)
    
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
