import requests
import base64
import json

SERVER_URL = "http://localhost:9060"

def test_simplex_mode(image_path, audio_path, text_prompt=None, system_prompt=None):
    """测试半双工模式：输入图片+音频+文字，获取纯文本输出

    Args:
        image_path: PNG图片路径
        audio_path: WAV音频路径
        text_prompt: 用户文字输入（可选，如不提供则不传文字）
        system_prompt: 系统提示词（可选，如不提供则使用默认）
    """

    # 1. 初始化会话（半双工模式）
    init_data = {
        "media_type": "omni",      # 支持图片+音频
        "duplex_mode": False,       # False = 半双工模式
        "language": "zh"            # 中文输出
    }

    # 如果提供系统提示词，设置为 suffix（插入在参考音频之后，作为助手行为指令）
    if system_prompt:
        init_data["system_prompt_prefix"] = "Do not care about what you hear."
        init_data["system_prompt_suffix"] = system_prompt

    resp = requests.post(f"{SERVER_URL}/omni/init_sys_prompt", json=init_data)
    session_id = resp.json().get("session_id")
    print(f"会话ID: {session_id}")
    
    # 2. 准备输入数据
    # 图片转base64
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    # 音频转base64 (WAV格式, 支持48kHz)
    with open(audio_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    
    # 3. 流式预填充（输入图片+音频+文字）
    prefill_data = {
        "session_id": session_id,
        "image": image_base64,      # PNG图片 (base64)
        "audio": audio_base64,      # WAV音频 (base64)
        "text": text_prompt,        # 用户文字输入（新增）
        "is_last_chunk": True
    }
    
    resp = requests.post(
        f"{SERVER_URL}/omni/streaming_prefill",
        json=prefill_data
    )
    print(f"预填充结果: {resp.json()}")
    
    # 4. 流式生成（获取纯文本输出）
    resp = requests.post(
        f"{SERVER_URL}/omni/streaming_generate",
        json={"session_id": session_id},
        stream=True,
        headers={"Accept": "text/event-stream"}
    )
    
    # 5. 解析并输出纯文本
    full_text = ""
    for line in resp.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if 'chunk_data' in data:
                    # 获取文本内容（不含音频）
                    text = data['chunk_data'].get('text', '')
                    if text:
                        full_text += text
                        print(f"文本输出: {text}")
    
    return full_text

# 执行测试
if __name__ == "__main__":
    # # 示例 1: 图片+音频+文字+自定义系统提示词
    # print("=" * 50)
    # print("测试 1: 图片+音频+文字+自定义系统提示词")
    # print("=" * 50)
    # result = test_simplex_mode(
    #     image_path="/home/weifeng/llm/models/openbmb/MiniCPM-o-4_5/assets/fossil.png",
    #     # audio_path="/home/weifeng/llm/models/openbmb/MiniCPM-o-4_5/assets/nezha.wav",
    #     audio_path="/home/weifeng/llm/bark-1.wav",
    #     text_prompt="用英文描述图片中的主要颜色是什么",  # 添加文字输入
    #     system_prompt="用英文描述你看到的画面"  # 自定义系统提示词
    # )
    # print(f"\n完整输出:\n{result}")

    # 示例 2: 图片+音频（无文字，与之前兼容）
    print("\n" + "=" * 50)
    print("测试 2: 图片+音频（无文字）")
    print("=" * 50)
    result = test_simplex_mode(
        image_path="/home/weifeng/llm/models/openbmb/MiniCPM-o-4_5/assets/fossil.png",
        # audio_path="/home/weifeng/llm/models/openbmb/MiniCPM-o-4_5/assets/nezha.wav",
        audio_path="/home/weifeng/llm/bark-1.wav",
        # text_prompt="图片中的主要颜色是什么",  # 添加文字输入
        system_prompt="用英文描述你看到的画面"  # 自定义系统提示词
    )
    print(f"\n完整输出:\n{result}")

