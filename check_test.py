import os
import sys
# 用于检测是否有效
# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("1. dotenv 模块加载成功")
except ImportError:
    print("❌ 错误：没有安装 python-dotenv，请运行 pip install python-dotenv")

# 检查 Key 是否读取成功
api_key = os.getenv("LLM_BINDING_API_KEY")
host = os.getenv("LLM_BINDING_HOST")

if api_key and api_key.startswith("sk-"):
    print(f"2. ✅ 成功读取 API Key: {api_key[:6]}******")
else:
    print("2. ❌ 错误：未读取到 API Key，请检查 .env 文件是否存在且内容正确")

if host and "deepseek" in host:
    print(f"3. ✅ 成功配置 DeepSeek 地址: {host}")
else:
    print(f"3. ⚠️ 警告：Host 地址为 {host}，确认这是你想要的吗？(如果是 OpenAI 官方则忽略此警告)")

# 检查 raganything 是否安装
try:
    import raganything
    print(f"4. ✅ 成功检测到 raganything 包，安装路径: {os.path.dirname(raganything.__file__)}")
except ImportError:
    print("4. ❌ 错误：找不到 raganything 包。请在终端运行: pip install -e .")

print("\n检测结束。如果全是对钩，就可以运行 Example 了！")