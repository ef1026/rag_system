import os

print("当前工作目录:", os.getcwd())
print("\n目录下所有文件列表:")

files = os.listdir(".")
env_found = False

for f in files:
    if "env" in f:
        print(f"👉 发现疑似配置文件: [{f}]")
        if f == ".env":
            print("   ✅ 文件名完全正确！")
            env_found = True
        elif f == ".env.txt":
            print("   ❌ 错误：文件变成了 .txt 文本文件！请重命名。")
        else:
            print(f"   ⚠️  警告：文件名不对，程序可能读不到。")

if not env_found:
    print("\n❌ 严重错误：根本没找到名字叫 .env 的文件！")
    print("   请检查你是不是把它建到 examples 文件夹里去了？")