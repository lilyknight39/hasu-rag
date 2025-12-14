import pkgutil
import langchain
import langchain_community
import importlib

print(f"🔎 LangChain Path: {langchain.__path__}")
print(f"🔎 Community Path: {langchain_community.__path__}")

# 1. 检查 langchain 下有哪些子模块
print("\n📦 langchain 子模块列表:")
for loader, module_name, is_pkg in pkgutil.walk_packages(langchain.__path__):
    if "retriever" in module_name:
        print(f" - langchain.{module_name}")

# 2. 检查 langchain_community 下有哪些子模块
print("\n📦 langchain_community 子模块列表:")
for loader, module_name, is_pkg in pkgutil.walk_packages(langchain_community.__path__):
    if "retriever" in module_name:
        print(f" - langchain_community.{module_name}")

# 3. 暴力搜索类名
print("\n🕵️‍♀️ 正在寻找 ContextualCompressionRetriever...")
targets = [
    "langchain.retrievers",
    "langchain.retrievers.contextual_compression",
    "langchain_community.retrievers",
    "langchain.chains", # 有时候会放在 chains 里
    "langchain_core.retrievers"
]

for t in targets:
    try:
        module = importlib.import_module(t)
        if hasattr(module, "ContextualCompressionRetriever"):
            print(f"✅ 找到了！请使用: from {t} import ContextualCompressionRetriever")
            break
    except ImportError:
        pass
    except Exception as e:
        print(f"   (检查 {t} 时出错: {e})")