#!/usr/bin/env python3
"""
验证requirements.txt中的包是否兼容Python 3.11.8
"""
import sys
import subprocess
import pkg_resources
from packaging import version


def check_python_version():
    """检查Python版本"""
    required = (3, 11, 8)
    current = sys.version_info

    print(f"🔍 检查Python版本: {current.major}.{current.minor}.{current.micro}")

    if current < required:
        print(f"❌ Python版本过低，需要 >= {required[0]}.{required[1]}.{required[2]}")
        return False
    else:
        print(f"✅ Python版本满足要求")
        return True


def read_requirements(file_path='requirements.txt'):
    """读取requirements.txt文件"""
    packages = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # 处理带版本号的包
                if '==' in line:
                    name, ver = line.split('==')
                    packages.append((name.strip(), ver.strip()))
                else:
                    packages.append((line.strip(), None))
    return packages


def check_imports(packages):
    """尝试导入包并检查版本"""
    results = []

    print("\n📦 检查包导入和版本...")
    for name, required_version in packages:
        try:
            # 尝试导入
            module = __import__(name.replace('-', '_'))

            # 获取版本
            installed_version = getattr(module, '__version__', None)
            if not installed_version:
                # 尝试通过pkg_resources获取
                installed_version = pkg_resources.get_distribution(name).version

            # 版本检查
            status = "✅"
            if required_version:
                if version.parse(installed_version) < version.parse(required_version):
                    status = "⚠️"
                    message = f"版本过低: {installed_version} < {required_version}"
                else:
                    message = f"版本OK: {installed_version} >= {required_version}"
            else:
                message = f"版本: {installed_version}"

            results.append({
                'name': name,
                'status': status,
                'installed': installed_version,
                'required': required_version,
                'message': message
            })

        except ImportError as e:
            results.append({
                'name': name,
                'status': "❌",
                'installed': None,
                'required': required_version,
                'message': f"导入失败: {e}"
            })
        except Exception as e:
            results.append({
                'name': name,
                'status': "❓",
                'installed': None,
                'required': required_version,
                'message': f"未知错误: {e}"
            })

    return results


def check_system_dependencies():
    """检查系统级依赖"""
    print("\n🔧 检查系统级依赖...")

    deps = [
        ('git', '--version', 'Git版本控制'),
        ('docker', '--version', 'Docker容器'),
        ('docker-compose', '--version', 'Docker Compose'),
    ]

    for cmd, version_arg, desc in deps:
        try:
            result = subprocess.run(
                [cmd, version_arg],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"  ✅ {desc}: 已安装")
                # 提取版本信息
                version_line = result.stdout.split('\n')[0]
                print(f"      版本: {version_line}")
            else:
                print(f"  ⚠️  {desc}: 未安装或不可用")
        except FileNotFoundError:
            print(f"  ❌ {desc}: 未安装")
        except Exception as e:
            print(f"  ❓ {desc}: 检查失败 - {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("Trading Chatbot 环境验证工具")
    print("=" * 60)

    # 1. 检查Python版本
    if not check_python_version():
        sys.exit(1)

    # 2. 读取requirements
    try:
        packages = read_requirements()
        print(f"\n📄 从requirements.txt读取到 {len(packages)} 个包")
    except FileNotFoundError:
        print("❌ 未找到requirements.txt文件")
        sys.exit(1)

    # 3. 检查包
    results = check_imports(packages)

    # 打印结果
    print("\n📊 检查结果汇总:")
    print("-" * 80)
    print(f"{'包名':<25} {'状态':<5} {'已安装':<15} {'要求':<15} {'说明'}")
    print("-" * 80)

    success = 0
    warning = 0
    error = 0

    for r in results:
        print(f"{r['name']:<25} {r['status']:<5} "
              f"{r['installed'] or 'N/A':<15} "
              f"{r['required'] or '任意':<15} "
              f"{r['message']}")

        if r['status'] == '✅':
            success += 1
        elif r['status'] == '⚠️':
            warning += 1
        else:
            error += 1

    print("-" * 80)
    print(f"总计: ✅ {success} | ⚠️  {warning} | ❌ {error}")

    # 4. 检查系统依赖
    check_system_dependencies()

    print("\n" + "=" * 60)
    if error == 0 and warning == 0:
        print("🎉 所有检查通过！环境准备就绪。")
    elif error == 0:
        print("⚠️  有警告但无错误，环境基本可用。")
    else:
        print("❌ 存在错误，请修复后重试。")
        sys.exit(1)


if __name__ == '__main__':
    main()