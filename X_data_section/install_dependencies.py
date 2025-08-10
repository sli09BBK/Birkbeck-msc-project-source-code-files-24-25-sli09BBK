#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
依赖安装脚本
用于安装advanced_user_analysis.py所需的所有依赖库
"""

import subprocess
import sys
import os
import platform

def check_pip():
    """检查pip是否已安装"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'])
        return True
    except subprocess.CalledProcessError:
        print("pip未安装，请先安装pip")
        return False

def install_requirements():
    """安装requirements.txt中的依赖"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    req_path = os.path.join(script_dir, 'requirements.txt')
    
    if not os.path.exists(req_path):
        print(f"错误：找不到requirements.txt文件")
        return False
    
    print("开始安装依赖...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_path])
        print("依赖安装完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"安装依赖时出错：{e}")
        return False

def check_optional_dependencies():
    """检查可选依赖是否已安装"""
    optional_deps = ['lightgbm', 'catboost']
    missing_deps = []
    
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"✓ {dep} 已安装")
        except ImportError:
            missing_deps.append(dep)
            print(f"✗ {dep} 未安装")
    
    return missing_deps

def install_missing_optional(missing_deps):
    """安装缺失的可选依赖"""
    if not missing_deps:
        return
    
    print("\n以下可选依赖未安装，但对模型优化有帮助：")
    for dep in missing_deps:
        print(f"- {dep}")
    
    choice = input("\n是否安装这些可选依赖？(y/n): ")
    if choice.lower() != 'y':
        print("跳过安装可选依赖")
        return
    
    for dep in missing_deps:
        print(f"\n正在安装 {dep}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"{dep} 安装成功！")
        except subprocess.CalledProcessError as e:
            print(f"安装 {dep} 时出错：{e}")

def main():
    """主函数"""
    print("=" * 50)
    print("Advanced User Analysis 依赖安装程序")
    print("=" * 50)
    
    # 检查系统信息
    print(f"Python版本: {platform.python_version()}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    
    # 检查pip
    if not check_pip():
        return
    
    # 安装基本依赖
    if not install_requirements():
        return
    
    # 检查并安装可选依赖
    print("\n检查可选依赖...")
    missing_deps = check_optional_dependencies()
    install_missing_optional(missing_deps)
    
    print("\n依赖安装完成！现在可以运行advanced_user_analysis.py了")
    print("=" * 50)

if __name__ == "__main__":
    main() 