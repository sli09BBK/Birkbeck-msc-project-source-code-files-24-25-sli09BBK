#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小红书数据分析系统安装脚本

这个脚本将帮助您快速设置和配置整个系统。
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional

class SystemSetup:
    """系统安装配置类"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "X_data_section"
        self.requirements_file = self.project_root / "requirements.txt"
        self.config_file = self.data_dir / "database_config.json"
        
        # 系统信息
        self.system = platform.system().lower()
        self.python_version = sys.version_infon
        
        print(f"🚀 小红书数据分析系统安装程序")
        print(f"📁 项目路径: {self.project_root}")
        print(f"🐍 Python版本: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        print(f"💻 操作系统: {platform.system()} {platform.release()}")
        print("="*60)
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        print("🔍 检查Python版本...")
        
        if self.python_version < (3, 8):
            print("❌ Python版本过低，需要Python 3.8或更高版本")
            print(f"   当前版本: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
            return False
        
        print(f"✅ Python版本检查通过: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        return True
    
    def check_mysql_installation(self) -> bool:
        """检查MySQL安装"""
        print("🔍 检查MySQL安装...")
        
        try:
            # 尝试连接MySQL命令行
            result = subprocess.run(['mysql', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                version_info = result.stdout.strip()
                print(f"✅ MySQL已安装: {version_info}")
                return True
            else:
                print("❌ MySQL未安装或未添加到PATH")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ MySQL未安装或未添加到PATH")
            return False
    
    def install_dependencies(self) -> bool:
        """安装Python依赖"""
        print("📦 安装Python依赖包...")
        
        if not self.requirements_file.exists():
            print(f"❌ 依赖文件不存在: {self.requirements_file}")
            return False
        
        try:
            # 升级pip
            print("   升级pip...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            
            # 安装依赖
            print("   安装项目依赖...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(self.requirements_file)], 
                         check=True, capture_output=True)
            
            print("✅ 依赖安装完成")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖安装失败: {e}")
            print("   请手动运行: pip install -r requirements.txt")
            return False
    
    def setup_database_config(self) -> bool:
        """设置数据库配置"""
        print("🗄️ 配置数据库连接...")
        
        # 默认配置
        default_config = {
            "host": "127.0.0.1",
            "user": "root",
            "password": "root",
            "database": "rednote",
            "port": 3306
        }
        
        # 如果配置文件已存在，询问是否覆盖
        if self.config_file.exists():
            response = input("   数据库配置文件已存在，是否重新配置？(y/N): ")
            if response.lower() != 'y':
                print("   跳过数据库配置")
                return True
        
        print("   请输入数据库连接信息:")
        
        # 获取用户输入
        config = {}
        config['host'] = input(f"   主机地址 [{default_config['host']}]: ") or default_config['host']
        config['user'] = input(f"   用户名 [{default_config['user']}]: ") or default_config['user']
        config['password'] = input("   密码: ")
        config['database'] = input(f"   数据库名 [{default_config['database']}]: ") or default_config['database']
        
        port_input = input(f"   端口 [{default_config['port']}]: ")
        config['port'] = int(port_input) if port_input else default_config['port']
        
        # 保存配置
        try:
            self.data_dir.mkdir(exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 数据库配置已保存: {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ 保存数据库配置失败: {e}")
            return False
    
    def test_database_connection(self) -> bool:
        """测试数据库连接"""
        print("🔗 测试数据库连接...")
        
        if not self.config_file.exists():
            print("❌ 数据库配置文件不存在")
            return False
        
        try:
            # 导入数据库模块
            sys.path.append(str(self.data_dir))
            from X_data_section.database_setup import DatabaseConfig
            
            # 加载配置
            config = DatabaseConfig.load_config(str(self.config_file))
            
            # 测试连接
            import pymysql
            connection = pymysql.connect(
                host=config['host'],
                user=config['user'],
                password=config['password'],
                port=int(config['port']),
                charset='utf8mb4'
            )
            
            connection.close()
            print("✅ 数据库连接测试成功")
            return True
            
        except Exception as e:
            print(f"❌ 数据库连接测试失败: {e}")
            print("   请检查数据库配置和MySQL服务状态")
            return False
    
    def setup_database_schema(self) -> bool:
        """设置数据库结构"""
        print("🏗️ 初始化数据库结构...")
        
        try:
            # 运行数据库设置脚本
            setup_script = self.data_dir / "database_setup.py"
            
            if not setup_script.exists():
                print(f"❌ 数据库设置脚本不存在: {setup_script}")
                return False
            
            # 执行数据库设置
            result = subprocess.run([
                sys.executable, str(setup_script),
                '--config', str(self.config_file)
            ], capture_output=True, text=True, cwd=str(self.data_dir))
            
            if result.returncode == 0:
                print("✅ 数据库结构初始化完成")
                return True
            else:
                print(f"❌ 数据库结构初始化失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 数据库结构初始化失败: {e}")
            return False
    
    def create_sample_data(self) -> bool:
        """创建示例数据"""
        print("📝 创建示例配置文件...")
        
        try:
            # 创建示例CSV文件
            sample_csv = self.project_root / "sample_data.csv"
            
            if not sample_csv.exists():
                sample_data = '''标题,内容,作者,发布时间,点赞数,评论数,分享数
"美食分享","今天做了超好吃的蛋糕","用户A","2024-12-19 10:00:00","100","20","5"
"旅行日记","三亚的海滩真的太美了","用户B","2024-12-19 11:00:00","200","30","10"
"护肤心得","这款面膜效果很好","用户C","2024-12-19 12:00:00","150","25","8"'''
                
                with open(sample_csv, 'w', encoding='utf-8') as f:
                    f.write(sample_data)
                
                print(f"✅ 示例数据文件已创建: {sample_csv}")
            
            return True
            
        except Exception as e:
            print(f"❌ 创建示例数据失败: {e}")
            return False
    
    def print_usage_instructions(self):
        """打印使用说明"""
        print("\n" + "="*60)
        print("🎉 安装完成！使用说明:")
        print("="*60)
        
        print("\n📖 快速开始:")
        print("   1. 图形界面:")
        print(f"      cd {self.data_dir}")
        print("      python gui_interface.py")
        
        print("\n   2. 命令行处理:")
        print(f"      cd {self.data_dir}")
        print("      python main_data_pipeline.py --input ../sample_data.csv")
        
        print("\n   3. 数据库迁移:")
        print(f"      cd {self.data_dir}")
        print("      python database_migration.py --csv-dir ../")
        
        print("\n📚 重要文件:")
        print(f"   - 配置文件: {self.config_file}")
        print(f"   - 示例数据: {self.project_root}/sample_data.csv")
        print(f"   - 文档: {self.project_root}/README.md")
        
        print("\n🔧 故障排除:")
        print("   - 查看日志文件了解详细错误信息")
        print("   - 确保MySQL服务正在运行")
        print("   - 检查防火墙和网络连接")
        
        print("\n📞 获取帮助:")
        print("   - 查看README.md文档")
        print("   - 检查GitHub Issues")
        print("   - 联系项目维护者")
        
        print("\n" + "="*60)
    
    def run_setup(self) -> bool:
        """运行完整安装流程"""
        print("开始系统安装...\n")
        
        steps = [
            ("检查Python版本", self.check_python_version),
            ("检查MySQL安装", self.check_mysql_installation),
            ("安装Python依赖", self.install_dependencies),
            ("配置数据库连接", self.setup_database_config),
            ("测试数据库连接", self.test_database_connection),
            ("初始化数据库结构", self.setup_database_schema),
            ("创建示例数据", self.create_sample_data)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}")
            try:
                success = step_func()
                if not success:
                    failed_steps.append(step_name)
                    
                    # 对于关键步骤，询问是否继续
                    if step_name in ["检查Python版本", "安装Python依赖"]:
                        response = input(f"   ⚠️  {step_name}失败，是否继续？(y/N): ")
                        if response.lower() != 'y':
                            print("   安装中止")
                            return False
                            
            except KeyboardInterrupt:
                print("\n⏹️  用户中断安装")
                return False
            except Exception as e:
                print(f"   ❌ {step_name}发生异常: {e}")
                failed_steps.append(step_name)
        
        # 安装总结
        print("\n" + "="*60)
        print("📊 安装总结")
        print("="*60)
        
        if failed_steps:
            print(f"⚠️  部分步骤失败 ({len(failed_steps)}/{len(steps)}):")
            for step in failed_steps:
                print(f"   - {step}")
            print("\n💡 建议手动完成失败的步骤")
        else:
            print("🎉 所有步骤完成成功！")
        
        # 显示使用说明
        self.print_usage_instructions()
        
        return len(failed_steps) == 0

def main():
    """主函数"""
    try:
        setup = SystemSetup()
        success = setup.run_setup()
        
        if success:
            print("\n🎊 安装完成！系统已准备就绪。")
            return 0
        else:
            print("\n⚠️  安装完成，但存在一些问题。请查看上述信息。")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  安装被用户中断")
        return 1
    except Exception as e:
        print(f"\n💥 安装过程中发生错误: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)