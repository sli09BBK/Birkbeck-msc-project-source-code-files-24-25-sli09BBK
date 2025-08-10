

"""
一键式高级用户分析与可视化启动脚本
整合了分析流程和可视化仪表盘功能
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time
import json
import logging
import threading
import http.server
import socketserver

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from X_data_section.advanced_user_analysis import AdvancedUserAnalysis
except ImportError:
    print("错误：无法导入AdvancedUserAnalysis模块，请确保已安装所有依赖")
    print("提示：运行 pip install mysql-connector-python jieba snownlp scikit-learn xgboost lightgbm matplotlib")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='一键运行高级用户分析与可视化仪表盘')
    
    # 数据库连接参数
    parser.add_argument('--host', default='127.0.0.1', help='数据库主机地址')
    parser.add_argument('--user', default='root', help='数据库用户名')
    parser.add_argument('--password', default='root', help='数据库密码')
    parser.add_argument('--database', default='rednote', help='数据库名称')
    
    # 运行选项
    parser.add_argument('--with-dashboard', action='store_true', help='运行分析后启动可视化仪表盘')
    parser.add_argument('--dashboard-only', action='store_true', help='仅启动可视化仪表盘，不运行分析')
    parser.add_argument('--port', type=int, default=8000, help='仪表盘Web服务器端口')
    parser.add_argument('--output', default='X_rednote_result', help='输出目录')
    parser.add_argument('--min-anomaly-percent', type=float, default=0.05, help='最小异常检测百分比')
    parser.add_argument('--adjust-r2', action='store_true', help='调整不切实际的高R²分数')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    return parser.parse_args()

def run_analysis(args):
    """运行完整分析流程"""
    try:
        logger.info("开始运行高级用户分析...")
        
        # 数据库配置
        db_config = {
            'host': args.host,
            'user': args.user,
            'password': args.password,
            'database': args.database,
            'charset': 'utf8mb4'
        }
        
        # 创建分析器
        analyzer = AdvancedUserAnalysis(db_config)
        analyzer.output_dir = args.output
        
        # 确保输出目录存在
        os.makedirs(args.output, exist_ok=True)
        
        # 连接数据库
        if not analyzer.connect_to_database():
            logger.error("数据库连接失败，退出")
            return False
        
        # 运行完整分析
        results = analyzer.run_complete_analysis()
        
        if results:
            logger.info(f"分析完成，结果保存在: {results['output_dir']}")
            
            # 打印关键指标
            if 'insights' in results and 'overview' in results['insights']:
                overview = results['insights']['overview']
                logger.info(f"总用户数: {overview.get('total_users', 0)}")
                logger.info(f"平均内容数: {overview.get('avg_content_count', 0):.2f}")
                logger.info(f"异常用户比例: {overview.get('anomaly_user_ratio', 0)*100:.2f}%")
            
            # 准备仪表盘数据
            prepare_dashboard_data(results)
            
            return True
        else:
            logger.error("分析失败")
            return False
            
    except Exception as e:
        logger.error(f"运行分析时出错: {e}", exc_info=args.debug)
        return False
    finally:
        # 确保关闭数据库连接
        try:
            analyzer.close_connection()
            logger.info("数据库连接已关闭")
        except:
            pass

def prepare_dashboard_data(results):
    """准备仪表盘数据"""
    try:
        # 创建仪表盘数据目录
        dashboard_data_dir = os.path.join(os.getcwd(), 'dashboard_data')
        os.makedirs(dashboard_data_dir, exist_ok=True)
        
        # 保存用户洞察数据
        if 'insights' in results:
            with open(os.path.join(dashboard_data_dir, 'user_insights.json'), 'w', encoding='utf-8') as f:
                json.dump(results['insights'], f, ensure_ascii=False, indent=2)
        
        # 保存模型结果数据
        if 'model_results' in results:
            with open(os.path.join(dashboard_data_dir, 'model_results.json'), 'w', encoding='utf-8') as f:
                json.dump(results['model_results'], f, ensure_ascii=False, indent=2)
        
        # 复制可视化图片到仪表盘数据目录
        if 'output_dir' in results:
            import shutil
            for img_file in ['clustering_visualization.png', 'feature_importance.png', 
                           'model_performance.png', 'cluster_radar_chart.png', 
                           'cluster_distribution.png']:
                src_path = os.path.join(results['output_dir'], img_file)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, os.path.join(dashboard_data_dir, img_file))
        
        logger.info(f"仪表盘数据准备完成: {dashboard_data_dir}")
        return True
    except Exception as e:
        logger.error(f"准备仪表盘数据时出错: {e}")
        return False

def start_dashboard_server(port=8000):
    """启动仪表盘Web服务器"""
    try:
        # 检查dashboard.html是否存在
        dashboard_path = os.path.join(os.getcwd(), 'X_rednote_visualization/dashboard.html')
        if not os.path.exists(dashboard_path):
            # 如果不存在，显示错误消息
            logger.error(f"找不到仪表盘文件: {dashboard_path}")
            return False
            
        # 创建HTTP服务器
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("", port), handler)
        
        logger.info(f"启动仪表盘服务器，端口: {port}")
        logger.info(f"请访问: http://localhost:{port}/X_rednote_visualization/dashboard.html")
        
        # 在浏览器中打开仪表盘
        webbrowser.open(f'http://localhost:{port}/X_rednote_visualization/dashboard.html')
        
        # 启动服务器
        httpd.serve_forever()
    except Exception as e:
        logger.error(f"启动仪表盘服务器时出错: {e}")
        return False

def main():
    """主函数"""
    args = parse_args()
    
    # 设置调试模式
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 如果只启动仪表盘
    if args.dashboard_only:
        # 在新线程中启动仪表盘服务器
        dashboard_thread = threading.Thread(target=start_dashboard_server, args=(args.port,))
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        logger.info("仪表盘服务器已启动，按Ctrl+C退出")
        try:
            # 保持主线程运行
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("服务器已停止")
        
        return 0
    
    # 运行分析
    success = run_analysis(args)
    
    # 如果分析成功且需要启动仪表盘
    if success and args.with_dashboard:
        # 在新线程中启动仪表盘服务器
        dashboard_thread = threading.Thread(target=start_dashboard_server, args=(args.port,))
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        logger.info("仪表盘服务器已启动，按Ctrl+C退出")
        try:
            # 保持主线程运行
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("服务器已停止")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 