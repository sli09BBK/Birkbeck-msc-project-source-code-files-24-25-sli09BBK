#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
仪表盘启动脚本
用于启动高级用户分析可视化仪表盘
"""

import os
import sys
import argparse
import webbrowser
import threading
import http.server
import socketserver
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='启动高级用户分析可视化仪表盘')
    parser.add_argument('--port', type=int, default=8000, help='Web服务器端口')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    return parser.parse_args()

def start_dashboard_server(port=8000, open_browser=True):
    """启动仪表盘Web服务器"""
    try:
        # 检查X_rednote_visualization/dashboard.html是否存在
        dashboard_path = os.path.join(os.getcwd(), 'X_rednote_visualization/dashboard.html')
        if not os.path.exists(dashboard_path):
            logger.error(f"找不到仪表盘文件: {dashboard_path}")
            return False
            
        # 创建HTTP服务器
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("", port), handler)
        
        logger.info(f"启动仪表盘服务器，端口: {port}")
        logger.info(f"请访问: http://localhost:{port}/X_rednote_visualization/dashboard.html")
        
        # 在浏览器中打开仪表盘
        if open_browser:
            webbrowser.open(f'http://localhost:{port}/X_rednote_visualization/dashboard.html')
        
        # 启动服务器
        httpd.serve_forever()
    except Exception as e:
        logger.error(f"启动仪表盘服务器时出错: {e}")
        return False

def main():
    """主函数"""
    args = parse_args()
    
    # 在新线程中启动仪表盘服务器
    dashboard_thread = threading.Thread(
        target=start_dashboard_server, 
        args=(args.port, not args.no_browser)
    )
    dashboard_thread.daemon = True
    dashboard_thread.start()
    
    logger.info("仪表盘服务器已启动，按Ctrl+C退出")
    try:
        # 保持主线程运行
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 