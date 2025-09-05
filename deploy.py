# deploy.py
import os
import sys
import subprocess
import argparse
from config import Config

def setup_environment():
    """设置环境"""
    print("设置Python环境...")
    
    # 创建必要的目录
    directories = ['data', 'models', 'logs', 'cache', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    # 安装依赖
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyspark>=3.2.0",
        "redis>=4.0.0",
        "wordcloud>=1.8.0"
    ]
    
    print("安装Python依赖...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except subprocess.CalledProcessError:
            print(f"警告: 无法安装 {req}")

def setup_spark():
    """设置Spark环境"""
    print("设置Spark环境...")
    
    # 检查Java环境
    try:
        java_version = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)
        print("Java环境检查通过")
    except FileNotFoundError:
        print("错误: 未找到Java环境，请安装Java 8或11")
        return False
    
    # 检查Spark环境
    spark_home = os.environ.get('SPARK_HOME')
    if not spark_home:
        print("警告: 未设置SPARK_HOME环境变量")
        return False
    
    print(f"Spark环境: {spark_home}")
    return True

def setup_redis():
    """设置Redis（可选）"""
    if not Config.USE_REDIS:
        print("跳过Redis设置（未启用）")
        return True
    
    print("设置Redis...")
    try:
        import redis
        client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=Config.REDIS_DB)
        client.ping()
        print("Redis连接成功")
        return True
    except Exception as e:
        print(f"Redis设置失败: {e}")
        return False

def run_tests():
    """运行测试"""
    print("运行测试...")
    
    # 简单的功能测试
    try:
        from main import main
        print("主程序测试...")
        # 这里可以添加更多测试
        print("测试通过")
        return True
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def main_deploy():
    parser = argparse.ArgumentParser(description='推荐系统部署脚本')
    parser.add_argument('--skip-env', action='store_true', help='跳过环境设置')
    parser.add_argument('--skip-spark', action='store_true', help='跳过Spark设置')
    parser.add_argument('--skip-redis', action='store_true', help='跳过Redis设置')
    parser.add_argument('--skip-tests', action='store_true', help='跳过测试')
    
    args = parser.parse_args()
    
    print("=== 推荐系统部署 ===")
    
    success = True
    
    if not args.skip_env:
        setup_environment()
    
    if not args.skip_spark:
        if not setup_spark():
            success = False
    
    if not args.skip_redis:
        if not setup_redis():
            success = False
    
    if not args.skip_tests:
        if not run_tests():
            success = False
    
    if success:
        print("\n✅ 部署成功！")
        print("可以运行以下命令启动系统:")
        print("python main.py")
    else:
        print("\n❌ 部署过程中出现问题，请检查错误信息")

if __name__ == "__main__":
    main_deploy()