"""
主程序测试文件
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.project_name.main import run_business_logic


class TestMain:
    """测试主程序功能"""
    
    def test_run_business_logic(self, capsys):
        """测试业务逻辑函数"""
        # 运行业务逻辑
        run_business_logic()
        
        # 捕获输出
        captured = capsys.readouterr()
        
        # 验证输出包含预期内容
        assert "开始运行业务逻辑" in captured.out
        assert "处理数据: [1, 2, 3, 4, 5]" in captured.out
        assert "结果: 15" in captured.out
        assert "业务逻辑执行完成" in captured.out
    
    def test_business_logic_calculation(self):
        """测试业务逻辑的计算结果"""
        # 这里可以添加更多的单元测试
        # 例如测试具体的计算逻辑
        data = [1, 2, 3, 4, 5]
        result = sum(data)
        assert result == 15
        
        # 测试空列表
        empty_data = []
        empty_result = sum(empty_data)
        assert empty_result == 0
        
        # 测试负数
        negative_data = [-1, -2, -3]
        negative_result = sum(negative_data)
        assert negative_result == -6


if __name__ == "__main__":
    pytest.main([__file__])

