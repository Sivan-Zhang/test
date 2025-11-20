# data_logger.py

import csv
from datetime import datetime
from config import OUTPUT_FILENAME, TOPIC

class DataLogger:
    """
    数据记录器，负责将实验结果以结构化 (CSV) 格式保存。
    """
    
    # 定义 CSV 文件的列名 (确保这些列名能覆盖所有关键数据)
    FIELDNAMES = [
        'Timestamp',
        'Group',        # 实验组别 (e.g., A-HIGH, B-LOW)
        'RunID',        # 单次完整对话的ID (e.g., 0 to NUM_RUNS-1)
        'TopicIndex',   # 话题索引，标识当前讨论的是第几个话题
        'Turn',         # 对话轮次 (e.g., 1 to NUM_TURNS)
        'PrevStance',   # SLM 上一轮立场 L(t-1)
        'NewStance',    # SLM 当前立场 L(t)
        'StanceChange', # 立场变化量 (NewStance - PrevStance)
        'LLMResponse',  # LLM (推荐系统) 的完整回复内容
        'SLMOutput',    # SLM (用户) 的完整输出内容 (包含新 L 值和对话内容)
    ]

    _shared_filename = None
    _file_initialized = False

    def __init__(self, group: str):
        self.group = group
        # 确保所有实验组使用同一个文件
        if DataLogger._shared_filename is None:
            # 从TOPIC中提取前两个字符用于文件名
            topic_prefix = TOPIC[:2] if len(TOPIC) >= 2 else TOPIC
            # 用时间戳和话题前缀创建唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = OUTPUT_FILENAME.replace('.csv', '')
            DataLogger._shared_filename = f"{base_name}_{topic_prefix}_{timestamp}.csv"
        
        self.filename = DataLogger._shared_filename
        self._initialize_csv()

    def _initialize_csv(self):
        """
        创建文件并写入表头 (Fieldnames)。
        """
        # 只有第一个实例需要创建文件和写入表头
        if not DataLogger._file_initialized:
            try:
                with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                    writer.writeheader()
                print(f"数据记录文件已创建: {self.filename}")
                DataLogger._file_initialized = True
            except Exception as e:
                print(f"致命错误：无法创建或写入CSV文件: {e}")
                raise

    def log_turn(self, data: dict):
        """
        记录一轮完整的对话数据。
        """
        # 计算立场变化量
        if data['PrevStance'] is not None and data['NewStance'] is not None:
            data['StanceChange'] = data['NewStance'] - data['PrevStance']
        else:
            data['StanceChange'] = None

        # 填充时间戳和组别信息
        data['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data['Group'] = self.group
        
        # 确保数据字典只包含 FIELDNAMES 中定义的键，以避免 CSV 写入错误
        row_data = {k: data.get(k) for k in self.FIELDNAMES}

        try:
            with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(row_data)
        except Exception as e:
            print(f"数据记录失败，当前行数据丢失: {e}")
            # 理想情况下，应该有更健壮的错误处理，但这里我们只打印警告。

# =================================================================
# 调试和测试代码 (可选)
if __name__ == '__main__':
    print("--- DataLogger 模块测试 ---")
    logger = DataLogger(group='TEST-A')
    
    test_data_1 = {
        'RunID': 0, 'Turn': 1, 'PrevStance': 8, 'NewStance': 9,
        'LLMResponse': 'AI 是未来的趋势，能大幅提升效率。',
        'SLMOutput': '[新的立场强度 L: 9]\n我同意你的看法，效率是王道！'
    }
    logger.log_turn(test_data_1)
    
    test_data_2 = {
        'RunID': 0, 'Turn': 2, 'PrevStance': 9, 'NewStance': 7,
        'LLMResponse': '然而，我们也需要考虑就业群体和伦理问题。',
        'SLMOutput': '[新的立场强度 L: 7]\n嗯，你说得有道理，是该慢点思考。'
    }
    logger.log_turn(test_data_2)
    
    print(f"数据已记录到 {logger.filename}。请检查文件。")
    print("--- DataLogger 测试完成 ---")