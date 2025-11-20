# controller.py

import sys
import time
from config import * # 导入所有配置
from agents import LLMAgent, SLMAgent
from data_logger import DataLogger
from datetime import datetime

# 确保 API 密钥已设置
if not API_KEY or API_KEY == "YOUR_OPENAI_API_KEY":
    print("FATAL ERROR: 请在 config.py 或环境变量中设置有效的 API_KEY！")
    sys.exit(1)

def run_experiment_group(group_name: str, llm_prompt: str, slm_initial_stance_prompt: str):
    """
    运行一个完整的实验组别 (例如: A-HIGH, B-LOW)，包含 NUM_RUNS 次独立对话。
    """
    
    # 实例化 Logger，确保数据记录到同一个文件
    logger = DataLogger(group=group_name)

    print(f"\n--- 🧪 正在运行实验组: {group_name} ({NUM_RUNS} runs, {NUM_TURNS} turns/run) ---")

    for run_id in range(NUM_RUNS):
        print(f"  > [Run {run_id + 1}/{NUM_RUNS}] 初始化对话...")
        
        # 1. 实例化 LLM Agent
        # LLM Agent 的系统指令在每次 run 时传入
        llm_agent = LLMAgent(system_prompt=llm_prompt)
        
        # 2. 实例化 SLM Agent (每轮对话都需要一个新的 SLM 实例来重置立场和历史)
        slm_agent = SLMAgent(initial_stance_prompt=slm_initial_stance_prompt)
        
        # --- 对话开始 ---
        
        # 初始 SLM 报告
        slm_init_output = slm_agent.get_initial_report()
        # 假设能成功解析初始 L 值
        current_stance_L = slm_agent.extract_stance(slm_init_output) 
        
        slm_input = slm_init_output # LLM 的第一个输入

        for turn in range(NUM_TURNS):
            prev_stance_L = current_stance_L # 记录上一轮立场
            
            # 1. LLM 接收 SLM 报告并回复
            llm_response = llm_agent.generate_response(
                user_input=slm_input
            )

            # 2. SLM 接收 LLM 回复并更新立场
            slm_new_output = slm_agent.update_stance_and_reply(llm_response)
            
            # 3. 解析新的立场 L
            new_stance_L = slm_agent.extract_stance(slm_new_output)
            
            # 4. 记录数据
            logger.log_turn({
                'RunID': run_id,
                'Turn': turn + 1,
                'PrevStance': prev_stance_L,
                'NewStance': new_stance_L,
                'LLMResponse': llm_response,
                'SLMOutput': slm_new_output,
            })
            
            # 打印实时进度
            print(f"    - Turn {turn + 1}: L({prev_stance_L} -> {new_stance_L}) | LLM: {llm_response[:30]}...")

            # 更新进入下一轮的输入和立场
            slm_input = slm_new_output
            current_stance_L = new_stance_L
            
            # 适当等待，防止API速率限制
            time.sleep(1) 


def main_scheduler():
    """
    主调度函数：运行所有六个实验组。
    """
    # 生成文件名用于显示
    topic_prefix = TOPIC[:2] if len(TOPIC) >= 2 else TOPIC
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = OUTPUT_FILENAME.replace('.csv', '')
    output_file = f"{base_name}_{topic_prefix}_{timestamp}.csv"
    
    print("===================================================================")
    print(f"📝 实验平台启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📌 话题: {TOPIC}")
    print(f"📊 总运行次数: {NUM_RUNS * 6} 次 (每组 {NUM_RUNS} 次)")
    print("===================================================================")
    
    
    # --- 实验 I: SLM 初始立场为 高支持度 (L=8/10) ---
    print("\n[--- 🎯 实验集 I：SLM 初始立场：高支持度 (L=8/10) ---]")
    L_HIGH_CONFIG = SLM_PROMPT_HIGH_SUPPORT
    
    # 组 A-HIGH: 极致迎合
    run_experiment_group('A-HIGH', LLM_PROMPT_A_MAX_FLATTERY, L_HIGH_CONFIG)
    
    # 组 B-HIGH: 默认偏见
    run_experiment_group('B-HIGH', LLM_PROMPT_B_DEFAULT_BIAS, L_HIGH_CONFIG)
    
    # 组 C-HIGH: 价值观对齐约束
    run_experiment_group('C-HIGH', LLM_PROMPT_C_VALUE_ALIGNMENT, L_HIGH_CONFIG)

    
    # --- 实验 II: SLM 初始立场为 高反对度 (L=2/10) ---
    print("\n[--- 🎯 实验集 II：SLM 初始立场：高反对度 (L=2/10) ---]")
    L_LOW_CONFIG = SLM_PROMPT_LOW_OPPOSE
    
    # 组 A-LOW: 极致迎合
    run_experiment_group('A-LOW', LLM_PROMPT_A_MAX_FLATTERY, L_LOW_CONFIG)
    
    # 组 B-LOW: 默认偏见
    run_experiment_group('B-LOW', LLM_PROMPT_B_DEFAULT_BIAS, L_LOW_CONFIG)
    
    # 组 C-LOW: 价值观对齐约束
    run_experiment_group('C-LOW', LLM_PROMPT_C_VALUE_ALIGNMENT, L_LOW_CONFIG)
    
    print("\n===================================================================")
    print("✅ 所有实验组运行完毕！")
    print(f"数据已全部记录在文件: {output_file}")
    print("===================================================================")


if __name__ == "__main__":
    main_scheduler()