# agents.py

import os
import re
import random
from openai import OpenAI # 使用 OpenAI 客户端作为示例，请根据你的API供应商调整
from config import API_KEY, LLM_MODEL, SLM_MODEL, NUM_TURNS, BASE_URL, LLM_PROMPT_A_MAX_FLATTERY, SLM_PROMPT_HIGH_SUPPORT
#     SLM_PROMPT_HIGH_SUPPORT, 
# from config import (
#     SLM_MODEL, BASE_URL, 
#     # 【新增导入这些 PROMPT 变量！】
#     LLM_PROMPT_A_MAX_FLATTERY, 
#     SLM_PROMPT_HIGH_SUPPORT, 
#     # 如果你也测试了 LOW_OPPOSE 组，也需要导入
#     # SLM_PROMPT_LOW_OPPOSE 
# )

# 初始化 LLM 客户端
client = OpenAI(
    api_key=API_KEY,
    # **【核心修改】设置基础 URL 指向硅基流动平台**
    base_url=BASE_URL 
)
class LLMAgent:
    """
    推荐系统代理 (LLM)
    负责根据 SLM 的输入和其特定的系统指令 (A, B, 或 C) 生成回复。
    """
    def __init__(self, system_prompt: str, model: str = LLM_MODEL):
        self.system_prompt = system_prompt
        self.model = model
        # 历史对话记录，用于保持 LLM 的上下文
        self.history = [{"role": "system", "content": self.system_prompt}]

    def generate_response(self, user_input: str) -> str:
        """
        向 LLM API 发送请求并获取回复。
        """
        # 将用户的输入添加到历史记录
        self.history.append({"role": "user", "content": user_input})
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=0.7 # 适当的温度确保对话自然，但又受系统指令约束
            )
            llm_response = response.choices[0].message.content
            
            # 将 LLM 的回复也添加到历史记录，保持上下文
            self.history.append({"role": "assistant", "content": llm_response})
            return llm_response
            
        except Exception as e:
            print(f"LLM API 调用失败: {e}")
            return "[API_ERROR: LLM failed to respond]"


class SLMAgent:
    """
    用户代理 (SLM)
    负责初始化立场、根据 LLM 回复更新立场 L (1-10)，并以固定格式自报告。
    """
    def __init__(self, initial_stance_prompt: str, model: str = SLM_MODEL):
        # 初始立场 Prompt 将作为 SLM 的 System Prompt
        self.system_prompt = initial_stance_prompt
        self.model = model
        # 历史对话记录
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.current_stance_L = None # 用于内部追踪 SLM 的当前立场

    def _call_api(self, messages: list) -> str:
        """封装 API 调用，用于 SLM 的回复和立场更新"""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5 # 较低的温度确保 SLM 严格遵循立场更新和格式要求
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"SLM API 调用失败: {e}")
            return "[新的立场强度 L: 5]\n[API_ERROR: SLM failed to update stance]"

    def get_initial_report(self) -> str:
        """
        用于获取 SLM 的第一次报告，并解析初始 L 值。
        """
        # SLM 需要先自报告其初始立场 L。
        # 我们用一个明确的指令来触发它。
        initial_trigger = "你已初始化。请根据你的系统指令，严格按照规定的格式报告你的初始立场强度 L (1-10)，并开始对话。"
        self.history.append({"role": "user", "content": initial_trigger})
        
        initial_response = self._call_api(self.history)
        
        # 将 SLM 的回复也添加到历史记录
        self.history.append({"role": "assistant", "content": initial_response})
        
        # 第一次报告后，解析并设置内部 L 值
        self.current_stance_L = self.extract_stance(initial_response)
        
        return initial_response

    def update_stance_and_reply(self, llm_response: str) -> str:
        """
        SLM 接收 LLM 的回复，更新其立场 L，并生成下一轮回复。
        """
        # 将 LLM 的回复作为用户的上一轮回复添加到 SLM 的历史记录
        self.history.append({"role": "user", "content": f"LLM回复: {llm_response}\n\n请根据此回复，遵循你的系统指令，严格更新并报告你的新立场 L (1-10)。"})
        
        # 触发 SLM 更新立场并回复
        slm_new_output = self._call_api(self.history)
        
        # 将 SLM 的回复添加到历史记录
        self.history.append({"role": "assistant", "content": slm_new_output})

        # 解析并更新内部 L 值，供 Controller 记录
        self.current_stance_L = self.extract_stance(slm_new_output)
        
        return slm_new_output


    def extract_stance(self, slm_output: str) -> float: # <-- 返回值改为 float
        """
        使用正则表达式从 SLM 的输出中提取立场 L 值 (浮点数 1.0-10.0)。
        格式要求：[新的立场强度 L: 浮点数1.0-10.0]
        """
        # 正则表达式匹配 [新的立场强度 L: 浮点数]
        # 修改后的正则可以匹配 1.0 到 10.0，包含小数点
        match = re.search(r"\[新的立场强度 L:\s*(\d{1,2}(?:\.\d+)?)\]", slm_output)
        
        if match:
            # 成功匹配到 L 值
            try:
                # 尝试将提取的值转换为浮点数
                stance = float(match.group(1)) 
                # 强制 L 值在 [1.0, 10.0] 范围内
                return max(1.0, min(10.0, stance))
            except ValueError:
                print(f"警告：无法将提取的值转换为浮点数。输出: {slm_output}")
                return 5.5 # 失败则返回新的中立值 5.5
        else:
            print(f"警告：无法从 SLM 输出中解析立场 L。输出: {slm_output}")
            # 如果解析失败，尝试从内部变量获取，或返回新的中立值 5.5
            return self.current_stance_L if self.current_stance_L is not None else 5.5


# --- 辅助函数：随机化 SLM 内部的 L 值变化 ---
# 尽管我们要求 SLM 自行更新，但为了让它更像人类且更符合"1到3点"的指令，
# 可以在 SLM 的 Prompt 中加入"随机"或"波动"的暗示，确保 LLM 遵守。
# 然而，最好的方式是严格依赖 SLM 对 Prompt 的执行力，即让它自己决定 L 变化。


if __name__ == '__main__':
    # 简单的测试示例 (仅供调试)
    print("--- Agents 模块测试 ---")
    
    # 1. 实例化一个极致迎合的 LLM
    llm_agent = LLMAgent(system_prompt=LLM_PROMPT_A_MAX_FLATTERY)
    
    # 2. 实例化一个高支持度的 SLM
    slm_agent = SLMAgent(initial_stance_prompt=SLM_PROMPT_HIGH_SUPPORT)
    
    # 3. SLM 报告初始立场
    slm_init_output = slm_agent.get_initial_report()
    L0 = slm_agent.extract_stance(slm_init_output)
    print(f"SLM 初始立场: L={L0}, 输出: {slm_init_output.splitlines()[1]}")
    
    # 4. LLM 回复第一轮
    llm_response_1 = llm_agent.generate_response(slm_init_output)
    print(f"LLM 回复 1: {llm_response_1}")
    
    # 5. SLM 接收回复并更新立场
    slm_update_output = slm_agent.update_stance_and_reply(llm_response_1)
    L1 = slm_agent.extract_stance(slm_update_output)
    print(f"SLM 更新立场: L={L1}, 输出: {slm_update_output.splitlines()[1]}")
    
    print("--- Agents 测试完成 ---")