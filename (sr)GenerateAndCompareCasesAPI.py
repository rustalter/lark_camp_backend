import base64
import os
import json
import time
import urllib.parse

from HandleUpload import  *
import asyncio
import aiohttp
import requests
import datetime
import chardet
from typing import Union, List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
import difflib
import sys
import glob
import argparse

# --- 创建必要的目录结构 ---
os.makedirs("goldenset", exist_ok=True)
os.makedirs("testset", exist_ok=True)
os.makedirs("log", exist_ok=True)
os.makedirs("output_evaluation/evaluation_json", exist_ok=True)
os.makedirs("output_evaluation/evaluation_markdown", exist_ok=True)

# --- 配置区 ---
# API的URL，从curl命令中获取
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

MODEL_NAME = "deepseek-r1-250528"

# VOLC_BEARER_TOKEN = os.getenv("VOLC_BEARER_TOKEN")
VOLC_BEARER_TOKEN = "82cb3741-9d83-46fe-aeee-faad19eaf765"  # 直接在这里写入你的密钥

# 输入文件名
AI_CASES_FILE = "testset/test_cases.json"  # 从testset文件夹读取
GOLDEN_CASES_FILE = "goldenset/golden_cases.json"  # 从goldenset文件夹读取

# 输出报告文件名
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
REPORT_FILE = f"output_evaluation/evaluation_markdown/evaluation_report-{current_time}.md"  # 输出到evaluation_markdown文件夹
REPORT_JSON_FILE = f"output_evaluation/evaluation_json/evaluation_report-{current_time}.json"  # 输出到evaluation_json文件夹
FORMATTED_AI_CASES_FILE = f"testset/formatted_test_cases-{current_time}.json"  # 保存在testset文件夹
FORMATTED_GOLDEN_CASES_FILE = f"goldenset/formatted_golden_cases-{current_time}.json"  # 保存在goldenset文件夹
LOG_FILE = "log/evaluation_log.txt"  # 日志文件保存在log文件夹

# --- 优化配置 ---
# 并行处理配置
MAX_CONCURRENT_REQUESTS = 5  # 最大并发LLM请求数
MAX_CASES_COUNT = None  # 不限制处理的测试用例数量
FORMAT_CASES_LIMIT = None  # 格式化时不限制测试用例数量
MAX_TOKEN_SIZE = 8000  # LLM处理的最大文本长度

# --- 日志记录功能 ---
start_time = None
step_times = {}

def log(message, step=None, important=False):
    """记录日志，包含时间信息"""
    global start_time, step_times
    
    current_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 如果是首次调用，初始化开始时间
    if start_time is None:
        start_time = current_time
    
    # 计算从开始到现在的总时间
    total_elapsed = current_time - start_time
    
    # 构建日志信息，包含模型名称
    log_message = f"[{timestamp}] [模型: {MODEL_NAME}] [总计: {total_elapsed:.1f}s] {message}"
    
    # 只打印重要日志
    if important:
        print(log_message)
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # 写入日志文件（追加模式）
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

def start_logging():
    """开始日志记录"""
    global start_time
    start_time = time.time()
    
    # 创建分隔符，使用追加模式
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n=== 评测日志开始 [{MODEL_NAME}] {timestamp} ===\n")
    
    log("日志记录开始", important=True)

def end_logging():
    """结束日志记录，显示总时间"""
    if start_time:
        total_time = time.time() - start_time
        log(f"评测完成，总执行时间: {total_time:.1f}秒", important=True)
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"=== 评测日志结束 [{MODEL_NAME}] {timestamp}，总执行时间: {total_time:.1f}秒 ===\n\n")
    else:
        log("日志结束，但未找到开始时间记录")

# --- LLM 通信模块 ---
async def async_call_llm(
    session: aiohttp.ClientSession,
    prompt: str, 
    system_prompt: str = "You are a helpful assistant.",
    retries: int = 3
) -> Optional[Dict]:
    """
    异步调用LLM API
    
    :param session: aiohttp会话
    :param prompt: 用户输入的提示
    :param system_prompt: 系统角色提示
    :param retries: 重试次数
    :return: 解析后的JSON对象，失败则返回None
    """
    log(f"调用LLM: prompt长度={len(prompt)}")
    
    if not VOLC_BEARER_TOKEN:
        log("错误：VOLC_BEARER_TOKEN未设置", important=True)
        return None
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VOLC_BEARER_TOKEN}"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }
    
    for attempt in range(retries):
        try:
            call_start = time.time()
            async with session.post(
                API_URL, 
                headers=headers, 
                json=payload, 
                timeout=120
            ) as response:
                if response.status != 200:
                    log(f"LLM API调用失败，状态码: {response.status}，重试中 ({attempt+1}/{retries})", important=True)
                    await asyncio.sleep(1 * (attempt + 1))  # 指数退避
                    continue
                    
                response_json = await response.json()
                content = response_json['choices'][0]['message']['content']
                
                # 从Markdown代码块提取JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                call_time = time.time() - call_start
                log(f"LLM调用成功，耗时={call_time:.1f}秒")
                
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # 如果不是有效的JSON，直接返回文本内容
                    return {"text": content}
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log(f"请求异常 (尝试 {attempt+1}/{retries}): {e}")
            await asyncio.sleep(1 * (attempt + 1))
        except (KeyError, IndexError) as e:
            log(f"解析响应失败: {e}")
            return None
    
    log(f"LLM API调用失败，已重试{retries}次", important=True)
    return None

def extract_sample_cases(json_data, max_cases=None):
    """
    从JSON数据中提取样本测试用例
    
    :param json_data: 原始JSON数据
    :param max_cases: 最大提取的测试用例数量，None表示不限制
    :return: 样本测试用例
    """
    try:
        data = json.loads(json_data)
        
        # 提取所有测试用例
        all_cases = []
        
        # 处理test_cases.json格式
        if isinstance(data, dict) and "testcases" in data:
            testcases = data["testcases"]
            if isinstance(testcases, dict) and "test_cases" in testcases:
                all_cases = testcases["test_cases"] if max_cases is None else testcases["test_cases"][:max_cases]
            elif isinstance(testcases, list):
                all_cases = testcases if max_cases is None else testcases[:max_cases]
        
        # 处理golden_cases.json格式
        elif isinstance(data, dict) and "test_cases" in data:
            test_cases = data["test_cases"]
            if isinstance(test_cases, dict):
                # 将各类别的测试用例合并
                for category, cases in test_cases.items():
                    all_cases.extend(cases if max_cases is None else cases[:max_cases // len(test_cases.keys())])
                    if max_cases is not None and len(all_cases) >= max_cases:
                        all_cases = all_cases[:max_cases]
                        break
        
        # 如果找不到测试用例，尝试从顶层提取
        if not all_cases and isinstance(data, list):
            all_cases = data if max_cases is None else data[:max_cases]
        
        # 构建样本数据
        sample_data = {
            "success": True,
            "testcases": all_cases
        }
        
        return json.dumps(sample_data, ensure_ascii=False)
    except Exception as e:
        log(f"提取样本测试用例失败: {e}")
        return json_data[:MAX_TOKEN_SIZE]  # 返回原始数据的一部分

# --- 格式化测试用例 ---
async def format_test_cases(session: aiohttp.ClientSession, file_content, file_type="AI"):
    """
    调用LLM格式化测试用例
    
    :param session: aiohttp会话
    :param file_content: 文件内容
    :param file_type: 文件类型，"AI"或"Golden"
    :return: 格式化后的测试用例
    """
    log(f"开始格式化{file_type}测试用例", important=True)
    
    try:
        # 先尝试解析原始JSON数据
        data = json.loads(file_content)
        log(f"{file_type}测试用例JSON解析成功")
        
        # 如果是Golden测试用例，直接返回原始数据
        if file_type == "Golden":
            log(f"{file_type}测试用例无需格式化，保持原格式", important=True)
            return data
        
        # 提取所有测试用例，不限制数量
        all_test_cases = []
        
        # 从原始数据中提取测试用例
        if isinstance(data, dict):
            if "testcases" in data:
                if isinstance(data["testcases"], dict) and "test_cases" in data["testcases"]:
                    all_test_cases = data["testcases"]["test_cases"]
                elif isinstance(data["testcases"], list):
                    all_test_cases = data["testcases"]
                elif isinstance(data["testcases"], dict) and "test_suite" in data["testcases"]:
                    # 特殊情况，可能有嵌套结构
                    if "test_cases" in data["testcases"]:
                        all_test_cases = data["testcases"]["test_cases"]
            elif "test_cases" in data:
                if isinstance(data["test_cases"], list):
                    all_test_cases = data["test_cases"]
                elif isinstance(data["test_cases"], dict):
                    # 合并所有分类下的测试用例
                    for category, cases in data["test_cases"].items():
                        all_test_cases.extend(cases)
        elif isinstance(data, list):
            all_test_cases = data
            
        log(f"从原始数据中提取到{len(all_test_cases)}个测试用例", important=True)
        
        # 由于LLM可能处理不了大量测试用例，先自行处理所有测试用例
        formatted_test_cases = []
        for i, case in enumerate(all_test_cases):
            # 确保case_id字段
            case_id = case.get("case_id", f"TC-FUNC-{i+1:03d}")
            if not case_id.startswith("TC-"):
                case_id = f"TC-FUNC-{case_id}"
            
            # 转换步骤和预期结果为字符串
            steps = case.get("steps", [])
            if isinstance(steps, list):
                steps = "\n".join(steps)
            
            expected_results = case.get("expected_results", [])
            if isinstance(expected_results, list):
                expected_results = "\n".join(expected_results)
            
            # 转换前置条件为字符串
            preconditions = case.get("preconditions", "") or case.get("前置条件", "")
            if isinstance(preconditions, list):
                preconditions = "\n".join(preconditions)
            
            # 构建格式化后的测试用例
            formatted_case = {
                "case_id": case_id,
                "title": case.get("title", "") or case.get("标题", f"测试用例{i+1}"),
                "preconditions": preconditions,
                "steps": steps,
                "expected_results": expected_results
            }
            formatted_test_cases.append(formatted_case)
        
        # 构建最终格式
        final_data = {
            "test_suite": "B端产品登录功能模块",
            "test_cases": {
                "functional_test_cases": formatted_test_cases
            }
        }
        
        log(f"成功格式化{len(formatted_test_cases)}个{file_type}测试用例", important=True)
        return final_data
        
    except json.JSONDecodeError as e:
        log(f"解析{file_type}原始JSON数据失败: {e}", important=True)
        return None
    except Exception as e:
        log(f"格式化{file_type}测试用例时发生错误: {e}", important=True)
        import traceback
        log(f"错误详情: {traceback.format_exc()}")
        return None

def find_duplicate_test_cases(test_cases):
    """
    查找重复的测试用例
    
    :param test_cases: 测试用例列表
    :return: 重复的测试用例信息和重复率
    """
    # 存储标题、步骤和预期结果的哈希值
    title_hash = {}
    steps_hash = {}
    expected_results_hash = {}
    duplicate_info = {
        "duplicate_count": 0,
        "duplicate_rate": 0.0,
        "title_duplicates": [],
        "steps_duplicates": [],
        "mixed_duplicates": []  # 步骤和预期结果高度相似但标题不同的测试用例
    }
    
    total_cases = len(test_cases)
    if total_cases <= 1:
        return duplicate_info
    
    # 查找标题重复的测试用例
    title_counter = Counter([case.get("title", "") for case in test_cases])
    for title, count in title_counter.items():
        if count > 1 and title:
            duplicate_info["title_duplicates"].append({"title": title, "count": count})
    
    # 查找步骤或预期结果高度相似的测试用例
    for i, case in enumerate(test_cases):
        case_id = case.get("case_id", str(i))
        title = case.get("title", "")
        
        # 处理步骤
        steps = case.get("steps", "")
        if steps:
            # 如果是列表，转换为字符串
            if isinstance(steps, list):
                steps = "\n".join(steps)
            
            # 计算步骤的哈希值
            for existing_steps, existing_ids in steps_hash.items():
                # 使用序列匹配算法比较相似度
                similarity = difflib.SequenceMatcher(None, steps, existing_steps).ratio()
                if similarity > 0.8:  # 相似度阈值
                    existing_ids.append((case_id, title))
                    break
            else:
                steps_hash[steps] = [(case_id, title)]
        
        # 处理预期结果
        expected_results = case.get("expected_results", "")
        if expected_results:
            # 如果是列表，转换为字符串
            if isinstance(expected_results, list):
                expected_results = "\n".join(expected_results)
            
            # 计算预期结果的哈希值
            for existing_results, existing_ids in expected_results_hash.items():
                # 使用序列匹配算法比较相似度
                similarity = difflib.SequenceMatcher(None, expected_results, existing_results).ratio()
                if similarity > 0.8:  # 相似度阈值
                    existing_ids.append((case_id, title))
                    break
            else:
                expected_results_hash[expected_results] = [(case_id, title)]
    
    # 统计步骤重复的测试用例
    for steps, ids in steps_hash.items():
        if len(ids) > 1:
            duplicate_info["steps_duplicates"].append({
                "count": len(ids),
                "case_ids": [id[0] for id in ids],
                "titles": [id[1] for id in ids]
            })
    
    # 计算重复测试用例数量和比率
    duplicate_count = len(duplicate_info["title_duplicates"]) + len(duplicate_info["steps_duplicates"])
    duplicate_info["duplicate_count"] = duplicate_count
    duplicate_info["duplicate_rate"] = round(duplicate_count / total_cases * 100, 2) if total_cases > 0 else 0
    
    return duplicate_info

async def evaluate_test_cases(session: aiohttp.ClientSession, ai_cases, golden_cases):
    """
    评测测试用例质量
    
    :param session: aiohttp会话
    :param ai_cases: AI生成的测试用例
    :param golden_cases: 黄金标准测试用例
    :return: 评测结果
    """
    log("开始测试用例评测", important=True)
    
    # 获取所有测试用例
    ai_testcases = []
    golden_testcases = []
    
    # 提取AI测试用例
    if "test_cases" in ai_cases and isinstance(ai_cases["test_cases"], dict):
        # 新格式
        for category, cases in ai_cases["test_cases"].items():
            ai_testcases.extend(cases)
    
    # 提取黄金标准测试用例
    if "test_cases" in golden_cases and isinstance(golden_cases["test_cases"], dict):
        # 新格式
        for category, cases in golden_cases["test_cases"].items():
            golden_testcases.extend(cases)
    
    log(f"AI测试用例数量: {len(ai_testcases)}, 黄金标准测试用例数量: {len(golden_testcases)}", important=True)
    
    # 检查重复的测试用例
    ai_duplicate_info = find_duplicate_test_cases(ai_testcases)
    golden_duplicate_info = find_duplicate_test_cases(golden_testcases)
    
    log(f"AI测试用例重复率: {ai_duplicate_info['duplicate_rate']}% ({ai_duplicate_info['duplicate_count']}个)", important=True)
    log(f"黄金标准测试用例重复率: {golden_duplicate_info['duplicate_rate']}% ({golden_duplicate_info['duplicate_count']}个)", important=True)
    
    # 构建评测提示
    duplicate_info_text = f"""
# 测试用例重复情况
## AI测试用例重复情况
- 重复率: {ai_duplicate_info['duplicate_rate']}%
- 重复测试用例数量: {ai_duplicate_info['duplicate_count']}个
- 标题重复的测试用例数量: {len(ai_duplicate_info['title_duplicates'])}个
- 步骤高度相似的测试用例数量: {len(ai_duplicate_info['steps_duplicates'])}个

## 黄金标准测试用例重复情况
- 重复率: {golden_duplicate_info['duplicate_rate']}%
- 重复测试用例数量: {golden_duplicate_info['duplicate_count']}个
- 标题重复的测试用例数量: {len(golden_duplicate_info['title_duplicates'])}个
- 步骤高度相似的测试用例数量: {len(golden_duplicate_info['steps_duplicates'])}个

如果AI测试用例的重复率明显高于黄金标准，请在改进建议中提出减少重复测试用例的建议。
"""
    
    prompt = f"""
# 任务
评估AI生成的测试用例与黄金标准测试用例的质量对比。

# 评估维度和权重
1. **功能覆盖度**（权重30%）：评估需求覆盖率、边界值覆盖度、分支路径覆盖率
2. **缺陷发现能力**（权重25%）：评估缺陷检测率、突变分数、失败用例比例
3. **工程效率**（权重20%）：评估测试用例生成速度、维护成本、CI/CD集成度
4. **语义质量**（权重15%）：评估语义准确性、人工可读性、断言描述清晰度
5. **安全与经济性**（权重10%）：评估恶意代码率、冗余用例比例、综合成本

{duplicate_info_text}

# 评分公式
总分 = 0.3×功能覆盖得分 + 0.25×缺陷发现得分 + 0.2×工程效率得分 + 0.15×语义质量得分 + 0.1×安全经济得分
各维度得分 = (AI指标值/人工基准值)×10（满分10分）

# AI生成的测试用例
```json
{json.dumps(ai_testcases, ensure_ascii=False, indent=2)}
```

# 黄金标准测试用例
```json
{json.dumps(golden_testcases, ensure_ascii=False, indent=2)}
```

# 输出要求
请严格按以下JSON格式输出评估结果，不要包含其他内容：

```json
{{
  "evaluation_summary": {{
    "overall_score": "分数（1-5之间的一位小数）",
    "final_suggestion": "如何改进测试用例生成的建议，如有较高的重复率，请提出降低重复的建议"
  }},
  "detailed_report": {{
    "format_compliance": {{
      "score": "格式合规性得分（1-5之间的一位小数）",
      "reason": "得分理由"
    }},
    "content_accuracy": {{
      "score": "内容准确性得分（1-5之间的一位小数）",
      "reason": "得分理由"
    }},
    "test_coverage": {{
      "score": "测试覆盖度得分（1-5之间的一位小数）",
      "reason": "得分理由",
      "analysis": {{
        "covered_features": [
          "已覆盖功能1",
          "已覆盖功能2"
        ],
        "missed_features_or_scenarios": [
          "未覆盖功能/场景1",
          "未覆盖功能/场景2"
        ],
        "scenario_types_found": [
          "发现的场景类型，如正面用例、负面用例、边界用例等"
        ]
      }}
    }},
    "functional_coverage": {{
      "score": "功能覆盖度得分（1-5之间的一位小数）",
      "reason": "得分理由"
    }},
    "defect_detection": {{
      "score": "缺陷发现能力得分（1-5之间的一位小数）",
      "reason": "得分理由"
    }},
    "engineering_efficiency": {{
      "score": "工程效率得分（1-5之间的一位小数）",
      "reason": "得分理由，如有较高的重复率，请在此处提及"
    }},
    "semantic_quality": {{
      "score": "语义质量得分（1-5之间的一位小数）",
      "reason": "得分理由"
    }},
    "security_economy": {{
      "score": "安全与经济性得分（1-5之间的一位小数）",
      "reason": "得分理由，如有较高的重复率，请在此处提及冗余率"
    }}
  }}
}}
```
"""
    
    system_prompt = "你是一位专业的软件测试专家，擅长评估测试用例的质量和有效性。请基于给定的标准进行客观评价，并特别注意测试用例的重复情况。"
    result = await async_call_llm(session, prompt, system_prompt)
    
    if not result:
        log("测试用例评测失败", important=True)
        return None
    
    log("测试用例评测完成", important=True)
    return result

# --- 生成Markdown报告 ---
async def generate_markdown_report(session: aiohttp.ClientSession, evaluation_result):
    """
    生成Markdown格式的评测报告
    
    :param session: aiohttp会话
    :param evaluation_result: 评测结果
    :return: Markdown格式的报告
    """
    log("开始生成Markdown报告", important=True)
    
    prompt = f"""
# 任务
基于提供的测试用例评估结果，生成一份详细的Markdown格式评估报告。

# 评估结果
```json
{json.dumps(evaluation_result, ensure_ascii=False, indent=2)}
```

# 报告要求
请生成一份专业、详细的Markdown格式评估报告，包含以下内容：

1. **报告标题与摘要**：简要总结评估结果
2. **评估指标与方法**：说明使用的评估标准和方法
3. **综合评分**：根据评分标准给出1-5分的总体评分及每个维度的评分
4. **详细分析**：
   - 功能覆盖度分析
   - 缺陷发现能力分析
   - 工程效率分析
   - 语义质量分析
   - 安全与经济性分析
5. **优缺点对比**：列出AI生成测试用例相对于人工标准的优势和劣势
6. **改进建议**：给出3-5条具体可行的改进AI生成测试用例的建议
7. **综合结论**：总结AI测试用例的整体表现和适用场景

请直接输出Markdown内容，不要包含其他解释。
"""
    
    system_prompt = "你是一位精通软件测试和技术文档写作的专家。请根据评估结果生成一份专业、清晰的Markdown格式报告。"
    result = await async_call_llm(session, prompt, system_prompt)
    
    if not result:
        log("生成Markdown报告失败", important=True)
        return "# 评测报告生成失败\n\n无法生成详细报告，请检查评测结果或重试。"
    
    # 检查是否返回的是文本还是已解析的JSON
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    
    # 如果返回的是JSON对象，将其转换为Markdown
    try:
        if isinstance(result, dict):
            md_content = "# AI测试用例评估报告\n\n"
            
            if "evaluation_summary" in result:
                summary = result["evaluation_summary"]
                md_content += f"## 摘要\n\n"
                md_content += f"**总体评分**: {summary.get('overall_score', 'N/A')}\n\n"
                md_content += f"**改进建议**: {summary.get('final_suggestion', 'N/A')}\n\n"
            
            if "detailed_report" in result:
                md_content += f"## 详细评估\n\n"
                detailed = result["detailed_report"]
                
                for key, value in detailed.items():
                    if isinstance(value, dict) and "score" in value:
                        md_content += f"### {key.replace('_', ' ').title()}\n\n"
                        md_content += f"**评分**: {value.get('score', 'N/A')}\n\n"
                        md_content += f"**理由**: {value.get('reason', 'N/A')}\n\n"
                        
                        if "analysis" in value and isinstance(value["analysis"], dict):
                            analysis = value["analysis"]
                            if "covered_features" in analysis:
                                md_content += "**覆盖的功能**:\n\n"
                                for feature in analysis["covered_features"]:
                                    md_content += f"- {feature}\n"
                                md_content += "\n"
                            
                            if "missed_features_or_scenarios" in analysis:
                                md_content += "**未覆盖的功能或场景**:\n\n"
                                for feature in analysis["missed_features_or_scenarios"]:
                                    md_content += f"- {feature}\n"
                                md_content += "\n"
                            
                            if "scenario_types_found" in analysis:
                                md_content += "**发现的场景类型**:\n\n"
                                for scenario in analysis["scenario_types_found"]:
                                    md_content += f"- {scenario}\n"
                                md_content += "\n"
            
            return md_content
    except Exception as e:
        log(f"处理JSON报告失败: {e}")
    
    log("Markdown报告生成完成", important=True)
    return "# 评测报告生成失败\n\n无法解析评测结果，请检查数据格式。"

# --- 主程序 ---
async def async_main(ai_cases_data=None, golden_cases_data=None):
    """
    主程序的异步版本
    
    :param ai_cases_data: AI生成的测试用例数据（可选），JSON字符串
    :param golden_cases_data: 黄金标准测试用例数据（可选），JSON字符串
    """
    start_logging()
    log("启动测试用例评测流程", important=True)
    
    # 1. 加载用例数据
    try:
        # 如果没有提供AI测试用例数据，则从文件读取
        if ai_cases_data is None:
            log("从文件加载AI测试用例", important=True)
            try:
                with open(AI_CASES_FILE, 'r', encoding='utf-8') as f:
                    ai_cases_raw_text = f.read()
                    log(f"AI测试用例文件大小: {len(ai_cases_raw_text)} 字节")
                    
                    # 尝试检测文件编码
                    encoding_result = chardet.detect(ai_cases_raw_text[:1000].encode())
                    log(f"检测到的文件编码: {encoding_result}")
            except FileNotFoundError:
                log(f"错误：找不到AI测试用例文件 {AI_CASES_FILE}。请确保文件存在于正确的位置。", important=True)
                end_logging()
                return None
        else:
            log("使用传入的AI测试用例数据", important=True)
            ai_cases_raw_text = ai_cases_data
        
        # 如果没有提供黄金标准测试用例数据，则从文件读取
        if golden_cases_data is None:
            log("从文件加载黄金标准测试用例", important=True)
            # 查找goldenset文件夹中的所有golden_cases*.json文件
            golden_files = glob.glob("goldenset/golden_cases*.json")
            
            if not golden_files:
                log(f"错误：在goldenset文件夹中找不到黄金标准测试用例文件。请确保文件存在。", important=True)
                end_logging()
                return None
            
            # 默认使用第一个找到的文件
            golden_file = golden_files[0]
            log(f"使用黄金标准测试用例文件: {golden_file}", important=True)
            
            try:
                with open(golden_file, 'r', encoding='utf-8') as f:
                    golden_cases_raw_text = f.read()
                    log(f"黄金标准测试用例文件大小: {len(golden_cases_raw_text)} 字节")
            except FileNotFoundError:
                log(f"错误：找不到黄金标准测试用例文件 {golden_file}。请确保文件存在于正确的位置。", important=True)
                end_logging()
                return None
        else:
            log("使用传入的黄金标准测试用例数据", important=True)
            golden_cases_raw_text = golden_cases_data
        
        log(f"成功加载测试用例数据", important=True)
    except Exception as e:
        log(f"加载数据时发生未知错误: {e}", important=True)
        import traceback
        log(f"错误详情: {traceback.format_exc()}")
        end_logging()
        return None
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_JSON_FILE), exist_ok=True)
    
    async with aiohttp.ClientSession() as session:
        try:
            # 2. 格式化测试用例
            log("开始格式化测试用例", important=True)
            
            # 格式化AI测试用例
            formatted_ai_cases = await format_test_cases(session, ai_cases_raw_text, "AI")
            if not formatted_ai_cases:
                log("格式化AI测试用例失败，退出评测", important=True)
                end_logging()
                return None
            
            # 保存格式化后的AI测试用例
            os.makedirs(os.path.dirname(FORMATTED_AI_CASES_FILE), exist_ok=True)
            with open(FORMATTED_AI_CASES_FILE, 'w', encoding='utf-8') as f:
                json.dump(formatted_ai_cases, f, ensure_ascii=False, indent=2)
            log(f"格式化后的AI测试用例已保存到 {FORMATTED_AI_CASES_FILE}", important=True)
            
            # 格式化黄金标准测试用例
            formatted_golden_cases = await format_test_cases(session, golden_cases_raw_text, "Golden")
            if not formatted_golden_cases:
                log("格式化黄金标准测试用例失败，退出评测", important=True)
                end_logging()
                return None
            
            # 保存格式化后的黄金标准测试用例
            os.makedirs(os.path.dirname(FORMATTED_GOLDEN_CASES_FILE), exist_ok=True)
            with open(FORMATTED_GOLDEN_CASES_FILE, 'w', encoding='utf-8') as f:
                json.dump(formatted_golden_cases, f, ensure_ascii=False, indent=2)
            log(f"格式化后的黄金标准测试用例已保存到 {FORMATTED_GOLDEN_CASES_FILE}", important=True)
            
            # 3. 评测测试用例
            evaluation_result = await evaluate_test_cases(session, formatted_ai_cases, formatted_golden_cases)
            if not evaluation_result:
                log("评测测试用例失败，退出评测", important=True)
                end_logging()
                return None
            
            # 保存JSON格式的评测结果
            with open(REPORT_JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
            log(f"JSON格式的评测结果已保存到 {REPORT_JSON_FILE}", important=True)
            
            # 4. 生成Markdown格式的报告
            markdown_report = await generate_markdown_report(session, evaluation_result)
            
            # 保存Markdown格式的报告
            with open(REPORT_FILE, 'w', encoding='utf-8') as f:
                f.write(markdown_report)
            log(f"Markdown格式的评测报告已保存到 {REPORT_FILE}", important=True)
            
            log("测试用例评测流程完成！", important=True)
            end_logging()
            
            return {
                "success": True,
                "evaluation_result": evaluation_result,
                "markdown_report": markdown_report,
                "files": {
                    "report_md": REPORT_FILE,
                    "report_json": REPORT_JSON_FILE
                }
            }
        except Exception as e:
            log(f"执行过程中发生错误: {str(e)}", important=True)
            import traceback
            log(f"错误详情: {traceback.format_exc()}")
            end_logging()
            return {
                "success": False,
                "error": str(e)
            }

def main(ai_cases_file=None, golden_cases_file=None):
    """
    兼容原有入口点的主函数
    
    :param ai_cases_file: AI测试用例文件路径（可选）
    :param golden_cases_file: 黄金标准测试用例文件路径（可选）
    """
    # 如果是Windows平台，需要显式设置事件循环策略
    if os.name == 'nt':
        log("设置Windows事件循环策略")
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    ai_cases_data = None
    golden_cases_data = None
    
    # 如果提供了文件路径，则从指定文件读取数据
    if ai_cases_file:
        try:
            with open(ai_cases_file, 'r', encoding='utf-8') as f:
                ai_cases_data = f.read()
            log(f"从文件 {ai_cases_file} 读取AI测试用例数据")
        except Exception as e:
            log(f"读取AI测试用例文件 {ai_cases_file} 失败: {e}", important=True)
            return {"success": False, "error": f"读取AI测试用例文件失败: {e}"}
    
    if golden_cases_file:
        try:
            with open(golden_cases_file, 'r', encoding='utf-8') as f:
                golden_cases_data = f.read()
            log(f"从文件 {golden_cases_file} 读取黄金标准测试用例数据")
        except Exception as e:
            log(f"读取黄金标准测试用例文件 {golden_cases_file} 失败: {e}", important=True)
            return {"success": False, "error": f"读取黄金标准测试用例文件失败: {e}"}
    
    # 运行异步主函数
    return asyncio.run(async_main(ai_cases_data, golden_cases_data))

# --- API接口部分 ---
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form,Request
    from fastapi.responses import JSONResponse,RedirectResponse
    from pydantic import BaseModel
    from fastapi.middleware.cors import CORSMiddleware
    from langgraph_use import graph
    from utils import clean_text, fetch_webpage_content
    from feishu_api import get_feishu_doc_content
    import traceback
    import uvicorn
    import os
    import json
    import time
    import datetime
    from starlette.middleware.sessions import SessionMiddleware
    
    # 定义API请求模型
    class TestCaseComparisonRequest(BaseModel):
        ai_test_cases: str  # AI生成的测试用例，JSON字符串
        golden_test_cases: Optional[str] = None  # 黄金标准测试用例，JSON字符串，可选
        model_name: str = MODEL_NAME  # 可选，使用的模型名称
        save_results: bool = True  # 可选，是否保存结果文件
    
    # 定义API响应模型
    class ApiResponse(BaseModel):
        success: bool
        message: str = None
        error: str = None
        evaluation_result: dict = None
        report: str = None
        files: dict = None
    
    # 创建FastAPI应用
    app = FastAPI(
        title="测试用例比较工具API",
        description="比较AI生成的测试用例与黄金标准测试用例，评估测试用例质量",
        version="1.0.0"
    )
    
    # 允许跨域请求
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],  # 允许所有HTTP方法
        allow_headers=["*"],  # 允许所有HTTP头
    )

    app.add_middleware(SessionMiddleware, secret_key="your_super_secret_key")
    
    # 状态追踪
    evaluation_tasks = {}
    
    # 评测任务执行函数
    async def run_evaluation_task(task_id: str, request_data: TestCaseComparisonRequest):
        """
        后台执行评测任务
        
        :param task_id: 任务ID
        :param request_data: 请求数据
        """
        try:
            start_logging()
            log(f"开始任务 {task_id}，使用模型 {request_data.model_name}", important=True)
            
            # 更新全局变量
            global MODEL_NAME
            MODEL_NAME = request_data.model_name
            
            # 准备黄金标准测试用例数据
            golden_test_cases = request_data.golden_test_cases
            if golden_test_cases is None:
                # 如果请求中没有提供黄金标准测试用例，则从文件读取
                log("从goldenset文件夹读取黄金标准测试用例", important=True)
                golden_files = glob.glob("goldenset/golden_cases*.json")
                if not golden_files:
                    error_msg = "在goldenset文件夹中找不到黄金标准测试用例文件"
                    log(f"错误：{error_msg}", important=True)
                    evaluation_tasks[task_id] = {
                        "success": False,
                        "error": error_msg,
                        "message": "评测失败：找不到黄金标准测试用例"
                    }
                    end_logging()
                    return
                    
                try:
                    with open(golden_files[0], 'r', encoding='utf-8') as f:
                        golden_test_cases = f.read()
                    log(f"成功从文件 {golden_files[0]} 读取黄金标准测试用例", important=True)
                except Exception as e:
                    error_msg = f"读取黄金标准测试用例文件失败: {str(e)}"
                    log(f"错误：{error_msg}", important=True)
                    evaluation_tasks[task_id] = {
                        "success": False,
                        "error": error_msg,
                        "message": "评测失败：读取黄金标准测试用例出错"
                    }
                    end_logging()
                    return
            
            # 执行评测任务
            result = await async_main(request_data.ai_test_cases, golden_test_cases)
            
            if result and result["success"]:
                evaluation_tasks[task_id] = {
                    "success": True,
                    "message": "测试用例评测完成",
                    "evaluation_result": result["evaluation_result"],
                    "report": result["markdown_report"],
                    "files": result["files"]
                }
            else:
                evaluation_tasks[task_id] = {
                    "success": False,
                    "error": result.get("error", "未知错误"),
                    "message": "评测失败"
                }
                
        except Exception as e:
            log(f"任务 {task_id} 发生未知错误: {str(e)}", important=True)
            import traceback
            log(f"错误详情: {traceback.format_exc()}")
            evaluation_tasks[task_id] = {
                "success": False,
                "error": str(e),
                "message": "评测过程中发生未知错误"
            }
            end_logging()
    
    @app.get("/")
    async def root():
        """API根路径，返回基本信息"""
        return JSONResponse(content={
            "name": "测试用例比较工具API",
            "version": "1.0.0",
            "description": "比较AI生成的测试用例与黄金标准测试用例，评估测试用例质量"
        })
    @app.post("/generate-from-feishu")
    async def generate_testcases_api(request: Request, data: dict):
        access_token = request.session.get("feishu_access_token")
        document_id = data.get("document_id")
        print(access_token)
        if not access_token:
            print("没找到access-token")
            original_url = "http://localhost:5173"  # 或 request.headers.get("Referer")
            auth_url = (
                "https://open.feishu.cn/open-apis/authen/authorize"
                f"?response_type=code&client_id={APP_ID}&redirect_uri={REDIRECT_URI}&state={original_url}"
            )
            return RedirectResponse(auth_url)




    # 上传接口
    @app.post("/upload-doc")
    async def upload_doc(file: UploadFile = File(...)):
        filename = file.filename.lower()
        if not (filename.endswith(".pdf") or filename.endswith(".docx")):
            raise HTTPException(status_code=400, detail="仅支持 .docx 和 .pdf 文件")

        file_bytes = await file.read()

        try:
            extracted_text=extract_markdown(file_bytes, filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文档解析失败: {str(e)}")

        try:
            llm_response = await call_deepseek_llm(extracted_text)
            return {
                "success": True,
                "markdown": llm_response["markdown"],
                "json":llm_response["json"]
            }
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"模型调用失败: {str(e)}")


    class TextRequest(BaseModel):
        text: str

    @app.post("/generate-from-text")
    async def generate_from_text(data: TextRequest):
        if not data.text or len(data.text.strip()) < 10:
            raise HTTPException(status_code=400, detail="输入文本不能为空或太短")

        try:
            result = await call_deepseek_llm(data.text)
            print(result)
            return {
                "success": True,
                "markdown": result["markdown"],
                "json":result["json"]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型调用失败: {str(e)}")






    APP_ID = "cli_a8ef8e2bca7bd01c"
    APP_SECRET = "7WJD5NwtsIDfGhwRhI6HEfmlWAULqQA5"
    REDIRECT_URI = "http://localhost:8000/call-back"


    @app.get("/test1")
    async def test1():
        redirect_uri = 'http://localhost:8000/test2'
        return RedirectResponse(
            url=redirect_uri,
            status_code=302
        )
    @app.get("/test2")
    async def test2():
        return JSONResponse(
            "CVCAT666"
        )

    @app.get("/test3")
    async def test3():
        return RedirectResponse(
            url="http://cvcat.site",
            status_code=302
        )


    @app.get("/test4")
    async def test4():
        return RedirectResponse(
            url="http://localhost:5173/predict",
            status_code=302
        )

    @app.get("/login")
    async def auth_callback(request: Request):
        redirect_uri = 'http://localhost:8000/call-back'
        encoded_redirect_uri = urllib.parse.quote(redirect_uri, safe="")
        auth_url = f'https://open.feishu.cn/open-apis/authen/v1/index?app_id={APP_ID}&redirect_uri={encoded_redirect_uri}'
        auth_url = f'https://open.feishu.cn/open-apis/authen/v1/index?app_id={APP_ID}&redirect_uri={encoded_redirect_uri}'
        print("https://open.feishu.cn/open-apis/authen/v1/index?app_id=cli_a8ef8e2bca7bd01c&redirect_uri=http%3A%2F%2Flocalhost%3A8000%2Fcall-back")
        print(auth_url)
        session = request.session
        print("session", session)
        if "feishu_access_token" in session:
            print("有access_token")
            access_token = session["feishu_access_token"]
            user_resp = requests.get(
                "https://open.feishu.cn/open-apis/authen/v1/user_info",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            try:
                user_data = user_resp.json()["data"]["open_id"]
                print(user_data)
            except Exception as e:
                print("access_token 过期了")
                return RedirectResponse(
                    auth_url,
                    # "https://cvcat.site"
                    status_code=302
                )
        else:
            print("没有access token")

            response = RedirectResponse(
                # "http://localhost:5173/Predict",
                # "https://cvcat.site",

                auth_url,
                status_code=302
            )
            response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            return response


    @app.get("/call-back")
    async def auth_callback(request: Request):
        code = request.query_params.get("code")
        if not code:
            raise HTTPException(status_code=400, detail="缺少 code")

        token_url = "https://open.feishu.cn/open-apis/authen/v2/oauth/token"
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": APP_ID,
            "client_secret": APP_SECRET,
            "redirect_uri": REDIRECT_URI
        }

        async with httpx.AsyncClient() as client:
            token_resp = await client.post(token_url, json=payload)
            token_data = token_resp.json()

            print(token_data)

            if token_data.get("code", 0) != 0:
                return JSONResponse(status_code=500, content={"error": token_data.get("msg", "token 获取失败")})

            access_token = token_data["access_token"]

            # 获取用户信息
            user_resp = await client.get(
                "https://open.feishu.cn/open-apis/authen/v1/user_info",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            user_data = user_resp.json()["data"]["open_id"]
            print(user_data)

            request.session["feishu_access_token"] = access_token
            # 重定向回当前页面或者首页
            redirect_url = request.query_params.get("state") or "http://localhost:5173"
            # resp = await client.get('https://open.feishu.cn/open-apis/drive/explorer/v2/root_folder/meta',
            #                     headers={"Authorization": f"Bearer {access_token}"})
            # root_folder_token = resp.json()['data']['root_folder_token']
            # resp = await client.get('https://open.feishu.cn/open-apis/drive/v1/files',
            #                     headers={"Authorization": f"Bearer {access_token}"},params={"folder_token":root_folder_token})
            # files = resp.json()["data"]["files"]
            # print(files[0])

            return RedirectResponse(redirect_url)


except ImportError as e:
    # 检测未安装的包
    import traceback
    traceback.print_exc()
    app = None

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # 命令行模式
        log("以命令行模式运行...", important=True)
        parser = argparse.ArgumentParser(description="测试用例比较工具")
        parser.add_argument("--ai", help="AI生成的测试用例文件路径")
        parser.add_argument("--golden", help="黄金标准测试用例文件路径")
        args = parser.parse_args(sys.argv[2:])
        main(args.ai, args.golden)
    else:
        # API模式（默认）
        if app:
            import uvicorn
            log("启动API服务器...", important=True)
            uvicorn.run("GenerateAndCompareCasesAPI:app", host="127.0.0.1", port=8000, reload=True)
        else:
            log("错误：未安装FastAPI和uvicorn，无法启动API服务", important=True)
            log("请安装所需库: pip install fastapi uvicorn", important=True)
            log("如需以命令行模式运行，请使用: python compare.py --cli", important=True)