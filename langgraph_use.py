import json
import re
import logging
import asyncio
import time
from typing import TypedDict
from langgraph.graph import StateGraph
from model_api import call_model
from typing import Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    prd_text: str
    prd_title: str
    requirements: str
    testcases: dict
    validated: str


async def extract_prd_title(state: GraphState) -> GraphState:
    logger.info("[Step 1] 尝试直接从 PRD 文本提取标题")
    lines = state['prd_text'].splitlines()
    for line in lines:
        line = line.strip()
        if line:
            line = re.sub(r"^#+\s*", "", line)
            if len(line) < 50:
                title = line.strip()
                logger.info(f"[Step 1] 提取标题成功: {title}")
                return {**state, "prd_title": title}
    logger.info("[Step 1] 未提取成功，调用模型")
    prompt = f"""请从以下产品需求文档中提取模块或系统名称作为文档标题，文中包含顺序图文，图片以 Markdown 格式插入，请结合文本和图片内容理解：
{state['prd_text'][:1500]}
（只返回标题，不要解释）"""
    try:
        res = await call_model(prompt)
        title = res.strip().splitlines()[0]
        title = re.sub(r"^#+\s*", "", title).strip()
    except Exception as e:
        logger.warning(f"标题提取失败，使用默认标题: {e}")
        title = "自动生成测试用例"
    return {**state, "prd_title": title}


async def extract_requirements(state: GraphState) -> GraphState:
    logger.info("[Step 2] 提取测试点")
    prompt = f"""你是一位资深测试工程师，根据以下产品需求文档，文中包含顺序图文，图片通过 Markdown 格式插入。请综合文本和图片内容，提取详细测试点（功能、易用、异常等维度），并按模块分类输出：
{state['prd_text']}

请按如下格式：
- 模块名：
  - 测试点1：
  - 测试点2：
"""
    try:
        requirements = await call_model(prompt)
        return {**state, "requirements": requirements.strip()}
    except Exception as e:
        logger.error(f"[Step 2] 测试点提取失败: {e}")
        raise


async def optimize_requirements(state: GraphState) -> GraphState:
    logger.info("[Step 3] 优化测试点")
    prompt = f"""你是一位测试专家，请对以下功能测试点内容进行检查和优化。产品需求文档包含顺序图文，图片以 Markdown 格式插入，请结合文本和图片内容理解：
目标：
1. 查漏补缺；
2. 确保测试点覆盖功能、易用性、兼容性、安全性、性能等；
3. 分类清晰，每个测试点都归属到模块，并注明测试维度（功能/异常/边界/兼容/安全等）。
请优化以下测试点内容，补充遗漏，分类清晰，并注明测试维度：
{state['requirements']}
"""
    try:
        optimized = await call_model(prompt)
        return {**state, "requirements": optimized.strip()}
    except Exception as e:
        logger.error(f"[Step 3] 优化失败: {e}")
        raise


MAX_CONCURRENT = 10
MAX_RETRIES = 2


async def generate_case(point: str, idx: int, semaphore: asyncio.Semaphore) -> Union[dict, None]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                start = time.time()
                logger.info(f"生成第 {idx} 个用例，第 {attempt} 次尝试")
                prompt = f"""你是一个测试用例生成专家。根据以下测试点，结合产品需求文档中的顺序图文（图片通过 Markdown 格式插入），生成格式规范的测试用例，输出 JSON 格式，字段包括：

{{
  "title": "简洁明确的测试标题",
  "precondition": "按点列出前置条件，例如 1. 系统正常运行；2. 测试账号已登录",
  "steps": ["1. 打开页面", "2. 输入信息", "3. 点击提交"],
  "expected_results": ["1. 页面跳转成功", "2. 显示欢迎信息"]
}}

请仅返回 JSON，不要附加文字。测试点如下：
{point}
你最多根据这些测试点生成5个主要的测试用例。
"""
                resp = await call_model(prompt)
                case_json = json.loads(resp)
                if isinstance(case_json, list):
                    case_json = case_json[0]
                duration = time.time() - start
                logger.info(f" 第 {idx} 个用例生成成功，用时 {duration:.2f}s")
                return {
                    "case_id": f"{idx:03d}",
                    "title": case_json["title"],
                    "preconditions": case_json["precondition"],
                    "steps": case_json["steps"],
                    "expected_results": case_json["expected_results"]
                }
        except Exception as e:
            logger.warning(f" 第 {idx} 个测试点失败（第 {attempt} 次）：{str(e)[:100]}...")
            await asyncio.sleep(1)
    logger.error(f" 第 {idx} 个用例多次失败，跳过")
    return None


async def generate_testcases(state: GraphState) -> GraphState:
    logger.info("[Step 4] 生成测试用例")
    raw_points = [
        line.strip("-• 0123456789.").strip()
        for line in state["requirements"].splitlines()
        if line.strip() and len(line.strip()) > 4
    ]
    if not raw_points:
        raise ValueError("未能提取有效的测试点")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    start = time.time()
    tasks = [generate_case(p, i + 1, semaphore) for i, p in enumerate(raw_points)]
    all_results = await asyncio.gather(*tasks)

    # 成功用例
    test_cases = [c for c in all_results if c is not None]

    # 统一重新编号，避免跳号
    for new_idx, case in enumerate(test_cases, start=1):
        case["case_id"] = f"{new_idx:03d}"

    # 失败的保留原位信息（供重试）
    failed_cases = [
        {"case_id": f"{i + 1:03d}", "requirement": p}
        for i, (p, r) in enumerate(zip(raw_points, all_results)) if r is None
    ]

    logger.info(
        f"[Step 4] 用例生成完成: 成功 {len(test_cases)} 个，失败 {len(failed_cases)} 个，用时 {time.time() - start:.2f}s")
    return {
        **state,
        "testcases": {
            "test_suite": state["prd_title"],
            "test_cases": test_cases,
            "failed_cases": failed_cases
        }
    }


async def validate_testcases(state: GraphState) -> GraphState:
    logger.info("[Step 5] 校验测试用例，移除高度一致的重复项")

    testcases = state.get("testcases", {}).get("test_cases", [])
    if not testcases:
        logger.warning("无测试用例可校验")
        return {
            **state,
            "validated": "无测试用例"
        }

    seen = set()
    unique_testcases = []
    for case in testcases:
        key = (case["title"], tuple(case["steps"]), tuple(case["expected_results"]))
        if key not in seen:
            seen.add(key)
            unique_testcases.append(case)
        else:
            logger.info(f"移除重复用例: {case['title']}")

    # 重新编号
    for idx, case in enumerate(unique_testcases, start=1):
        case["case_id"] = f"{idx:03d}"

    removed_count = len(testcases) - len(unique_testcases)
    validated_msg = f"共去除重复用例 {removed_count} 条" if removed_count else "未发现重复用例"
    logger.info(validated_msg)

    logger.info(f"[Step 5] 校验完成：{validated_msg}")

    return {
        **state,
        "testcases": {
            "test_suite": state["prd_title"],
            "test_cases": unique_testcases
        }
    }


workflow = StateGraph(GraphState)
workflow.add_node("step1_extract_title", extract_prd_title)
workflow.add_node("step2_extract_requirements", extract_requirements)
workflow.add_node("step3_optimize_requirements", optimize_requirements)
workflow.add_node("step4_generate_testcases", generate_testcases)
workflow.add_node("step5_validate_testcases", validate_testcases)

workflow.set_entry_point("step1_extract_title")
workflow.add_edge("step1_extract_title", "step2_extract_requirements")
workflow.add_edge("step2_extract_requirements", "step3_optimize_requirements")
workflow.add_edge("step3_optimize_requirements", "step4_generate_testcases")
workflow.add_edge("step4_generate_testcases", "step5_validate_testcases")
workflow.set_finish_point("step5_validate_testcases")

graph = workflow.compile()
