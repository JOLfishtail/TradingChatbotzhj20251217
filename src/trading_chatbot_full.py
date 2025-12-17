"""
Trading Chatbot - Full Implementation for Interview
Using your existing environment with LangChain 1.x, LangGraph 1.x, etc.
"""
import os
import asyncio
import json
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# LangChain 1.x imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph, END
from langchain_community.embeddings import DashScopeEmbeddings

# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import langchain
import langgraph

# ==================== 配置 ====================
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"  # 替换为你的密钥


class Config:
    EMBEDDING_MODEL = "text-embedding-ada-002"
    CHROMA_PERSIST_DIR = "./chroma_db"
    AI_MODEL_NAME = os.getenv("AI_MODEL", "qwen-plus")
    AI_API_BASE = os.getenv(
        "AI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    AI_DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", 0.0))


# ==================== 数据模型 ====================
class AgentType(Enum):
    FAQ = "faq_agent"
    TRADING = "trading_agent"
    MARKET = "market_agent"
    SAFETY = "safety_filter"


class Intent(Enum):
    WITHDRAW = "withdraw"
    DEPOSIT = "deposit"
    BUY = "buy"
    SELL = "sell"
    CLEAR = "clear"
    ALERT = "alert"
    PRICE = "price"
    MARKET = "market"
    FAQ = "faq"  # 新增
    UNKNOWN = "unknown"


@dataclass
class AgentResponse:
    """代理响应"""
    response: str
    agent_used: str
    confidence: float
    tools_called: List[str]
    metadata: Dict[str, Any]


class AgentState(TypedDict):
    """LangGraph状态"""
    query: str
    user_id: str
    intent: str
    response: str
    agent_used: str
    tools_called: List[str]
    confidence: float
    metadata: Dict[str, Any]
    needs_clarification: bool
    clarification_question: str


# ==================== RAG系统 ====================
class TradingRAG:
    """交易RAG系统"""

    def __init__(self):
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",  # DashScope提供的文本嵌入模型
            dashscope_api_key=Config.AI_DASHSCOPE_API_KEY
        )
        self.llm = ChatOpenAI(
            model=Config.AI_MODEL_NAME,  # 例如 “qwen-plus”
            openai_api_key=Config.AI_DASHSCOPE_API_KEY,  # 你的DashScope API密钥
            openai_api_base=Config.AI_API_BASE,  # 即 “https://dashscope.aliyuncs.com/compatible-mode/v1”
            temperature=Config.AI_TEMPERATURE,
            streaming=True  # 如果需要流式输出可以保留
        )
        self.vector_store = None

    def initialize(self, documents: List[Document]):
        """初始化向量存储"""
        print("🔄 初始化RAG系统...")

        # 文本分割
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n"
        )
        texts = text_splitter.split_documents(documents)

        # 创建向量存储
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )
        # 移除这一行，因为ChromaDB 0.4.x会自动持久化
        # self.vector_store.persist()  # 删除或注释掉这一行

        print(f"✅ RAG系统初始化完成，加载了 {len(texts)} 个文档块")

    def query(self, question: str, k: int = 3) -> str:
        """查询RAG系统"""
        if not self.vector_store:
            return "RAG系统未初始化"

        # 检索相关文档
        docs = self.vector_store.similarity_search(question, k=k)

        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in docs])

        # 生成回答
        prompt = f"""你是一个专业的交易平台助手。根据以下信息回答问题。
如果信息不足，请礼貌地说明。

相关信息：
{context}

用户问题：{question}

请提供准确、有帮助的回答："""

        response = self.llm.invoke(prompt)
        return response.content


# ==================== 交易工具 ====================
class TradingTools:
    """交易工具（模拟）"""

    def __init__(self):
        self.positions = {
            "AAPL": {"quantity": 10, "avg_price": 170.50},
            "TSLA": {"quantity": 5, "avg_price": 240.00},
            "GOOGL": {"quantity": 3, "avg_price": 140.00}
        }
        self.stock_prices = {
            "AAPL": 175.50,
            "TSLA": 245.30,
            "GOOGL": 142.80,
            "MSFT": 330.20,
            "AMZN": 145.60
        }
        self.order_history = []

    def buy_stock(self, symbol: str, quantity: int, user_id: str) -> Dict:
        """买入股票"""
        symbol = symbol.upper()
        if symbol not in self.stock_prices:
            return {"success": False, "message": f"股票{symbol}不存在"}

        price = self.stock_prices[symbol]
        total_cost = price * quantity

        # 更新持仓
        if symbol in self.positions:
            self.positions[symbol]["quantity"] += quantity
        else:
            self.positions[symbol] = {
                "quantity": quantity,
                "avg_price": price
            }

        # 记录订单
        order = {
            "order_id": f"BUY_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "user_id": user_id,
            "symbol": symbol,
            "action": "BUY",
            "quantity": quantity,
            "price": price,
            "total_cost": total_cost,
            "timestamp": datetime.now().isoformat()
        }
        self.order_history.append(order)

        return {
            "success": True,
            "message": f"成功买入{quantity}股{symbol}，价格${price:.2f}，总成本${total_cost:.2f}",
            "order_id": order["order_id"],
            "details": order
        }

    def sell_stock(self, symbol: str, quantity: int, user_id: str) -> Dict:
        """卖出股票"""
        symbol = symbol.upper()

        if symbol not in self.positions:
            return {"success": False, "message": f"没有{symbol}的持仓"}

        if self.positions[symbol]["quantity"] < quantity:
            return {
                "success": False,
                "message": f"持仓不足。当前持有{self.positions[symbol]['quantity']}股{symbol}"
            }

        price = self.stock_prices.get(symbol, 100.0)
        total_revenue = price * quantity

        # 更新持仓
        self.positions[symbol]["quantity"] -= quantity
        if self.positions[symbol]["quantity"] == 0:
            del self.positions[symbol]

        # 记录订单
        order = {
            "order_id": f"SELL_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "user_id": user_id,
            "symbol": symbol,
            "action": "SELL",
            "quantity": quantity,
            "price": price,
            "total_revenue": total_revenue,
            "timestamp": datetime.now().isoformat()
        }
        self.order_history.append(order)

        return {
            "success": True,
            "message": f"成功卖出{quantity}股{symbol}，价格${price:.2f}，总收入${total_revenue:.2f}",
            "order_id": order["order_id"],
            "details": order
        }

    def clear_positions(self, symbol: str, user_id: str) -> Dict:
        """清仓"""
        symbol = symbol.upper()

        if symbol not in self.positions:
            return {"success": False, "message": f"没有{symbol}的持仓"}

        quantity = self.positions[symbol]["quantity"]
        return self.sell_stock(symbol, quantity, user_id)

    def stock_price_alert(self, symbol: str, target_price: float,
                          condition: str = "above", user_id: str = None) -> Dict:
        """设置价格提醒"""
        symbol = symbol.upper()
        current_price = self.stock_prices.get(symbol, 100.0)

        alert = {
            "alert_id": f"ALERT_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "user_id": user_id,
            "symbol": symbol,
            "target_price": target_price,
            "current_price": current_price,
            "condition": condition,
            "status": "active",
            "created_at": datetime.now().isoformat()
        }

        return {
            "success": True,
            "message": f"已设置{symbol}价格提醒：当价格{condition} ${target_price:.2f}时通知您（当前${current_price:.2f}）",
            "alert": alert
        }

    def get_stock_price(self, symbol: str) -> Dict:
        """获取股票价格"""
        symbol = symbol.upper()
        price = self.stock_prices.get(symbol)

        if price:
            return {
                "success": True,
                "symbol": symbol,
                "price": price,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": f"找不到股票{symbol}的价格信息"
            }

    def get_portfolio(self, user_id: str) -> Dict:
        """获取投资组合"""
        portfolio = {}
        total_value = 0.0

        for symbol, position in self.positions.items():
            current_price = self.stock_prices.get(symbol, 0)
            position_value = current_price * position["quantity"]
            total_value += position_value

            portfolio[symbol] = {
                "quantity": position["quantity"],
                "avg_price": position["avg_price"],
                "current_price": current_price,
                "position_value": position_value,
                "pnl": (current_price - position["avg_price"]) * position["quantity"]
            }

        return {
            "success": True,
            "user_id": user_id,
            "portfolio": portfolio,
            "total_value": total_value,
            "timestamp": datetime.now().isoformat()
        }


# ==================== 安全护栏 ====================
class SafetyGuardrail:
    """安全护栏"""

    def __init__(self):
        self.restricted_keywords = [
            "hack", "cheat", "illegal", "fraud", "scam",
            "insider trading", "market manipulation",
            "bypass", "unauthorized", "exploit"
        ]

        self.restricted_actions = [
            "transfer all money", "close all accounts",
            "delete account", "reset password",
            "show all users", "admin access"
        ]

    def check(self, query: str, user_id: str) -> Dict:
        """安全检查"""
        query_lower = query.lower()

        # 检查关键词
        for keyword in self.restricted_keywords:
            if keyword in query_lower:
                return {
                    "safe": False,
                    "action": "block",
                    "message": f"查询包含受限关键词：'{keyword}'",
                    "reason": "restricted_keyword"
                }

        # 检查危险操作
        for action in self.restricted_actions:
            if action in query_lower:
                return {
                    "safe": False,
                    "action": "block",
                    "message": f"操作不被允许：'{action}'",
                    "reason": "restricted_action"
                }

        # 高频交易检查（简单示例）
        # 在实际系统中这里会有更复杂的逻辑

        return {
            "safe": True,
            "action": "allow",
            "message": "安全检查通过"
        }


# ==================== 意图分类器 ====================
class IntentClassifier:
    """意图分类器"""

    def __init__(self):
        self.intent_patterns = {
            Intent.WITHDRAW: ["withdraw", "提现", "取出", "取钱", "withdrawal"],
            Intent.DEPOSIT: ["deposit", "存款", "存入", "存钱"],
            Intent.BUY: ["buy", "买入", "购买", "购入", "开仓"],
            Intent.SELL: ["sell", "卖出", "出售", "卖掉", "平仓"],
            Intent.CLEAR: ["clear", "清仓", "清空", "全部卖出"],
            Intent.ALERT: ["alert", "提醒", "通知", "预警"],
            Intent.PRICE: ["price", "价格", "股价", "行情", "报价"],
            Intent.MARKET: ["market", "市场", "分析", "趋势", "行情"],
            # 添加费用相关关键词
            # 扩展FAQ关键词，包含"how to"查询
            Intent.FAQ: [
                "fee", "fees", "手续费", "佣金", "交易费", "trading fee", "commission",
                "how to", "what is", "help", "support", "question", "query", "guide",
                "tutorial", "manual", "instruction", "explain", "describe", "tell me about"
            ]
        }

    def classify(self, query: str) -> Intent:
        """分类意图"""
        query_lower = query.lower()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return intent

        return Intent.UNKNOWN


# ==================== 代理系统 ====================
class FAQAgent:
    """FAQ代理"""

    def __init__(self, rag: TradingRAG):
        self.rag = rag
        self.name = AgentType.FAQ.value

    async def process(self, query: str, metadata: Dict = None) -> AgentResponse:
        """处理FAQ查询"""
        try:
            # 使用RAG获取答案
            answer = self.rag.query(query)

            return AgentResponse(
                response=answer,
                agent_used=self.name,
                confidence=0.85,
                tools_called=["rag_retrieval", "llm_generation"],
                metadata={
                    "rag_used": True,
                    "source": "document_retrieval"
                }
            )
        except Exception as e:
            return AgentResponse(
                response=f"抱歉，处理问题时出现错误：{str(e)}",
                agent_used=self.name,
                confidence=0.3,
                tools_called=[],
                metadata={"error": str(e)}
            )


class TradingAgent:
    """交易代理"""

    def __init__(self, trading_tools: TradingTools):
        self.tools = trading_tools
        self.name = AgentType.TRADING.value

    async def process(self, query: str, intent: Intent,
                      user_id: str, metadata: Dict = None) -> AgentResponse:
        """处理交易查询"""
        query_lower = query.lower()
        tools_called = []
        action_result = None

        # 提取股票代码
        symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
        found_symbol = None
        for symbol in symbols:
            if symbol.lower() in query_lower:
                found_symbol = symbol
                break

        # 默认使用AAPL
        if not found_symbol:
            found_symbol = "AAPL"

        # 根据意图执行操作
        if intent == Intent.BUY:
            # 提取数量
            quantity = 10
            import re
            numbers = re.findall(r'\b\d+\b', query)
            if numbers:
                quantity = int(numbers[0])

            result = self.tools.buy_stock(found_symbol, quantity, user_id)
            tools_called = ["buy_stock"]
            action_result = result

        elif intent == Intent.SELL:
            quantity = 5
            import re
            numbers = re.findall(r'\b\d+\b', query)
            if numbers:
                quantity = int(numbers[0])

            result = self.tools.sell_stock(found_symbol, quantity, user_id)
            tools_called = ["sell_stock"]
            action_result = result

        elif intent == Intent.CLEAR:
            result = self.tools.clear_positions(found_symbol, user_id)
            tools_called = ["clear_positions"]
            action_result = result

        elif intent == Intent.ALERT:
            # 提取目标价格
            target_price = 180.0
            import re
            prices = re.findall(r'\b\d+\.?\d*\b', query)
            if prices:
                target_price = float(prices[0])

            result = self.tools.stock_price_alert(found_symbol, target_price, "above", user_id)
            tools_called = ["stock_price_alert"]
            action_result = result

        elif intent == Intent.PRICE:
            result = self.tools.get_stock_price(found_symbol)
            tools_called = ["get_stock_price"]
            action_result = result
        else:
            return AgentResponse(
                response="请提供具体的交易指令（买入、卖出、清仓、设置提醒等）",
                agent_used=self.name,
                confidence=0.4,
                tools_called=[],
                metadata={"error": "ambiguous_trading_instruction"}
            )

        # 根据操作结果构建响应
        if action_result.get("success", False):
            confidence = 0.95
            response = action_result["message"]
        else:
            confidence = 0.5
            response = f"操作失败：{action_result.get('message', '未知错误')}"

        return AgentResponse(
            response=response,
            agent_used=self.name,
            confidence=confidence,
            tools_called=tools_called,
            metadata=action_result
        )


class MarketAgent:
    """市场代理"""

    def __init__(self, trading_tools: TradingTools):
        self.tools = trading_tools
        self.name = AgentType.MARKET.value

    async def process(self, query: str, metadata: Dict = None) -> AgentResponse:
        """处理市场查询"""
        query_lower = query.lower()

        # 检查是否询问特定股票
        symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
        found_symbols = []

        for symbol in symbols:
            if symbol.lower() in query_lower:
                found_symbols.append(symbol)

        if found_symbols:
            # 获取多个股票价格
            price_info = []
            for symbol in found_symbols[:3]:  # 限制最多3个
                result = self.tools.get_stock_price(symbol)
                if result["success"]:
                    price_info.append(f"{symbol}: ${result['price']:.2f}")

            if price_info:
                response = f"当前股价：{'，'.join(price_info)}"
                confidence = 0.9
                tools_called = ["get_stock_price"]
                metadata = {"stocks_queried": found_symbols}
            else:
                response = "无法获取股价信息"
                confidence = 0.3
                tools_called = []
        else:
            # 一般市场分析
            response = "市场分析：今日科技股表现强劲，AAPL和TSLA领涨。建议关注财报季表现。"
            confidence = 0.7
            tools_called = ["market_analysis"]
            metadata = {"analysis_type": "general_market"}

        return AgentResponse(
            response=response,
            agent_used=self.name,
            confidence=confidence,
            tools_called=tools_called,
            metadata=metadata or {}
        )


# ==================== 主协调器（使用LangGraph） ====================
class TradingOrchestrator:
    """交易协调器"""

    def __init__(self):
        print("🚀 初始化交易聊天机器人...")

        # 初始化组件
        self.trading_tools = TradingTools()
        self.safety_checker = SafetyGuardrail()
        self.intent_classifier = IntentClassifier()

        # 初始化RAG并加载文档
        self.rag = TradingRAG()
        self._load_sample_documents()

        # 初始化代理
        self.faq_agent = FAQAgent(self.rag)
        self.trading_agent = TradingAgent(self.trading_tools)
        self.market_agent = MarketAgent(self.trading_tools)

        # 构建LangGraph工作流
        self.workflow = self._build_workflow()

        print("✅ 交易聊天机器人初始化完成")

    def _load_sample_documents(self):
        """加载示例文档"""
        sample_docs = [
            Document(
                page_content="如何提现：登录账户 -> 进入提现页面 -> 输入金额 -> 选择收款方式 -> 确认提现。每日限额$10,000。",
                metadata={"source": "faq", "type": "withdrawal"}
            ),
            Document(
                page_content="如何存款：支持银行转账、信用卡、支付宝、微信支付。进入存款页面选择方式并输入金额。最低存款$100。",
                metadata={"source": "faq", "type": "deposit"}
            ),
            Document(
                page_content="股票交易费用：买入费用0.1%，卖出费用0.1%。VIP用户可享受费率优惠。",
                metadata={"source": "faq", "type": "trading_fee"}
            ),
            Document(
                page_content="如何设置价格提醒：进入股票详情页 -> 点击提醒按钮 -> 设置目标价格 -> 确认。",
                metadata={"source": "faq", "type": "price_alert"}
            ),
            Document(
                page_content="苹果公司(AAPL)是全球最大的科技公司之一，主要产品包括iPhone、iPad、Mac等。",
                metadata={"source": "stock_info", "symbol": "AAPL"}
            ),
            Document(
                page_content="特斯拉(TSLA)是电动汽车和清洁能源公司，以创新和技术领先著称。",
                metadata={"source": "stock_info", "symbol": "TSLA"}
            )
        ]

        self.rag.initialize(sample_docs)

    def _build_workflow(self):
        """构建LangGraph工作流"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("safety_check", self._safety_check)
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("route", self._route_to_agent)
        workflow.add_node("faq_agent", self._call_faq_agent)
        workflow.add_node("trading_agent", self._call_trading_agent)
        workflow.add_node("market_agent", self._call_market_agent)
        workflow.add_node("clarify", self._ask_clarification)

        # 设置入口点
        workflow.set_entry_point("safety_check")

        # 添加边
        workflow.add_edge("safety_check", "classify_intent")
        workflow.add_edge("clarify", END)

        # 条件路由
        workflow.add_conditional_edges(
            "classify_intent",
            self._decide_next_step,
            {
                "needs_clarification": "clarify",
                "route": "route"
            }
        )

        workflow.add_conditional_edges(
            "route",
            self._select_agent,
            {
                "faq_agent": "faq_agent",
                "trading_agent": "trading_agent",
                "market_agent": "market_agent"
            }
        )

        workflow.add_edge("faq_agent", END)
        workflow.add_edge("trading_agent", END)
        workflow.add_edge("market_agent", END)

        return workflow.compile()

    def _safety_check(self, state: AgentState) -> AgentState:
        """安全检查节点"""
        safety_result = self.safety_checker.check(state["query"], state["user_id"])

        if not safety_result["safe"]:
            state["response"] = f"⚠️ 安全检查未通过：{safety_result['message']}"
            state["agent_used"] = AgentType.SAFETY.value
            state["confidence"] = 1.0
            state["tools_called"] = ["safety_check"]
            state["metadata"] = safety_result

        return state

    def _classify_intent(self, state: AgentState) -> AgentState:
        """意图分类节点"""
        if "response" in state and state["response"]:  # 安全检查已阻止
            return state

        intent = self.intent_classifier.classify(state["query"])
        state["intent"] = intent.value

        # 检查是否需要澄清
        if intent == Intent.UNKNOWN:
            state["needs_clarification"] = True
            state["clarification_question"] = (
                "我不太确定您想做什么。您是想要：\n"
                "1. 了解平台操作（如提现、存款）\n"
                "2. 进行交易（买入、卖出股票）\n"
                "3. 获取市场信息（股价、分析）\n"
                "请明确说明您的需求。"
            )
        else:
            state["needs_clarification"] = False

        return state

    def _decide_next_step(self, state: AgentState) -> str:
        """决定下一步"""
        if state.get("needs_clarification", False):
            return "needs_clarification"
        return "route"

    def _route_to_agent(self, state: AgentState) -> AgentState:
        """路由节点"""
        intent = Intent(state["intent"])

        if intent in [Intent.WITHDRAW, Intent.DEPOSIT, Intent.FAQ]:
            state["agent_used"] = AgentType.FAQ.value
        elif intent in [Intent.BUY, Intent.SELL, Intent.CLEAR, Intent.ALERT]:
            state["agent_used"] = AgentType.TRADING.value
        elif intent in [Intent.PRICE, Intent.MARKET]:
            state["agent_used"] = AgentType.MARKET.value
        else:
            state["agent_used"] = AgentType.FAQ.value  # 默认

        return state

    def _select_agent(self, state: AgentState) -> str:
        """选择代理"""
        return state["agent_used"]

    async def _call_faq_agent(self, state: AgentState) -> AgentState:
        """调用FAQ代理"""
        result = await self.faq_agent.process(state["query"])
        self._update_state_from_response(state, result)
        return state

    async def _call_trading_agent(self, state: AgentState) -> AgentState:
        """调用交易代理"""
        intent = Intent(state["intent"])
        result = await self.trading_agent.process(
            state["query"], intent, state["user_id"]
        )
        self._update_state_from_response(state, result)
        return state

    async def _call_market_agent(self, state: AgentState) -> AgentState:
        """调用市场代理"""
        result = await self.market_agent.process(state["query"])
        self._update_state_from_response(state, result)
        return state

    def _ask_clarification(self, state: AgentState) -> AgentState:
        """请求澄清"""
        state["response"] = state["clarification_question"]
        state["agent_used"] = "clarification_agent"
        state["confidence"] = 0.5
        state["tools_called"] = []
        return state

    def _update_state_from_response(self, state: AgentState, response: AgentResponse):
        """从响应更新状态"""
        state["response"] = response.response
        state["confidence"] = response.confidence
        state["tools_called"] = response.tools_called
        state["metadata"] = response.metadata

    async def process_query(self, query: str, user_id: str = "user_001") -> AgentState:
        """处理用户查询"""
        initial_state = AgentState(
            query=query,
            user_id=user_id,
            intent="",
            response="",
            agent_used="",
            tools_called=[],
            confidence=0.0,
            metadata={},
            needs_clarification=False,
            clarification_question=""
        )

        result = await self.workflow.ainvoke(initial_state)
        return result

    def get_portfolio(self, user_id: str = "user_001") -> Dict:
        """获取投资组合"""
        return self.trading_tools.get_portfolio(user_id)


# ==================== FastAPI 服务 ====================
app = FastAPI(title="Trading Chatbot API", version="1.0.0")

# 初始化协调器（在实际应用中应该使用依赖注入）
orchestrator = None


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的代码
    global orchestrator
    print("🚀 初始化交易聊天机器人...")
    orchestrator = TradingOrchestrator()
    yield
    # 关闭时的代码（如果需要）
    print("🛑 关闭交易聊天机器人...")
    # 清理代码

app = FastAPI(title="Trading Chatbot API", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "user_001"


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    confidence: float
    tools_called: List[str]
    metadata: Dict[str, Any]


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """聊天端点"""
    try:
        result = await orchestrator.process_query(
            query=request.message,
            user_id=request.user_id
        )

        return ChatResponse(
            response=result["response"],
            agent_used=result["agent_used"],
            confidence=result["confidence"],
            tools_called=result["tools_called"],
            metadata=result.get("metadata", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio")
async def get_portfolio(user_id: str = "user_001"):
    """获取投资组合"""
    try:
        portfolio = orchestrator.get_portfolio(user_id)
        return portfolio
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "orchestrator": "initialized" if orchestrator else "not_initialized",
            "rag": "ready",
            "agents": ["faq", "trading", "market"]
        }
    }

def get_package_version(package_name):
    """安全地获取包的版本号"""
    try:
        # 尝试标准方法
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except:
        # 备用方案：尝试直接访问 __version__ 属性
        try:
            if package_name == "langchain":
                import langchain
                return langchain.__version__
            elif package_name == "langgraph":
                import langgraph
                # 如果 langgraph 没有 __version__，就返回一个占位符
                return getattr(langgraph, "__version__", "unknown (check via pip)")
        except:
            return "unknown"
# ==================== CLI 演示 ====================
async def run_cli_demo():
    """运行CLI演示"""
    print("=" * 80)
    print("🤖 TRADING CHATBOT DEMO - Full Implementation")
    print("=" * 80)
    print(f"Using: LangChain {get_package_version('langchain')}, LangGraph {get_package_version('langgraph')}")
    print("=" * 80)

    # 初始化协调器
    bot = TradingOrchestrator()

    # 测试查询
    test_queries = [
        "how to withdraw money on the platform",
        "how to deposit money",
        "buy 10 shares of AAPL",
        "what is the price of TSLA",
        "sell 5 shares of TSLA",
        "market analysis for today",
        "clear my AAPL positions",
        "set price alert for TSLA at $250",
        "how to set up price alerts",
        "what are the trading fees on your platform"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"Query {i}: {query}")
        print(f"{'-' * 60}")

        result = await bot.process_query(query)

        print(f"Agent: {result['agent_used']}")
        print(f"Response: {result['response']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Tools: {result['tools_called']}")

        # 显示一些元数据
        if result.get('metadata'):
            metadata_preview = json.dumps(result['metadata'], ensure_ascii=False)[:100]
            if len(metadata_preview) > 100:
                metadata_preview = metadata_preview[:97] + "..."
            print(f"Metadata: {metadata_preview}")

    # 显示投资组合
    print(f"\n{'=' * 80}")
    print("📊 PORTFOLIO SUMMARY")
    print(f"{'=' * 80}")

    portfolio = bot.get_portfolio()
    if portfolio["success"]:
        for symbol, data in portfolio["portfolio"].items():
            print(f"{symbol}:")
            print(f"  Quantity: {data['quantity']} shares")
            print(f"  Avg Price: ${data['avg_price']:.2f}")
            print(f"  Current Price: ${data['current_price']:.2f}")
            print(f"  Position Value: ${data['position_value']:.2f}")
            print(f"  P&L: ${data['pnl']:.2f}")
            print()

        print(f"Total Portfolio Value: ${portfolio['total_value']:.2f}")
    else:
        print("Failed to get portfolio")

    print(f"\n{'=' * 80}")
    print("🎉 DEMO COMPLETE!")
    print("This implementation uses your existing environment:")

    # 使用get_package_version函数安全获取版本
    print(f"- LangChain {get_package_version('langchain')}")
    print(f"- LangGraph {get_package_version('langgraph')}")

    # 尝试获取其他包的版本
    try:
        import openai
        print(f"- OpenAI {get_package_version('openai')}")
    except:
        print("- OpenAI version unknown")

    try:
        import chromadb
        print(f"- ChromaDB {get_package_version('chromadb')}")
    except:
        print("- ChromaDB version unknown")

    try:
        import fastapi
        print(f"- FastAPI {get_package_version('fastapi')}")
    except:
        print("- FastAPI version unknown")

    print(f"{'=' * 80}")


# ==================== 主函数 ====================
if __name__ == "__main__":
    import sys
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "api":

        # 启动API服务
        print("Starting Trading Chatbot API on http://localhost:8000")
        print("API Documentation: http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # 运行CLI演示
        asyncio.run(run_cli_demo())