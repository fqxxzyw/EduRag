"""
rag_qa 核对代码目录
检索策略选择器
"""
from langchain_core.prompts import PromptTemplate
from rag_qa.core.llm_provider import get_llm
from base import logger, Config


# todo
class StrategySelector:
    """用于根据用户查询选择最合适的检索增强策略.
    """
    # todo 初始化方法: 从大模型工厂获取LLM对象, 加载策略选择的提示模块.
    def __init__(self):
        self.logger = logger
        # LLM对象: 从大模型工厂获取（单例，全局只创建一次）
        self.llm = get_llm()
        self.logger.info(f"LLM初始化完成, 使用的模型: {Config.LLM_MODEL}")
        # 获取策略选择的提示模板(定义好的提问格式, 用于引导大模型选择策略)
        self.prompt_template = self._get_strategy_prompt()

    # todo 3.4 : 整合模板和大模型调用,
    def select_strategy(self, query):
        """选择检索策略的核心方法
        根据用户查询, 选择最合适的检索增强策略.
        :param query: 用户输入的查询文本, 例如: "什么是AI?"
        :return: 字符串, 选中的检索策略名称, 例如: 直接检索, 子查询检索, HyDE检索, 回溯检索
        """
        # 1. 格式化提示模板, 将用户查询 填充到 提示模板的 {query} 变量中, 生成发给大模型的完整提示.
        prompt = self.prompt_template.format(query=query)

        # 2.调用 LLM 获取检索策略
        try:
            response = self.llm.invoke(prompt)
            strategy = response.content.strip()
            self.logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")
            return strategy
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return "直接检索"

    def _get_strategy_prompt(self):
        """定义提示模板(类似: 填空题)
            用于告诉大模型如何根据用户查询选择检索策略.
            模板中包含4中检索策略的适用场景.
        :return: LangChain的PromptTemplate对象
        """
        template_str = """# Role
你是一个专业的 RAG（检索增强生成）查询意图分析专家。你的唯一任务是：精准识别用户查询的核心意图，并将其映射到最合适的检索策略。

# Strategies & Decision Boundaries
请严格基于以下定义进行判断，不要自行创造策略：

1. **直接检索**
   - 🔑 核心特征：【事实明确】、【实体具体】、【无需推理】。
   - ✅ 适用：询问具体数值、定义、时间表、特定代码写法、单一知识点。
   - 📝 示例："AI学科学费是多少？" / "Python如何读取CSV？" / "TCP三次握手过程？"

2. **假设问题检索 (HyDE)**
   - 🔑 核心特征：【语义抽象】、【概念宽泛】、【需要发散/总结】。
   - ✅ 适用：询问“有哪些应用”、“前景如何”、“为什么重要”等没有标准答案的开放性问题。
   - 📝 示例："人工智能在教育领域的应用有哪些？" / "给我讲个故事吧？" / "AI对未来的影响？"

3. **子查询检索**
   - 🔑 核心特征：【多实体对比】、【多维度并列】、【显式连接词】。
   - ✅ 适用：包含"A和B的区别"、"X与Y的优缺点"、"同时列出..."等需要拆分检索再合并的问题。
   - 📝 示例："比较Milvus和Zilliz Cloud的优缺点" / "Docker和虚拟机的区别？" / "Java和Python哪个更适合初学者？"

4. **回溯问题检索**
   - 🔑 核心特征：【场景复杂】、【口语化/冗余】、【隐含前提】、【长难句】。
   - ✅ 适用：描述一个具体业务场景并询问可行性，或包含大量背景描述的复杂问题，需先提取核心痛点转化为标准问法。
   - 📝 示例："我有个100亿条记录的数据集，存Milvus可以吗？" -> 转化 -> "Milvus支持的最大数据量级及性能瓶颈？"

# Workflow
1. 分析用户查询 `{query}` 的语义结构。
2. 逐一匹配上述 4 种策略的【核心特征】。
3. 确定唯一最佳策略。

# Output Rules (CRITICAL)
- ⛔ **禁止**输出任何分析过程、推理步骤、标点符号或额外文本。
- ⛔ **禁止**使用 Markdown 格式（如加粗、列表）。
- ✅ **仅允许**返回以下四个字符串之一：
  直接检索
  假设问题检索
  子查询检索
  回溯问题检索

# Input
Query: {query}

# Output
        """
        return PromptTemplate(
            template=template_str,
            # 模板中的变量, 将来会被用户实际查询替换.
            input_variables=["query"],
        )

    def _test_SELECT_STRATEGY(self):
        print('测试 ➡️ 选择检索策略功能：')
        retrieval_strategies = {
            "直接检索": [
                "1024乘以768等于多少？",
                "Python如何读取CSV文件？",
                "解释一下什么是RESTful API。",
                "AI学科的最新课程安排是什么时候？",
                "Java零基础入门班学费是多少？",
                "数据库三大范式是什么？",
                "写一个SQL查询，找出'students'表中所有姓'张'的学生。",
                "测试开发课程包含自动化测试工具的使用吗？",
                "大数据课程的老师背景如何？",
                "5的阶乘是多少？"
            ],
            "假设问题检索": [
                "AI专业的就业前景怎么样？",
                "人工智能现在有哪些主要的应用领域？",
                "人工智能专业的学习门槛高吗？",
                "人工智能专业毕业后的薪资待遇如何？",
                "人工智能专业需要具备哪些能力？",
                "人工智能专业的核心竞争力如何？",
                "人工智能专业学完后可以做什么？",
                "人工智能专业的课程难易程度如何？",
                "人工智能专业毕业生的竞争力如何？",
                "人工智能专业毕业后好找工作吗？"
            ],
            "子查询检索": [
                "什么是Docker？它和虚拟机的区别？",
                "Java和Python哪个更适合初学者？",
                "解释HTTP和HTTPS的主要区别。",
                "解释一下并发和并行的区别。",
                "Python中的列表和元组有什么不同？",
                "什么是黑盒测试和白盒测试？",
                "自动化测试和手动测试哪个更重要？",
                "什么是链表？它和数组有什么区别？",
                "Java Web开发和Java后端开发有什么区别？",
                "Python的多线程和多进程有什么区别？"
            ],
            "回溯问题检索": [
                "我有个100亿条记录的数据集，把它存储到Milvus中进行查询。可以吗？",
                "学习Java需要准备什么样的电脑配置？",
                "大数据分析需要学习哪些编程语言？",
                "大数据课程会涉及到Hadoop和Spark吗？",
                "运维课程有没有关于Kubernetes的内容？",
                "运维课程有没有针对网络安全的讲解？",
                "运维需要学习Linux操作系统吗？",
                "大数据课程对数学基础要求高吗？",
                "测试开发课程对英语水平有要求吗？",
                "AI课程是不是很难学？"
            ]
        }

        import random
        # 从字典中随机选择一个分类（key）
        random_category = random.choice(list(retrieval_strategies.keys()))
        # 从该分类对应的问题列表中随机选择一个问题
        random_query = random.choice(retrieval_strategies[random_category])
        # 打印结果
        print(f"【随机问题】: {random_query}")
        print(f"【所属分类】: {random_category}")
        # 4. 调用模型，输出策略名称进行对比
        ret = self.select_strategy(query=random_query)
        print(f"【模型输出】: {ret}")


if __name__ == '__main__':

    ss = StrategySelector()
    # ss._test_SELECT_STRATEGY()

    ss.select_strategy(query="人工智能专业毕业生的竞争力如何?")
    ss.select_strategy(query='Python中的列表和元组有什么不同？')
