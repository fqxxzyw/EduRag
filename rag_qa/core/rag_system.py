"""
rag_qa 核对代码目录
rag核心流程
"""
import time, traceback
from rag_qa.core.prompts import RAGPrompts  # 提示模板
from rag_qa.core.query_classifier import QueryClassifier  # 查询分类器(意图识别)
from rag_qa.core.strategy_selector import StrategySelector  # 策略选择器(4种策略)
from rag_qa.core.vector_store import VectorStore  # 向量库混合检索
from rag_qa.core.llm_provider import get_llm  # 大模型工厂
from base import logger, Config  # 日志和配置


# todo 定义RAGSystem类: 封装RAG系统所需的核心组件(向量库, 大模型, 分类器等...)
class RAGSystem:
    # todo 初始化方法: 创建RAG系统所需的核心组件.
    def __init__(self):
        """初始化RAG系统所需组件
        """
        # 向量库对象
        self.vector_store = VectorStore()
        # 基础提示词模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        # 查询分类器: 用于判断用户查询是 "通用知识" 还是 "专业咨询"
        self.query_classifier = QueryClassifier(
            new_model_path=Config.BERT_QUERY_PATH)
        # 检索策略选择器: 用于根据用户查询选择最合适的检索策略
        # 直接检索, 子查询检索, HyDE检索, 回溯检索
        self.strategy_selector = StrategySelector()
        # 大模型对象: 从大模型工厂获取（单例，全局只创建一次）
        self.llm = get_llm()


    # todo 直接检索策略实现：向量库检索→[Document]
    def _retrieve_directly(self, query, source_filter):
        """直接检索: 用用户查询调用向量库的hybrid_search_with_rerank混合检索, 返回结果
        :param query: 用户的原始查询文本
        :param source_filter: 检索来源过滤条件, 例如'ai'→表示只检索和AI学科相关的文档.
        :return: list[Document]
        """
        logger.info(f"直接检索策略, 查询信息为: {query}")
        try:
            docs = self.vector_store.hybrid_search_with_rerank(
                query=query,
                k=Config.RETRIEVAL_K,
                source_filter=source_filter
            )
            logger.info(f"直接检索策略检索结果: {len(docs)} 个文档")
            return docs # [Document]
        except Exception as e:
            # 若检索失败, 记录日志, 并返回空列表.
            logger.error(f"直接检索策略执行失败: {e}")
            return []

    def _test_RETRIEVE_DIRECTLY(
            self,
            query="如何使用Milvus进行分布式部署?",
            source_filter="ai"
        ):
        print('测试 ➡️ 直接检索策略：')
        ret = self._retrieve_directly(query=query, source_filter=source_filter)
        print(ret)

    # todo 回溯检索策略实现：提示词→大模型→简化后的问题→向量库混合检索→[Document]
    def _retrieve_backtracking(self, query, source_filter):
        """使用【回溯问题】提示词模板，向大模型提问，简化问题，再用简化的问题检索文档
        :param query: 用户的原始查询文本
        :param source_filter: 检索来源过滤条件, 例如'ai'→表示只检索和AI学科相关的文档.
        :return: list[Document]
        """
        logger.info(f"回溯问题策略检索, 查询信息为: {query}")
        # 提示词模板
        prompt = RAGPrompts.backtracking_prompt().format(query=query)

        try:
            # 大语言模型生成简化问题
            new_query = self.llm.invoke(prompt).content.strip()
            logger.info(f"生成的简化问题为: {new_query}")

            # 用回溯问题调用向量库检索, 调用hybrid_search_with_rerank混合检索, 并返回结果
            docs = self.vector_store.hybrid_search_with_rerank(
                query=new_query,  # 检索关键词: 简化后的回溯问题.
                k=Config.RETRIEVAL_K,  # 检索数量, 从conf配置文件加载.
                source_filter=source_filter  # 检索来源过滤条件.
            )
            logger.info(f"回溯问题策略检索结果: {len(docs)} 个文档")
            return docs
        except Exception as e:
            # 若生成回溯问题或检索失败, 记录日志, 并返回空列表.
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    def _test_RETRIEVE_BACKTRACKING(
            self,
            query="如何使用Milvus进行分布式部署?",
            source_filter="ai"
        ):
        print('测试 ➡️ 回溯问题策略：')
        ret = self._retrieve_backtracking(query=query, source_filter=source_filter)
        print(ret)


    # todo 子查询检索策略实现：提示词→大模型→多个子问题→向量库混合检索→[Document]
    def _retrieve_subqueries(self, query, source_filter):
        """使用【子查询检索】提示词模板，向大模型提问，获取多个子问题
            分别向量检索后，合并去重、返回
        :param query: 用户的原始查询文本
        :param source_filter: 检索来源过滤条件, 'ai' → 只检索AI学科相关文档
        :return: list[Document]
        """
        # 子查询提示词
        logger.info(f"子查询策略进行检索, 查询信息为: {query}")
        prompt = RAGPrompts.subquery_prompt().format(query=query)
        try:
            # 大模型生成子查询
            new_querys = self.llm.invoke(prompt).content.strip()
            # 拆分子查询: 根据换行符进行拆分
            querys = [q.strip() for q in new_querys.split('\n') if q.strip()]
            logger.info(f"生成的子查询为: {querys}")
            if not querys: # 如果子查询生成为空列表, 返回空列表.
                logger.warning("子查询生成无效, 请检查模型生成结果.")
                return []

            # 遍历子查询, 分别检索并收集结果.
            all_docs = []  # 存储所有子查询结果
            for query in querys:
                # 用子查询调用向量库的hybrid_search_with_rerank混合检索, 获取相关文档.
                # k是conf.RETRIEVAL_K的一般(避免单条子查询返回过多文档)
                docs = self.vector_store.hybrid_search_with_rerank(
                    query=query,                 # 检索关键词: 子查询.
                    k = Config.RETRIEVAL_K // 2, # 检索数量, 从conf配置文件加载.
                    source_filter=source_filter  # 检索来源过滤条件.
                )
                # 添加子查询结果到总结果列表.
                all_docs.extend(docs) # 列表队尾拼接列表
                logger.info(f"子查询: {query}, 检索结果: {len(docs)} 个文档")

            # 6.对所有检索结果去重: 基于文档内容(page_content)去重.
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}  # 用文档内容做键, 字典的键具有唯一性.
            unique_docs = list(unique_docs_dict.values())
            # print(unique_docs)
            logger.info(f"所有子查询共检索到{len(all_docs)}个文档, 去重后剩 {len(unique_docs)} 个文档")
            # 返回去重后的文档列表[Document]
            return unique_docs
        except Exception as e:
            # 异常处理: 若生成子查询或检索失败, 记录日志, 并返回空列表.
            logger.error(f"子查询策略执行失败: {e}")
            return []

    def _test_RETRIEVE_SUBQUERIES(self):
        print('测试 ➡️ 子查询策略：')
        self._retrieve_subqueries(
            query="我有1个包含100亿条记录的数据集, 想把它存储到Milvus中进行查询, OK吗?",
            source_filter="ai"
        )


    # todo HyDE检索策略实现：提示词→大模型→假设的答案→向量库混合检索→[Document]
    def _retrieve_hyde(self, query, source_filter):
        """使用【HyDE策略(生成假设答案, 用假设答案检索文档)】提示词模板，向大模型提问
            用大模型回答再去混合检索文档
        :param query: 用户的原始查询文本
        :param source_filter: 检索来源过滤条件, 例如: 'ai' -> 表示只检索和AI学科相关的文档.
        :return: list[Document]
        """
        logger.info(f"HyDE策略进行检索, 查询信息为: {query}")
        prompt = RAGPrompts.hyde_prompt().format(query=query)

        try:
            # 使用大语言模型生成假设答案: 用户模板格式化原始查询, 输入LLM生成假设答案.
            hyde_answer = self.llm.invoke(prompt).content.strip()
            logger.info(f"生成的假设答案为: {hyde_answer}")

            # 用假设答案调用向量库混合检索, 获取相关文档.
            docs = self.vector_store.hybrid_search_with_rerank(
                query=hyde_answer,                  # 检索关键词: 假设答案.
                k = Config.RETRIEVAL_K,             # 检索数量, 从conf配置文件加载.
                source_filter=source_filter         # 检索来源过滤条件.
            )
            logger.info(f"HyDE策略: 假设答案检索结果: {len(docs)} 个文档")
            return docs     # 返回相关文档.

        except Exception as e:
            # 异常处理: 若生成假设答案或检索失败, 记录日志, 返回空列表.
            logger.error(f"HyDE策略执行失败: {e}")
            return []

    def _test_RETRIEVE_HYDE(self):
        print('测试 ➡️ HyDE策略：')
        self._retrieve_hyde(
            query="人工智能在教育领域的应用有哪些具体方向?",
            source_filter="ai"
        )


    # todo 💡【检索统一入口】根据策略进行向量检索：获取策略→调用相应策略检索函数→[Document]
    def _retrieve_entry(self, query, source_filter=None, strategy=None):
        """检索策略执行的 统一入口
        根据检索策略调用对应的检索方法，返回[Document]
        :param query: 用户的查询文本.
        :param source_filter: 检索来源过滤条件.
        :param strategy: None|直接检索|回溯问题检索|子查询检索|假设问题检索
        :return: list[Document] 最终上下文文档列表
        """
        # 若未指定策略, 自动选择检索策略: 调用策略选择器的select_strategy方法.
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)  # 获取策略选择结果

        # 根据策略调用对应的检索方法, 获取候选文档列表.
        if strategy == "回溯问题检索":
            ranked_docs = self._retrieve_backtracking(query, source_filter)
        elif strategy == "子查询检索":
            ranked_docs = self._retrieve_subqueries(query, source_filter)
        elif strategy == "假设问题检索":
            ranked_docs = self._retrieve_hyde(query, source_filter)
        else:  # 默认 或“直接检索”
            ranked_docs = self._retrieve_directly(query, source_filter)
        logger.info(f'策略: {strategy}, 检索到 {len(ranked_docs)} 个候选文档')
        # 筛选最终上下文文档, 截取前conf.CANDIDATE_M个文档作为最终上下文文档.(控制上下文长度, 避免LLM输入超限)

        # 只获取前M个文档.
        final_ranked_docs = ranked_docs[:Config.CANDIDATE_M]
        # 打印日志, 并返回结果
        logger.info(f'最终上下文文档: {len(final_ranked_docs)} 个!')
        return final_ranked_docs

    def _test_RETRIEVE_ENTRY(self):
        print('测试 ➡️ 检索并合并:')
        self._retrieve_entry(
            query="AI就业方向?",
            source_filter="ai",
            # strategy=None|直接检索|回溯问题检索|子查询检索|假设问题检索
        )


    # todo 💡【RAG核心流程】对外接口
    def generate_answer(self, query, source_filter=None):
        """ RAG系统对外核心接口:
            接收用户查询, 完成 "查询分类→策略选择→文档检索→答案生成" 全流程
        :param query: 用户的原始查询文本, 例如: "地球为什么圆的?"
        :param source_filter: 检索来源过滤条件, 仅对"专业咨询"生效
            ai→只检索AI学科的相关文档, java→只检索java学科的相关文档...
        :return: 生成的最终答案
        """
        logger.info(f'开始处理查询: {query}, 学科过滤: {source_filter}')
        # 记录查询开始时间，用于计算整个查询耗时
        start_time = time.time()
        # 1. todo 判断查询类型 (通用知识, 专业咨询) 🔙 查询分类器
        query_category = self.query_classifier.predict_category(query)
        logger.info(f'查询分类结果: {query_category}')
        # 2. todo "通用知识"类查询: 直接调用LLM生成答案 (不检索文档)
        if query_category == '通用知识':
            logger.info("查询为<<通用知识>>, 直接调用LLM生成答案")
            # 2.1 提示词模板: 通用知识无需上下文, 仅传入问题和客服电话
            prompt_input = self.rag_prompt.format(
                context="",  # 通用知识无需上下文, 仅传入问题
                question=query,  # (用户要查询的)问题
                phone=Config.CUSTOMER_SERVICE_PHONE  # 配置中的客服电话, 用于错误提示
            )
            try: # 2.2 大模型生成答案 并返回结果
                answer = self.llm.invoke(prompt_input).content.strip()
            except Exception as e: # 异常时返回包含客服电话的错误提示.
                logger.error(f"直接调用LLM失败: {e}")
                logger.error(traceback.format_exc())
                answer = f'抱歉, 处理您的通用知识问题时出错, 请联系人工客服: {Config.CUSTOMER_SERVICE_PHONE}'
            # 记录通用知识查询的处理结果(耗时, 查询内容等...)
            processing_time = time.time() - start_time
            logger.info(f'处理通用知识查询结束, 耗时: {processing_time:.2f}秒, 查询内容: {query}')
            # 返回查询结果.
            return answer

        # 3. todo "专业咨询"类查询，执行完整RAG流程：策略选择→文档检索→结合上下文生成答案
        logger.info("查询为<<专业咨询>>, 执行完整RAG流程")
        # 3.1 检索策略 🔙 调用策略选择器确定最合适的检索策略
        strategy = self.strategy_selector.select_strategy(query)
        # 3.2 检索相关文档作为上下文 🔙 调用 _retrieve_entry 函数
        context_docs = self._retrieve_entry(
            query, # 用户查询文本
            source_filter=source_filter, # 学科过滤
            strategy=strategy # 检索策略
        )
        # 3.3 构建上下文文本 🔙 将文档列表的page_content属性拼接位字符串.
        if context_docs: # 每个文档占一段, 用两个换行分割
            context = '\n\n'.join([doc.page_content for doc in context_docs])
            logger.info(f'最终上下文文档: {len(context_docs)} 个, 总长度: {len(context)} 字')
        else: # 若未检索到文档, 上下文为空字符串.
            context = ''
            logger.info(f'未检索到文档, 上下文为空字符串')
        # 3.4 提示词
        prompt = self.rag_prompt.format(
            context=context,        # 上下文
            question=query,         #  问题
            phone=Config.CUSTOMER_SERVICE_PHONE  # 客服电话
        )
        # 3.5 大模型生成答案，基于上下文
        try:
            answer = self.llm.invoke(prompt).content.strip()
        except Exception as e:
            logger.error(f"调用LLM失败: {traceback.format_exc()}")
            answer = f'抱歉, 处理您的专业咨询时出错, 请联系人工客服: {Config.CUSTOMER_SERVICE_PHONE}'
        # 计算耗时
        processing_time = time.time() - start_time
        logger.info(f'查询处理完成, 耗时: {processing_time:.2f}秒, 查询内容为: {query}')
        # 返回查询结果
        return answer

    def _test_GENERATE_ANSWER(self):
        print('测试 ➡️ RAG主逻辑:')
        answer = self.generate_answer(
            query="黑马AI课程大纲有哪些主要内容?",
            source_filter='ai'
        )
        print(answer)


    # TODO 增加：💡【RAG核心流程（带历史会话）】对外接口
    # 提示词 新函数要覆盖旧函数的功能，且不影响旧函数的调用，新函数完全能够替换旧函数，不影响调用逻辑（旧调用的参数、返回值不受影响）
    def generate_answer_withhistory(self,
                                     query,
                                     source_filter=None,
                                     history=None):
        """ 【新】RAG系统对外核心接口:
            接收用户查询, 完成 "查询分类→策略选择→文档检索→答案生成" 全流程
        :param query: 用户的原始查询文本, 例如: "地球为什么圆的?"
        :param source_filter: 检索来源过滤条件, 仅对"专业咨询"生效
            ai→只检索AI学科的相关文档, java→只检索java学科的相关文档...
        :param history: 用户和机器人的对话历史，用于生成提示词
        """
        logger.info(f'开始处理查询: {query}, 学科过滤: {source_filter}')
        # 记录查询开始时间，用于计算整个查询耗时
        start_time = time.time()

        '''TODO 处理history历史对话'''
        history_context = "" # 初始化空对话历史
        # 校验历史参数类型，非列表直接置空
        if history is not None and not isinstance(history, list):
            logger.warning(f'因历史消息格式无效：{type(history)}，忽略历史消息')
            history = []
        # 截取最近5轮对话并拼接上下文
        if history:
            history = (f"Q:{h['question']}\nA:{h['answer']}" for h in history[-5:])
            history_context = "\n".join(history)
            '''history = [
                {"question": "1+1等于几", "answer": "等于2"},
                {"question": "什么是RAG", "answer": "检索增强生成"}
            ] ➡️ 
            history_context = """
            Q:1+1等于几
            A:等于2
            Q:什么是RAG
            A:检索增强生成"""
            '''
            logger.info(f'使用对话历史：{history_context[:50]}')

        '''TODO 使用带历史会话的提示词模板'''
        prompt = RAGPrompts.rag_prompt_withhistory()

        # 1. todo 判断查询类型 (通用知识, 专业咨询) 🔙 查询分类器
        query_category = self.query_classifier.predict_category(query)
        logger.info(f'查询分类结果: {query_category}')
        # 2. todo "通用知识"类查询: 直接调用LLM生成答案 (不检索文档)
        if query_category == '通用知识':
            logger.info("查询为<<通用知识>>, 直接调用LLM生成答案")
            # 2.1 TODO 带历史会话的提示词模板: 通用知识无需上下文, 仅需问题和客服电话、历史会话
            prompt_input = prompt.format(
                history=history_context, # todo 对话历史
                context="",  # 通用知识无需上下文, 仅传入问题
                question=query,  # (用户要查询的)问题
                phone=Config.CUSTOMER_SERVICE_PHONE  # 配置中的客服电话, 用于错误提示
            )
            try: # 2.2 大模型生成答案 并 TODO 流式返回
                for chunk in self.llm.stream(prompt_input):
                    yield chunk.content
                processing_time = time.time() - start_time
                logger.info(f'处理通用知识查询结束, 耗时: {processing_time:.2f}秒, 查询内容: {query}')
            except Exception as e: # 异常时返回包含客服电话的错误提示.
                processing_time = time.time() - start_time
                logger.error(f"直接调用LLM失败: {e}，耗时: {processing_time:.2f}秒")
                logger.error(traceback.format_exc())
                # TODO 整个函数都yield !
                yield f'抱歉, 处理您的通用知识问题时出错, 请联系人工客服: {Config.CUSTOMER_SERVICE_PHONE}'

        # 3. todo "专业咨询"类查询，执行完整RAG流程：策略选择→文档检索→结合上下文生成答案
        else:
            logger.info("查询为<<专业咨询>>, 执行完整RAG流程")
            # 3.1 检索策略 🔙 调用策略选择器确定最合适的检索策略
            strategy = self.strategy_selector.select_strategy(query)
            # 3.2 检索相关文档作为上下文 🔙 调用 _retrieve_entry 函数
            # TODO 对上下文检索所需的关键词进行增强
            if history: # 将历史对话的所有问题 和 当前问题 用\n\n拼接在一起
                history_questions = "\n\n".join([h['question'] for h in history])
                retrieve_query = f"{history_questions}\n\n{query}"
            else: # 如果没有历史对话, 则直接使用当前问题
                retrieve_query = query
            # 检索相关文档作为上下文
            context_docs = self._retrieve_entry(
                retrieve_query, # 用户查询文本
                source_filter=source_filter, # 学科过滤
                strategy=strategy # 检索策略
            )
            # 3.3 构建上下文文本 🔙 将文档列表的page_content属性拼接位字符串.
            if context_docs: # 每个文档占一段, 用两个换行分割
                context = '\n\n'.join([doc.page_content for doc in context_docs])
                logger.info(f'最终上下文文档: {len(context_docs)} 个, 总长度: {len(context)} 字')
            else: # 若未检索到文档, 上下文为空字符串.
                context = ''
                logger.info(f'未检索到文档, 上下文为空字符串')
            # 3.4 提示词
            prompt_input = prompt.format(
                history=history_context, # 对话历史
                context=context,        # 上下文
                question=query,         #  问题
                phone=Config.CUSTOMER_SERVICE_PHONE  # 客服电话
            )
            # 3.5 大模型生成答案，基于上下文、历史对话
            try:
                for chunk in self.llm.stream(prompt_input):
                    yield chunk.content
                processing_time = time.time() - start_time
                logger.info(f'查询处理完成, 耗时: {processing_time:.2f}秒, 查询内容为: {query}')
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"调用LLM失败: {e}，耗时: {processing_time:.2f}秒")
                logger.error(traceback.format_exc())
                # TODO 整个函数都yield !
                yield f'抱歉, 处理您的专业咨询时出错, 请联系人工客服: {Config.CUSTOMER_SERVICE_PHONE}'


    def _test_GENERATE_ANSWER_WITHHISTORY(
            self,
            query=None,
            source_filter='ai',
            history=None
        ):
        print('测试 ➡️ 生成答案:')
        query1 = "课程中都有哪些项目？"
        answer1 = self.generate_answer(query1, source_filter)
        history = [
            {'question': query1, 'answer': answer1}
        ]
        print(f'history: {history}')

        query2 = "最后一章中有哪些？"
        answer_gen = self.generate_answer_withhistory(query2, source_filter, history)
        answer2 = "".join(list(answer_gen))
        history = [
            {'question': query1, 'answer': answer1},
            {'question': query2, 'answer': answer2}
        ]
        print(f'history: {history}')

        query3 = "详细说说"
        answer_gen = self.generate_answer_withhistory(query3, source_filter, history)
        answer3 = "".join(list(answer_gen))
        history = [
            {'question': query1, 'answer': answer1},
            {'question': query2, 'answer': answer2},
            {'question': query3, 'answer': answer3}
        ]
        print(f'history: {history}')


if __name__ == '__main__':
    rag_system = RAGSystem()

    # rag_system._test_RETRIEVE_DIRECTLY() # 直接检索
    # rag_system._test_RETRIEVE_BACKTRACKING() # 回溯检索
    # rag_system._test_RETRIEVE_SUBQUERIES() # 子查询检索
    # rag_system._test_RETRIEVE_HYDE() # hyde假设答案再向量检索

    # rag_system._test_RETRIEVE_ENTRY() # 测试检索统一入口：根据策略进行向量检索
    # rag_system._test_GENERATE_ANSWER()  # 测试RAG主逻辑

    # 调用示例
    answer = rag_system.generate_answer(query="地球为什么是圆的?", source_filter='ai')
    print(answer) # 通用知识
    answer = rag_system.generate_answer(query="黑马AI课程大纲有哪些内容?", source_filter='ai')
    print(answer) # 专业咨询
    answer = rag_system.generate_answer(query="人工智能就业课的涉及需求是什么?", source_filter='ai')
    print(answer) # 专业咨询

    # 测试带历史会话的RAG主逻辑
    rag_system._test_GENERATE_ANSWER_WITHHISTORY()
