
import traceback
import numpy as np
from rank_bm25 import BM25Okapi
from mysql_qa.utils.mysql_client import MysqlClient
from mysql_qa.utils.redis_client import RedisClient
from mysql_qa.utils.preprocess import preprocess_text
from base import logger


class Bm25SoftmaxSearch:
    """BM25+softmax检索"""
    def __init__(
        self,
        mysql_client=MysqlClient(), # mysql客户端
        redis_client=RedisClient(), # redis客户端
    ):
        self.logger = logger # 日志
        self.mysql_client = mysql_client
        self.redis_client = redis_client

        # 原始问题列表 [问题1, 问题2,...]
        self.original_questions = None
        # 分词后的问题列表 [[词1，词2...], [词1, 词2...]...]
        self.tokenized_questions = None
        # bm25模型
        self.bm25 = None
        # 加载数据
        self._load_data()

    # todo ===== 构造bm25模型，构造问题列表，构造问题分词列表 =====
    def _load_data(self):
        """ 构造bm25对象，加载数据到内存中：
        1. 判断redis中是否有数据，两key但凡有一个没数据
            就从mysql中读取数据
                mysql没数据就终止函数
                有数据就缓存到redis中
        2. 构造bm25对象"""

        # redis中 问题列表 的key名 [问题1, 问题2,...]
        original_key = 'edurag:qa_original_questions'
        # redis中 问题分词列表 的key名 [[词1,词2,..],...]
        tokenized_key = 'edurag:qa_tokenized_questions' # 问题分词之后的key

        # 从redis中批量获取原始问题列表
        self.original_questions = self.redis_client.get_data(original_key)
        # 从redis中批量获取问题分词列表
        self.tokenized_questions = self.redis_client.get_data(tokenized_key)

        # 如果redis中没有原始问题列表，或 没有 问题分词列表，就执行加载逻辑
        if ((not self.original_questions)
                or (not self.tokenized_questions)):
            # 👉从mysql中获取原始问题列表 [(问题1,), (问题2,), ...]
            self.original_questions = [
                question_str # ➡️构造成 [问题1, 问题2, ...]
                for (question_str,) in self.mysql_client.fetch_all_questions()
            ]
            # 💡调试：看一下 问题分词列表 的情况
            # print(len(self.original_questions),
            #       self.original_questions[0])

            # 如果mysql中没有数据，就终止函数
            if not self.original_questions:
                self.logger.error(f"没有原始问题数据，请先插入数据！")
                return

            # 👉构造问题分词列表 [[词1，词2...], [词1, 词2...]...]
            self.tokenized_questions = [
                # todo 分词，不要生成器，直接返回字符串构成的列表
                preprocess_text(q, is_generator=False)
                for q in self.original_questions
            ] # ➡️ [问题1, 问题2, ...]
            # 调试：看一下 问题分词列表 的情况
            # print(len(self.tokenized_questions),
            #       self.tokenized_questions[0])

            # todo 分词后的问题列表存入redis中
            self.redis_client.set_data(tokenized_key,
                                       self.tokenized_questions)
            self.logger.info(f"成功加载到redis：{len(self.tokenized_questions)}")

            # todo 问题列表存入redis中
            self.redis_client.set_data(original_key,
                                       self.original_questions)
            self.logger.info(f"成功加载到redis：{len(self.original_questions)}")

        # 初始化bm25算法模型: 传入 问题分词列表
        self.bm25 = BM25Okapi(self.tokenized_questions)
        self.logger.info(f"BM25模型初始化成功")

    @staticmethod
    def _test_LOAD_DATA(question):
        # 测试 _load_data函数
        bm25search_obj = Bm25SoftmaxSearch()
        print("🔍数据查询功能 检查开始")
        print("query:", question)
        print("redis:", bm25search_obj.redis_client.get_answer(question))
        print("mysql:", bm25search_obj.mysql_client.fetch_answer(question))
        print("🔍数据查询功能 检查结束")
        print("🔍检查 问题列表 和 问题分词列表 中的第一条数据")
        print(bm25search_obj.original_questions[0])
        print('分词：', bm25search_obj.tokenized_questions[0])

    # todo ===== softmax: 削峰归一 =====
    def _softmax(self, scores):
        """对bm25模型返回的分数进行softmax归一化处理：
        0. BM25算的分数 → 转换成0~1之间的概率分布，所有结果加起来=1（归一化）
        1. 所有数减最大值，做e的指数，计算结果，归一：每个结果分别➗️所有结果的和
            np.exp([..]-其中最大值) / np.sum(np.exp([..]-其中最大值))
        2. 参数scores 是bm25算法算出来的一组数字
        3. 函数API说明
            3.1. np.max(数组x) 找出数组中最大的那个数
                np.max([1,3,5]) == 5
            3.2. np.exp(数组x) 自然常数e的x次方（e≈2.718）
                np.exp([1,3,5]) == [e的1次方, e的3次方, e的5次方]
            3.3. np.sum(数组x) 对x所有元素求和
                np.sum([1,3,5]) == 9
        4. 减去最大值：防计算爆炸溢出
            4.1 假设 scores = [1000, 1001, 1002]
            4.2 np.exp(scores)  # 计算 e^1000, e^1001, e^1002
                结果是：天文数字！电脑直接显示：[inf, inf, inf]
            4.3 减去最大值 scores - np.max(scores)
                变成：[-2, -1, 0]，数字就变小了，就不会计算爆炸内存溢出
        """
        exp_scores = np.exp(scores-np.max(scores))
        return exp_scores / np.sum(exp_scores)

    @staticmethod # 演示函数
    def _test_SOFTMAX(scores=[2,4,6]):
        # todo 步骤拆解：SOFTMAX的原理
        print('演示➡️ softmax的过程')
        print("0. 原始分数 scores：\n\t", scores) # [2,4,6]

        max_score = np.max(scores)
        print("1. 最大值 np.max：\n\t", max_score) # 6

        scores_shifted = scores - max_score
        print("2. 减去最大值后：\n\t", scores_shifted) # [-4 -2  0]

        exp_scores = np.exp(scores_shifted) # e的多少次方
        print("3. 指数 np.exp 后：\n\t", exp_scores) # [0.01831564 0.13533528 1.]

        sum_exp = np.sum(exp_scores)
        print("4. 指数总和：\n\t", sum_exp) # 1.1536509221253468

        result = exp_scores / sum_exp
        print("5. 最终 softmax 概率得分：\n\t", result) # [0.01587624 0.11731043 0.86681333]
        print("6. 概率总和为1，即所有可能性的概率加起来是100%➡️", np.sum(result)) # 1.0
        print("【结论】归一化：把一组数字转换成【总和为1】的概率分布：")
        #  [2, 4, 6] ➡️ [0.01587624 0.11731043 0.86681333] ➡️ 总和为1
        print('\t', scores, '➡️', result, '➡️', '总和为1')
        return result

    @staticmethod # 演示函数
    def _test_SCORES_MAX(scores=[1000, 1001, 1002]):
        # todo 计算爆炸的演示
        # （减去最大值的作用：防止计算爆炸）
        # 计算 e^1000, e^1001, e^1002 ==> [inf inf inf]
        print('演示➡️ 计算爆炸的结果')
        print(np.exp(scores))

    # todo ========== 检索 ==========
    def search(self, query, threshold=0.85):
        """
        参数 query : 问题字符串
        参数 threshold=0.85 : 设置的相似度阈值
            >=0.85 相似度极高，判定命中
        """
        if not query or not isinstance(query, str):
            self.logger.error(f"查询参数错误：{query}: {type(query)}")
            return None, False

        # redis缓存拿答案
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            self.logger.info(f"查询缓存成功：{query}")
            return cached_answer, False

        # 如果redis中没有答案，就搜索
        try:
            # TODO 【核心逻辑】
            # todo 1. 将问题进行分词，返回分词列表
            # mysql_qa.utils.preprocess.preprocess_text
            query_tokens = preprocess_text(query, is_generator=False)
            # todo 2. 计算bm25得分：分词列表 → 已加载的BM25对象 → 分数数组
            scores = self.bm25.get_scores(query_tokens)
            # # 💡调试
            # print(query_tokens, scores)
            # todo 3. 再计算softmax归一化处理：bm25得分数组 → 概率数组
            softmax_scores = self._softmax(scores)
            # todo 4. 获取最高分（概率最大）的索引下标，基于下标拿最高分
            best_idx = softmax_scores.argmax() # 数组→最大值的索引
            best_score = softmax_scores[best_idx] # 最高分
            logger.info(f"问题：{query}，最高分：{best_score}")

            # todo 5 最高分超过阈值0.85, 匹配度最高
            if best_score >= threshold:
                # todo 5.1. 获取原始问题
                original_question = self.original_questions[best_idx]

                # todo 5.2. 获取最相似的问题的答案
                # 从redis中获取最相似的问题的答案
                redis_result = self.redis_client.get_answer(
                    original_question
                )
                # 如果redis中有
                if redis_result:
                    answer = redis_result
                    self.logger.info(f"从redis中查询成功，答案长度：{len(answer)}")
                else: # redis中没有，再从mysql中查找答案
                    answer = self.mysql_client.fetch_answer(
                        original_question
                    )
                    self.logger.info(f"从mysql中查询成功，答案长度：{len(answer)}")

                # 无论redis中有没有，都再缓存一下【edu_rag:answer:问题 :答案】，并返回
                if answer: # 缓存到redis
                    self.redis_client.set_answer(query, answer)
                    self.logger.info(f"搜索成功，softmax相似度：{best_score:.3f}")
                    # 返回答案
                    return answer, False

            # todo 6. 最高分低于超过阈值0.85, 返回 None, True
            self.logger.info(f"没有可靠答案，softmax相似度最高分：{best_score:.3f}")
            return None, True # 无答案

        # b25-softmax计算 或 搜索失败
        except Exception as e:
            self.logger.error(f"❌搜索失败：{traceback.format_exc()}")
            return None, True


if __name__ == '__main__':
    import random

    # print('测试➡️ _load_data 加载数据')
    # q_list = [
    #     '关联子查询的执行顺序是什么',
    #     '归一化算法问题',
    #     'os.path.dirname() 方法有什么作用？',
    #     '函数和类区别',
    #     '什么是张量'
    # ]
    # q = random.choice(q_list)
    # Bm25SoftmaxSearch._test_LOAD_DATA(q)

    # print('演示➡️ softmax的过程')
    # Bm25SoftmaxSearch._test_SOFTMAX([2,4,6])
    # print('演示➡️ 计算爆炸的结果')
    # Bm25SoftmaxSearch._test_SCORES_MAX([1000, 1001, 1002])

    print('测试➡️ search核心逻辑')
    q_list = [
        '子查询的执行顺序是什么', '算法问题：归一化',
        'os.path.dirname()',
        '类和函数区别', '张量是什么'
    ]
    q = random.choice(q_list)
    result = Bm25SoftmaxSearch().search(q)
    print(result)

