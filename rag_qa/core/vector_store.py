"""
rag_qa 核对代码目录
向量存储检索
"""

import sys, torch
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值md5
import hashlib
# 导入 Milvus 相关类
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入BGE-M3模型嵌入函数，用于生成文档和查询的向量
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入CrossEncoder，接入重排模型，用于重排序
from sentence_transformers import CrossEncoder

from base import Config, logger, get_abs_path
# 导入 process_documents文档加载、分割处理函数
from rag_qa.core.process_documents import process_documents


class VectorStore:

    def __init__(self,
                 host= Config.MILVUS_HOST,
                 port= Config.MILVUS_PORT,
                 database= Config.MILVUS_DATABASE,
                 collection_name= Config.MILVUS_COLLECTION):
        self.logger = logger

        # milvus客户端
        self.client = MilvusClient(
            uri=f'http://{host}:{port}',
            db_name=database, # 指定要使用的数据库名
        )
        self.collection_name = collection_name # 集合名称

        # 确定当前环境是用cpu还是gpu 如果cuda可用，则使用cuda，否则使用cpu
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = Config.DEVICE # 使用配置的设备
        self.logger.info(f'使用设备：【{self.device}】')

        # todo 初始化 BGE-Reranker-large 模型，用于重排序检索结果
        self.reranker = CrossEncoder(
            Config.RERANK_MODEL_PATH,
            device=self.device, # 使用CPU or GPU
            # max_length=512, # 最大长度 bge-reranker-large支持8192
        )

        # todo 初始化 BGE-M3 模型的嵌入函数，用于生成文档和查询的向量
        self.embedding_function = BGEM3EmbeddingFunction(
            Config.EMBEDDING_MODEL_PATH,
            # self.device=='cuda' ➡️ True or False
            # True 有显卡（cuda）就开fp16半精度16位浮点数省显存、速度快
            # False 没显卡（cpu）就关fp16，使用fp32单精度32位浮点数
            # 到底是fp16还是fp32，稠密向量字段的类型要一致！
            use_fp16=(self.device=='cuda'),
            device=self.device, # 使用CPU or GPU(cuda)
        )

        # 获取bge-m3的维度是1024（稠密向量维度）
        self.dense_dim = self.embedding_function.dim['dense']

        # 创建、加载向量库集合
        self._create_or_load_collection()

    def _test_INIT(self):
        """测试初始化"""
        print("测试 ➡️ 初始化功能")
        print(self.client.list_collections())
        print(self.device)
        print(self.dense_dim)

    def _test_BGE_M3(self):
        print("测试 ➡️ BGE-M3模型批量产生向量")
        texts = [
            "需要劳动合同、医疗诊断证明、工伤认定申请表",
            "需要身份证、户口本、结婚证",
            "需要租房合同、房东身份证"
        ]
        embeddings = self.embedding_function(texts)
        # print('-' * 20, '输出的向量的细节➡️ ：')
        print('1. embeddings.kesy:', embeddings.keys())
        print('2. 稠密向量数量:', len(embeddings['dense']))
        print('3. 一个稠密向量:', embeddings['dense'][0])
        print('4. 稀疏向量矩阵:', embeddings['sparse'])
        print('5. 第一个稀疏向量的COO格式:', embeddings['sparse'][0])
        print('6. 第一个稀疏向量的CSR格式:', embeddings['sparse'][0].tocsr())
        # print('6. 第一个稀疏向量的CSR格式:', embeddings['sparse'][[0]])
        # print('7. 第1个稀疏向量的列号列表:', embeddings["sparse"][[0]].indices)
        print('7. 第1个稀疏向量的列号列表:', embeddings["sparse"][0].tocsr().indices)
        # print('8. 第1个稀疏向量的权重列表:', embeddings["sparse"][[0]].data)
        print('8. 第1个稀疏向量的权重列表:', embeddings["sparse"][0].tocsr().data)
        # print('-'*20)
        '''【输出向量的细节】
        1. embeddings.kesy: dict_keys(['dense', 'sparse'])
        2. 稠密向量数量: 86
        3. 一个稠密向量: [-0.0509  -0.02028 -0.02322 ... -0.01363  0.0667  -0.0329 ]（1024个）
        4. 稀疏向量矩阵: <Compressed Sparse Row sparse array of dtype 'float64'
                with 5947 stored elements and shape (86, 250002)>
            Coords	Values
            (0, 4)	0.7548828125
            ... 共5947个
        # 说明：
            # 86句话（86个子块）里，一共有5947个非零权重，其他全是0，不存、不占空间
            # 共250002列=BGE-M3 稀疏向量固定维度（最大词表编号），只有5947个列非0
            # (0, 4)	0.7548828125  →  行号，列号  权重（越大越重要）
            # 行列定位的是一个词（token），权重就是这个词在这个句子（子块）的重要性得分
            # 一行就是分割后的一个子块，从0-85一共86行（86个子块）

        5. 第一个稀疏向量的COO格式: <COOrdinate sparse array of dtype 'float64'
                with 17 stored elements and shape (250002,)>
            Coords	Values
            (7081,)	0.115966796875
            ...共17个
        # 说明：
            # 第一句话（0号子块）里，一共有96个非零权重，其他全是0，不存、不占空间
            # 共250002列=BGE-M3 稀疏向量固定维度（最大词表编号），只有96个列非0
            # (4,)	0.7548828125 →  列号  权重（这个词在这个句子（子块）的重要性得分，越大越重要）
        6. 第一个稀疏向量的CSR格式: 长得跟 COO格式 一样！
        7. 第1个稀疏向量的列号列表: [  7081   7200  11394  17866  21001  28748  29606  35073  38274  75092
          81492 118484 121318 136229 146563 158204 203593]
        8. 第1个稀疏向量的权重列表: [0.1159668  0.2590332  0.50537109 0.20959473 0.0690918  0.09832764
         0.17053223 0.16992188 0.1114502  0.06787109 0.15771484 0.27587891
         0.38037109 0.37841797 0.08074951 0.06591797 0.24084473]
        '''

    def _test_BGE_RERANKER_LARGE(self):
        print("测试 ➡️ BGE-RERANKER-LARGE模型")
        """测试BGE-RERANKER-LARGE模型"""
        # 你的问题
        query = "工伤认定需要什么材料？"
        # 候选文档
        docs = [
            "需要劳动合同、医疗诊断证明、工伤认定申请表",
            "需要身份证、户口本、结婚证",
            "需要租房合同、房东身份证"
        ]
        # 构造嵌套列表，每个元素是一个列表，包含问题、文档，用于模型输入
        inputs = [[query, doc] for doc in docs]
        # 进模型计算分数
        scores = self.reranker.predict(inputs)
        # 按分数从高到低排序
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        # 输出结果
        print("重排序结果：")
        for doc, score in reranked:
            print(f"分数：{score:.2f} | 内容：{doc}")

    # =========创建集合=========
    def _create_or_load_collection(self):
        '''创建、加载向量库集合'''
        # todo 1. 检查指定集合是否已存在
        if not self.client.has_collection(self.collection_name):
            # 2.1 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(
                auto_id=False, # 禁用自动 ID
                enable_dynamic_field=True # 启用动态字段
            )

            # todo 2.2 添加字段
            # 主键 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 标量 文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            # TODO 稠密向量类型由self.device决定：GPU用FLOAT16_VECTOR，CPU用32位的FLOAT_VECTOR
            dense_type = DataType.FLOAT16_VECTOR if self.device=='cuda' else DataType.FLOAT_VECTOR
            schema.add_field(field_name="dense_vector", datatype=dense_type, dim=self.dense_dim)
            # 稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 标量 父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 标量 父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 标量 学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 标量 时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # todo 2.3 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)，
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT", # 先聚类，再暴力搜索
                metric_type="IP",
                params={"nlist": 128} # 聚类分为128组
            )
            # 稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                # 稀疏向量倒排索引，词出现在哪个文档
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )
            """参数drop_ratio_build: 0.2 建索引时扔掉20%最没用的特征
            1. 一堆文章（向量），每篇有 1000 个词（维度）：
                大部分词是停用词、不重要的小权重词（比如：的、是、和）。
                建索引时，扔200个词权重小的词
            2. drop_ratio_build一般设为0.1~0.2，0.2是最佳实践
                【永久丢失】一点精度，省空间、提速       
            """

            # todo 2.4 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema, # 字段定义
                index_params=index_params # 索引
            )
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")

        else: # 如果集合已存在
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")

        # todo 3. 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    def _test_CREATE_OR_LOAD_COLLECTION(self):
        print("测试 ➡️ 创建、加载向量库集合")
        self._create_or_load_collection()
        # 查看集合的字段信息
        print(self.client.describe_collection(self.collection_name))

        # 查看所有的索引名
        index_list = self.client.list_indexes(self.collection_name)
        print("集合中的索引名称列表：", index_list)
        for index in index_list:
            print('*' * 30)
            print("索引名称：", index)
            # 查看具体某个索引的详细配置
            index_detail = self.client.describe_index(
                collection_name=self.collection_name,
                index_name=index
            )
            print(f"\n【{index}】索引的详细配置：", index_detail)

    # =========向集合中添加数据=========
    def add_documents(self, documents:[Document]):
        """向集合中添加数据"""
        # 1. 提取所有文档的内容列表 [str]:
        texts = [doc.page_content for doc in documents]

        # 2. 使用BGE-M3模型将文档内容转换为向量
        # 输入: texts文本列表 [str]
        # 输出: 字典, 包含: dense(稠密向量列表), sparse(稀疏向量)
        embeddings = self.embedding_function(texts)

        # 3.初始化空列表: 存储待插入的数据 [dict]
        data_list = []
        # 4. 遍历每个文档, 组装插入数据
        # i: 当前文档下标; doc: 当前Document文档对象
        for i, doc in enumerate(documents):
            # 4.1 基于文档内容生成md5指纹str，作为文档的唯一ID
            """文本一模一样→MD5相同，差一点→MD5就完全不同，32 位字符串
            理论上不可逆（文本→MD5✅️，MD5→文本❌️）
            除了md5以外 类似的有base64 sha1 sha256 sha512"""
            text_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            # print(f'text_hash: {text_hash}')
            # 例如: text_hash: 54027a0ef66b897b4967ac4fe2dcea6f

            # 4.2 构造稀疏向量
            sparse_vector = {} # 一个文档的稀疏向量字典
            # 4.2.1 获取第i个文档的稀疏向量的CSR格式
            # row = embeddings["sparse"][[i]]
            row = embeddings["sparse"][i].tocsr()
            # 4.2.2 分别获取一个（子块）稀疏向量 每个词的列号和权重.
            #   (0, 121318)	0.765625  →  行号,列号 权重
            # 获取 子块的列号（词编号） 列表, 例如: [10, 25, 50]
            indices = row.indices
            # 获取 子块的词权重分 列表, 例如: [0.1, 0.2, 0.3]
            values = row.data
            # 4.2.3 组装稀疏向量: {列号: 权重} → {10:0.1, 25:0.2, 50:0.3}
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value

            # 4.3 组装单条文档数据(字段要与Schema完全一致)
            data_list.append({
                "id": text_hash,                        # 唯一ID(MD5哈希)
                'text': doc.page_content,               # 文档内容
                "dense_vector": embeddings["dense"][i], # 稠密向量(BGE-M3生成)
                "sparse_vector": sparse_vector,         # 稀疏向量(BGE-M3生成)
                # 从元数据doc.metadata字典中获取
                "parent_id": doc.metadata["parent_id"], # 父文档ID
                "parent_content": doc.metadata["parent_content"],   # 父文档内容
                "source": doc.metadata.get("source", "unknown"),    # 文档来源
                "timestamp": doc.metadata.get("timestamp", "unknown"),  # 文档时间戳
            })

        # 5. 如果data_list非空列表，就插入数据 (避免无效操作)
        if data_list:
            # upsert(update+insert)插入数据，存在就更新，不存在就插入
            # 搭配每个文档的md5指纹，实现 自动去重 + 增量更新
            self.client.upsert(self.collection_name, data_list)
            logger.info(f"已插入 {len(data_list)} 条数据到集合 {self.collection_name}")
        # 6. 加载集合
        self.client.load_collection(self.collection_name)
        logger.info(f"已加载集合 {self.collection_name}")

    def _test_ADD_DOCUMENTS(self):
        print("测试 ➡️ 文档入库")
        # 加载并分割文档，获取所有子块构成的列表 [Document]
        documents = process_documents(directory_path=Config.DATA_DIR)
        print('子块文档数：', len(documents))
        # 添加数据
        self.add_documents(documents)

    # ========混合检索（粗排）+ 重排序（精排）=========
    def hybrid_search_with_rerank(
            self,
            query,
            k=Config.RETRIEVAL_K,
            source_filter=None
        ):
        """该函数用于执行混合检索(稠密 + 稀疏向量) + 结果重排序, 返回精准父文档
        :param query: 用户查询文本, 例如: AI学科的课程内容是什么?
        :param k: 混合检索返回的Top-K子块数量, 默认从conf中读取
        :param source_filter: 学科过滤条件, 例如: "ai", 仅检索ai学科的文档, None:不过滤
        :return: 重排序后的Top-M父文档列表
        """
        # todo 1. query向量化
        # 1. 生成查询文本的向量 BGE-M3模型.
        query_embedding = self.embedding_function([query])
        # 2. 提取查询的稠密向量.
        dense_query_vector = query_embedding["dense"][0]
        # print(f'dense_query_vector: {dense_query_vector}, {dense_query_vector.shape}')

        # 3. 处理查询的稀疏向量: 转换为Milvus支持的字典格式.
        sparse_query_vector = {}
        # 3.1 获取查询的稀疏向量行.(仅1个查询, 获取第0行即可)
        row = query_embedding["sparse"][0].tocsr()
        # 3.2 获取索引和权重.
        indices = row.indices   # 索引列表.
        values = row.data       # 权重列表.
        # 3.3 组装稀疏向量字段: 将索引与权重配对, 即: {索引: 权重}
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value

        # 4. 构造expr过滤表达式: "" 或者 "source == 'ai'"【内层用单引号】按学科过滤
        filter_expr = f"source == '{source_filter}'" if source_filter else ""

        # 5.构建稠密向量检索请求: 定义稠密向量的检索参数.
        dense_request = AnnSearchRequest(
            data = [dense_query_vector],    # [query稠密向量]
            anns_field="dense_vector",      # 稠密向量字段名称
            # 检索参数 "nprobe": 10 ➡️ 查询10个中心点与query向量最相似的聚类（IVF索引）
            param = {"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,            # 检索的Top-K子块数量
            expr=filter_expr,   # 应用过滤表达式(按学科过滤)    # "" 或者 "source == 'ai'"
        )

        # 6. 构建稀疏向量检索请求: 构建稀疏向量检索参数.
        sparse_request = AnnSearchRequest(
            data = [sparse_query_vector],   # [query稀疏向量]
            anns_field="sparse_vector",     # 稀疏向量字段名称
            param = {"metric_type": "IP", "params": {}},  # 检索参数(IP 内积相似度)
            limit=k,            # 检索的Top-K子块数量
            expr=filter_expr,   # 应用过滤表达式(按学科过滤)
        )

        # 7. 创建加权排序器.
        # 稠密向量-语义权重(1.0), 稀疏向量-关键词权重(0.7)
        # ranker = WeightedRanker(1.0, 0.7)

        # 8. 执行混合检索: 同时进行稠密+稀疏向量检索
        # 加权排序器融合结果, 返回Top-K个结果Hit构成的[Hit]（[dict]）
        # 一个命中（召回）结果Hit本质是一个字典dict
        results = self.client.hybrid_search(
            collection_name=self.collection_name,   # 目标集合(Milvus表)
            reqs=[dense_request, sparse_request],   # 混合检索请求列表(稠密 + 稀疏)
            # 稠密向量-语义权重(1.0), 稀疏向量-关键词权重(0.7)
            ranker=WeightedRanker(1.0, 0.7),      # 加权排序器
            limit=k,            # 返回最相似的k个结果 Top-k
            # 返回结果字段: 文本内容, 父文档ID, 父文档内容, 来源, 时间戳
            output_fields= ["text", 'parent_id', 'parent_content', 'source', 'timestamp']
        )[0] # hybrid_search()返回嵌套列表[list], 取第0个元素list(因为只有1个查询)
        # print(f'results: {results}')        # 查看检索结果 [Hit, Hit, Hit]
        # print(f'results: {type(results)}')  # 查看结果类型 <class 'list'>
        # print(f'results: {len(results)}')   # 查看结果数量 3
        # print(f'results: {results[0]["entity"]}')   # 查看第1个结果中的所有字段

        # TODO 封装函数：将检索的一个hit命中结果构造Document对象
        # def _doc_from_hit(hit):
        #     hit_entity = hit["entity"]
        #     return Document(
        #         page_content=hit_entity["text"],
        #         metadata={
        #             "parent_id": hit_entity["parent_id"],
        #             "parent_content": hit_entity["parent_content"],
        #             "source": hit_entity["source"],
        #             "timestamp": hit_entity["timestamp"]
        #         }
        #     )
        # TODO 改成静态方法, 放本函数外边
        # 静态方法在类中用类名调用比self调用，能提升微乎其微的性能

        # 9.将检索结果转换为Document对象列表, 目的: 统一格式, 方便后续处理.
        topK_documents = [VectorStore._doc_from_hit(hit) for hit in results]
        # print(f'topK_documents: {topK_documents}') → [Document]

        # TODO 封装函数：从子块列表中提取全部父文档的文本内容，去重、构造并返回[Document]
        # def _get_unique_parent_docs(topK_documents):
        #     parent_ids = set()  # 放parent_id的集合，自动去重
        #     unique_docs = []  # 存储去重后的父文档的list
        #
        #     for doc in topK_documents:
        #         parent_id = doc.metadata.get("parent_id")
        #
        #         # 如果 parent_id 存在且不在集合中，则加入结果集
        #         if parent_id and (parent_id not in parent_ids):
        #             parent_ids.add(parent_id)
        #             # 获取子文档的parent_content（父块的完整本文内容）
        #             parent_content = doc.metadata.get("parent_content")
        #             # 构造父文档
        #             parent_doc = Document(
        #                 page_content=parent_content,
        #                 metadata=doc.metadata
        #             )
        #             # 父文档加入结果list
        #             unique_docs.append(parent_doc)
        #     # 返回去重后的父文档列表 [Document]
        #     return unique_docs
        # TODO 改成静态方法, 放本函数外边

        # 10. 从子块列表中提取全部父文档的文本内容，去重、构造并返回[Document].
        parent_docs = VectorStore._get_unique_parent_docs(topK_documents)
        # print(f'parent_docs: {parent_docs}') # [Document]

        # 11. 当 父文档数量 < 最终结果要求的数量，跳过重排序直接返回.
        if len(parent_docs) < Config.CANDIDATE_M:
            return parent_docs[:Config.CANDIDATE_M] # 列表切片

        # 12. 父文档数量 >= 最终结果要求的数量, 执行重排序(提升最终结果的相关性)
        # 12.1 构建"查询-文档"配对列表
        pairs = [[query, doc.page_content] for doc in parent_docs]
        # 12.2 todo 重排：计算相关性得分, 得分越高, 相关性越强
        scores = self.reranker.predict(pairs)
        # print(f'scores: {scores}')
        # 12.3 按得分进行reverse降序排序
        ranked_parent_docs = [
            doc for _, doc in
                sorted(zip(scores, parent_docs), reverse=True)
            ]

        # 14. 返回重排后的Top-M个文档.
        return ranked_parent_docs[:Config.CANDIDATE_M]

    @staticmethod
    def _doc_from_hit(hit):
        '''将检索的一个hit命中结果构造Document对象'''
        hit_entity = hit["entity"]
        return Document(
            page_content=hit_entity["text"],
            metadata={
                "parent_id": hit_entity["parent_id"],
                "parent_content": hit_entity["parent_content"],
                "source": hit_entity["source"],
                "timestamp": hit_entity["timestamp"]
            }
        )

    @staticmethod
    def _get_unique_parent_docs(topK_documents):
        """从子块列表中提取全部父文档的文本内容，
        去重、构造并返回[Document]"""
        parent_ids = set()  # 放parent_id的集合，自动去重
        unique_docs = []  # 存储去重后的父文档的list

        for doc in topK_documents:
            parent_id = doc.metadata.get("parent_id")

            # 如果 parent_id 存在且不在集合中，则加入结果集
            if parent_id and (parent_id not in parent_ids):
                parent_ids.add(parent_id)
                # 获取子文档的parent_content（父块的完整本文内容）
                parent_content = doc.metadata.get("parent_content")
                # 构造父文档
                parent_doc = Document(
                    page_content=parent_content,
                    metadata=doc.metadata
                )
                # 父文档加入结果list
                unique_docs.append(parent_doc)
        # 返回去重后的父文档列表 [Document]
        return unique_docs

    def _test_HYBRID_SEARCH_WITH_RERANK(self, query):
        print(f'测试 ➡️ 混合检索 + 重排序: {query}')
        ret = self.hybrid_search_with_rerank(query)
        print(f'返回结果数量: {len(ret)}')
        for doc in ret:
            print(doc.__dict__.keys())
            print(doc.metadata.keys())
            print(doc.page_content)



if __name__ == '__main__':

    # vector_store = VectorStore()
    # vector_store._test_INIT()
    #
    # vector_store = VectorStore()
    # vector_store._test_BGE_M3()
    #
    # vector_store = VectorStore()
    # vector_store._test_BGE_RERANKER_LARGE()
    #
    # vector_store = VectorStore()
    # vector_store._test_CREATE_OR_LOAD_COLLECTION()

    # vector_store = VectorStore()
    # vector_store._test_ADD_DOCUMENTS()

    print('测试 ➡️ 混合检索')
    vector_store = VectorStore()
    # vector_store._test_HYBRID_SEARCH_WITH_RERANK('大模型课程介绍')
    # vector_store._test_HYBRID_SEARCH_WITH_RERANK('语言模型评估指标')
    # vector_store._test_HYBRID_SEARCH_WITH_RERANK('AI学科的课程内容是什么？')
    # vector_store._test_HYBRID_SEARCH_WITH_RERANK('黑马AI课程大纲有哪些主要内容?')
    vector_store._test_HYBRID_SEARCH_WITH_RERANK('蔡徐坤练习时长两年半，王者归来！大家别感冒！')

