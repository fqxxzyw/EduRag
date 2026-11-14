"""
rag_qa 核对代码目录
意图识别
"""
import os, json
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
# TrainingArguments: 负责定义训练的各种参数和配置
#   比如: 训练多久, 用多大批次, 保存到哪等
#   相当于: 训练任务的说明书. 它是Trainer的"操作指南"
# Trainer: 负责实际执行训练任务
#   用模型, 数据, TrainingArguments的配置 跑训练
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split # 划分训练集、验证集
from sklearn.metrics import classification_report, confusion_matrix

from base import logger, Config


# TODO 定义QueryClassfier类:
#  封装BERT查询分类的完整流程
#  1. 加载预训练、微调模型
#  2. 处理数据
#     - 获取训练集、测试集
#     - 预处理 文本转张量，标签转[1,0]
#     - 组装Dataset数据集
#  3. 构建训练参数
#  4. 构建训练对象
#  5. 微调训练
#  6. 保存模型
#  7. 评估模型
#  8. 使用模型进行预测，实现检索的意图识别：
#     → 能够把输入进行分类：{"通用知识": 0, "专业咨询": 1}
class QueryClassifier:
    # todo 0. --------初始化，加载预训练、微调模型--------
    def __init__(self,
                 new_model_path=Config.BERT_QUERY_PATH):
        """初始化方法:
            配置模型路径, 记载分词器, 选择设备, 定义标签映射
        :param new_model_path: 模型保存的路径.
        """
        self.logger = logger
        # 存储模型路径: 用于后续保存模型
        self.new_model_path = new_model_path
        # 加载分词器, 指定BERT-BASE-CHINESE路径
        self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_PATH, )
        # 预训练模型对象 在_load_model()方法中定义
        self.model = None
        # 训练器对象 在_get_trainer方法中定义
        self.trainer = None
        # 选择模型运行设备. 优先使用GPU, 如果没有GPU, 则使用CPU.
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = Config.DEVICE
        logger.info(f'模型运行设备: {self.device}')
        # 定义标签映射, 将文本标签 转为 数字, 便于模型训练.
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        # 7. 加载模型: 初始化时, 自动调用_load_model(), 确保模型可用.
        self._load_model()

    def _load_model(self):
        """加载模型：从指定路径加载 微调模型, 若不存在则加载 预训练模型
        """
        # 1. 判断模型路径是否存在(即: 是否有已训练的模型)
        if os.path.exists(self.new_model_path):
            # 加载已微调的模型
            self.model = BertForSequenceClassification.from_pretrained(self.new_model_path)
            self.model.to(self.device) # 让模型使用指定设备. cpu 或 cuda
            logger.info(f'加载微调模型成功: {self.new_model_path}')
        else: # 2. 若模型路径不存在, 加载bert-base-chinese预训练模型
            self.model = BertForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=Config.BERT_PATH,
                num_labels=len(self.label_map) # 分类映射字典 {"通用知识": 0, "专业咨询": 1}
            )
            self.model.to(self.device) # 让模型使用指定设备. cpu 或 cuda
            logger.info(f'加载预训练模型成功: {Config.BERT_PATH}')

    def _test_INIT(self):
        print('测试 ➡️ 初始化功能：')
        print('使用训练的设备是：', self.device)
        print('预训练模型路径:', Config.BERT_PATH)
        print('新模型保存路径:', self.new_model_path)
        print('当前模型状态：', self.model)
        # 在输出的最后能看到 in_features=768 维, out_features=2 分两类


    # todo 1. --------数据预处理1. 获取 训练集、测试集--------
    def _get_train_data_split(self,
                              data_file=Config.BERT_TRAIN_DATA,
                              random_state=42, # 随机种子
                              ):
        """获取训练集、测试集
        """
        # 检查数据集文件是否存在 不存在直接报错
        if not os.path.exists(data_file):
            logger.error(f'数据库文件 {data_file} 不存在!')
            raise FileNotFoundError(f'数据集文件 {data_file} 不存在!')

        # 加载数据集: 从Json文件中读取查询文本和对应标签. 转成 [dict]
        with open(data_file, 'r', encoding='utf-8') as f:
            # 按行解析Json数据, 每行一条样本转成dict, 保存成列表 [dict]
            data = [json.loads(value) for value in f.readlines()]

        # 分别提取文本和标签
        # {"query": "1024乘以768等于多少？", "label": "通用知识"}
        texts = [item['query'] for item in data] # 特征文本
        labels = [item['label'] for item in data] # 分类标签

        # 划分训练集和验证集. 80%作为训练集, 20%作为验证集.
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=random_state
        )
        '''train_test_split：数据集划分工具 
            sklearn.model_selection.train_test_split
        参数：
            texts, labels：一一对应的特征输入文本[str] 和 分类标签[str]
            test_size=0.2：20% 做验证集，80% 做训练集
            random_state=42：固定随机种子，每次划分结果一致，方便复现
        输出：
            train_texts / train_labels：训练集文本 + 标签（用于模型训练）
            val_texts / val_labels：验证集文本 + 标签（训练中评估效果、调参）
        '''
        return train_texts, val_texts, train_labels, val_labels

    def _test_GET_TRAIN_DATA_SPLIT(self):
        print('测试 ➡️ 获取训练集、测试集：')
        train_texts, val_texts, train_labels, val_labels = (
            self._get_train_data_split()
        )
        print('训练集文本数量：', len(train_texts))
        print('训练集标签数量：', len(train_labels))
        print('验证集文本数量：', len(val_texts))
        print('验证集标签数量：', len(val_labels))
        print('训练集文本前5行：', train_texts[:5])
        print('训练集标签前5行：', train_labels[:5])
        print('验证集文本前5行：', val_texts[:5])
        print('验证集标签前5行：', val_labels[:5])


    # todo 2. --------数据预处理2. 返回 文本编码和标签--------
    def _preprocess_data(self, texts, labels):
        """数据预处理：文本列表[str]→tensor; 标签字典→[0, 1]
        1. BERT-BASE-CHINESE构成的分词器对文本进行编码（分词, 截断, 填充）
        2. 将标签转换为数字
        :param texts: 待处理的文本列表 [str]
        :param labels: 文本对应的标签列表(通用知识 或 专业咨询) [str]
        :return:
            encodings: texts[str]→pytorch的张量
            labels: 标签字典→[0, 1]
        """
        # 1. 文本编码: 用BERT分词器进行分词
        encodings = self.tokenizer(
            texts, # 待处理的文本列表 [str]
            max_length=128, # 最大长度，BERT-BASE-CHINESE最多支持512
            truncation=True, # 超过128长度就截断，超出部分丢弃
            padding="max_length", # 不够128的填充到最大长度128
            return_tensors="pt" # 要求返回pytorch张量类型
            # 如果不加return_tensors="pt"，返回list
            # pytorch张量类型可以直接进行训练
        )
        # 2. 标签转换, 把文本标签转成数字.
        # self.label_map = {"通用知识": 0, "专业咨询": 1}
        labels = [self.label_map[label] for label in labels]
        # 3. 返回结果.
        return encodings, labels

    def _test_PREPROCESS_DATA(self):
        print('测试 ➡️ 数据预处理：')
        texts = ["1024乘以768等于多少？", "专业咨询如何使用BERT进行分类？"]
        labels = ["通用知识", "专业咨询"]
        encodings, labels = self._preprocess_data(texts, labels)
        print('文本编码：', encodings) # dict
        print('标签：', labels) # 标签： [0, 1]
        '''文本编码： 
        {'input_ids': tensor([[101,11570,733,809,12472,5023,754,1914,2208,8043,   102,0, 0,...
        'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, ...
         'attention_mask': tensor([[1,1,1,1,1,1,1,1,1,1,1,0,0,...
        # input_ids：tokens_id 字符转成数字 101[CLS]表示开头，102[SEP]表示结束
        # attention_mask：注意力掩码 1和0构成，有内容的标记为1，填充的标记为0
        # token_type_ids：句子边界标记 0和1构成，标记当前输入的 Token 属于哪一个句子
            单句任务：token_type_ids 全是0
            双句/句对任务：第一句全是0，第二句全是1
        # 单句任务是让AI读懂“一句话”（分类），双句任务是让A 理解“两句话之间的逻辑和语义联系”
        # 双句/句对任务包含：问答、推理（包含、矛盾、中立）、语义相似度计算、下一句预测
        # 单句任务包含：
            单句分类：情感分析、语法连贯性判断、按标签分类
            单句标注/序列标注：
                命名实体识别（NER）：蔡徐坤开大了！——标注出蔡徐坤是人名
                词性标注：给句子中的每个单词标上它是名词、动词还是形容词
        '''


    # todo 3. --------数据预处理3. 返回 Dataset类型的数据集--------
    def _create_dataset(self, encodings, labels):
        # 1. 定义DataSet类, 继承PyTorch的DataSet
        class DataSet(torch.utils.data.Dataset):
            # 初始化方法
            def __init__(self, encodings, labels):
                super().__init__() # 初始化父类信息
                self.encodings = encodings # 文本编码
                self.labels = labels # 标签
            # 返回数据集长度
            def __len__(self):
                return len(self.labels)
            # 获取数据集的某个样本
            def __getitem__(self, idx):
                '''提取第idx条数据，添加张量形式的标签，返回字典'''
                item = {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx],
                    'token_type_ids': self.encodings['token_type_ids'][idx],
                    'labels': torch.tensor(self.labels[idx])
                }
                return item
        # 2. 返回实例化DataSet对象
        return DataSet(encodings, labels)

    def _test_CREATE_DATASET(self):
        texts = ["1024乘以768等于多少？", "专业咨询如何使用BERT进行分类？"]
        labels = ["通用知识", "专业咨询"]
        encodings, labels = self._preprocess_data(texts, labels) #文本编码和标签
        dataset = self._create_dataset(encodings, labels)
        print('数据集长度：', len(dataset))
        print('数据集第一个样本：', dataset[0])
        # dataset[0] 调用的 DataSet.__getitem__ 函数
        '''{
            'input_ids': tensor([101, 2345, ..., 102]), # 只有第0条的输入tokensID
            'attention_mask': tensor([1, 1, ..., 1]),  # 只有第0条的注意力掩码
            'token_type_ids': tensor([0, 0, ..., 0]), # 只有第0条的句子边界标记
            'labels': tensor(0) # 只有第0条的标签
        }'''


    # todo 4. --------定义并返回训练参数对象--------
    def _get_training_args(self):
        # 配置训练参数，返回训练参数对象
        training_args = TrainingArguments(
            output_dir=self.new_model_path,     # todo 检查点、日志、最终模型的保存路径
                                            # 如果目录不存在，Trainer 会自动创建
                                            # 训练中断后，可以通过指向该目录进行断点续训
            num_train_epochs=3,                 # todo 训练轮数 2-4即可 过大容易导致过拟合
            per_device_train_batch_size=8,      # todo 每次训练迭代时，送入GPU/CPU的样本数量
                                            # 如果多显卡并行 每个设备的样本数量
                                            # 比如4张卡 全局Batch Size就是 8*4=32
                                            # 显存不足时请调小此值
            per_device_eval_batch_size=16,      # todo 验证(每个设备的)批次大小
                                            # 验证阶段不需要计算梯度，显存占用较小
                                            # 比训练时的batch size更大以加快评估速度
            warmup_steps=20,                    # todo 学习率预热步数
                                            # 训练前20步 学习率从0线性增加到默认的初始学习率。
                                            # 防止初期破坏预权重分布，避免初始震荡
            # learning_rate=2e-5                # todo 默认学习率5e-5，即 0.00005
                                            # 通常在 1e-5 到 5e-5 之间调整
                                            # 过高会破坏预训练权重，过低则收敛太慢
            weight_decay=0.01,                  # todo 权重衰减系数（L2 正则化系数）
                                            # 限制模型参数变得过大，防止过拟合 通常都是0.01
            logging_steps=10,                   # todo 每隔多少步保存一次日志
            eval_strategy="epoch",              # todo 评估频率 每轮训练结束后进行验证评估.
            save_strategy="epoch",              # todo 保存频率 每轮训练结束后, 保存模型检查点.
                                            # "no" 不进行评估
                                            # "steps" 每隔n步评估一次(需要参数eval_steps=n)
            load_best_model_at_end=True,        # todo 训练结束后加载验证集表现最好的模型.
                                            # 可以防止因为训练过度，保存频率>=评估频率才生效
                                            # eval_strategy必须设置为epoch或steps
            save_total_limit=1,                 # todo 只保存一个检查点,即最优的模型,节省空间
            metric_for_best_model="eval_loss",  # todo 以验证集的损失作为最优模型的评判标准
                                            # 默认情况下Loss越小越好
                                            # 如果用准确率Accuracy作为指标 需要额外设置
                                            #   greater_is_better=True
            fp16=(True if self.device=='cuda' else False), # todo 是cuda就启用半精度训练
                                            # 显存占用减半 速度提升约2倍 通常不会损失精度
        )
        return training_args

    def _test_GET_TRAINING_ARGS(self):
        print('测试 ➡️ 查看完整的模型训练参数')
        training_args = self._get_training_args()
        print(training_args)


    # todo 5. --------定义并返回训练对象--------
    def _get_trainer(self, train_dataset, val_dataset, training_args):
        """构建并返回配置好的Trainer对象
        :train_dataset (torch.utils.data.Dataset): 用于模型微调的训练集
        :val_dataset (torch.utils.data.Dataset): 用于验证和监控模型过拟合情况的验证集
        :training_args :
        return: transformers.Trainer 模型训练器对象
        """
        def compute_metrics(eval_pred):
            """计算准确率"""
            # 接收模型输出的预测分数logits和真实标签labels
            logits, labels = eval_pred
            # 从分数里选出得分最高的类别当作预测结果
            # axis=-1 最后一维，就是在每一行（每个样本）
            predictions = np.argmax(logits, axis=-1)
            # 对比预测和真实标签，算出正确率并返回
            accuracy = (predictions == labels).mean()
            print("accuracy:", accuracy)
            '''举例：
            预测分数 logits = [
                [1.2, -0.5],   # 第1条样本：类别0分1.2，类别1分-0.5
                [0.3, 2.1],    # 第2条样本：类别0分0.3，类别1分2.1
                [-0.8, 0.6]    # 第3条样本：类别0分-0.8，类别1分0.6
            ]
            predictions = [0, 1, 1]
            真实标签 labels = [0, 1, 0]
            np.argmax(logits, axis=-1) # 在每行（最后一维）取最大值下标
            predictions == labels # 逐一比对，返回[True, True, False] → [1, 1, 0]
            .mean() 求平均 (1+1+0)÷3≈0.667 # 66.7%
            最终返回 {"accuracy": 0.667}
            '''
            return {"accuracy": accuracy}

        # 初始化 Trainer: 封装模型, 训练参数, 数据集和评估指标计算方法.
        trainer = Trainer(
            model=self.model,                # 要被训练(微调)的预训练模型
            args=training_args,              # 训练参数
            train_dataset=train_dataset,     # 训练数据集
            eval_dataset=val_dataset,        # 验证数据集
            compute_metrics=compute_metrics  # 自定义的计算评估指标(准确率)函数名
        )
        return trainer

    def _test_GET_TRAINER(self):
        print('测试 ➡️ 创建模型训练对象')
        train_texts = ["1024乘以768等于多少？", "专业咨询如何使用BERT进行分类？"]
        train_labels = ["通用知识", "专业咨询"]
        val_texts = ["1024乘以768等于多少？", "专业咨询如何使用BERT进行分类？"]
        val_labels = ["通用知识", "专业咨询"]
        # 文本编码和标签
        train_encodings, train_labels = self._preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self._preprocess_data(val_texts, val_labels)
        # 获取训练集和验证集对象
        train_dataset = self._create_dataset(train_encodings, train_labels)
        val_dataset = self._create_dataset(val_encodings, val_labels)
        training_args = self._get_training_args()
        trainer = self._get_trainer(train_dataset, val_dataset, training_args)
        print(trainer)
        '''第一次运行测试函数会自动创建存放新微调模型的空目录，
        再次运行就会报错，因为文件夹里是空的——可以删除该空目录
        '''


    # todo 6. --------模型训练、评估--------
    def _train_model(self, data_file=Config.BERT_TRAIN_DATA):
        """训练模型、评估模型"""
        # 1. 数据集前置准备
        train_texts, val_texts, train_labels, val_labels = self._get_train_data_split(data_file)
        self.logger.info(f'分割训练集和验证集，大小分别为：{len(train_texts)}, {len(val_texts)}')
        # 2. 数据预处理 【注意：文本列表[str]→编码; 标签字典→[0, 1]】
        train_encodings, train_labels = self._preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self._preprocess_data(val_texts, val_labels)
        self.logger.info('文本列表[str]转为pytorch张量tensor; 标签字典转[0, 1]')
        # 3. 创建数据集对象
        train_dataset = self._create_dataset(train_encodings, train_labels)
        val_dataset = self._create_dataset(val_encodings, val_labels)
        self.logger.info('本文和标签放到一起构建并返回Dataset类型的训练集和测试集')
        # 4. 创建训练参数对象
        training_args = self._get_training_args()
        self.logger.info(f'已创建训练参数对象，完整参数：{training_args}')
        # 5. 创建模型训练对象
        trainer = self._get_trainer(train_dataset, val_dataset, training_args)
        self.logger.info(f'已创建模型训练对象：{trainer}')
        # 6. 训练模型
        self.logger.info('开始训练模型...')
        trainer.train()
        # 7. 保存模型 和分词器到指定路径, 便于后续加载使用
        self.model.save_pretrained(self.new_model_path)
        self.tokenizer.save_pretrained(self.new_model_path)
        self.logger.info(f'模型已保存到：{self.new_model_path}')
        # 8. 用验证集评估模型准确率
        self._evaluate_model(val_texts, val_labels)

    def _evaluate_model(self, val_texts, val_labels):
        """用验证集评估模型性能
        :param texts: 待评估的文本列表(需要预测的输入文本)
        :param labels: 文本对应的真实标签列表(已转为数字形式), 0:通用知识, 1:专业咨询
        :return: 无,评估结果通过日志输出.
        """
        # 1.文本进行分词编码
        encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        logger.info('开始评估模型性能...')
        # 2. 将编码后的文本 和 真实标签封装为PyTorch的 DataSet对象.
        # 注意 使用已经转换为[1,0]的val_labels
        dataset = self._create_dataset(encodings, val_labels)
        # 3. 打印数据集信息: 验证数据集长度和第一个样本格式是否正确.
        self.logger.info(f'输出数据集样本数量: {len(dataset)}')
        self.logger.info(f'输出第一个样本的格式(编码 + 标签): {dataset[0]}')
        # 4. 初始化Trainer: 仅传入模型(评估阶段无须训练参数！)
        trainer = Trainer(model=self.model)
        # 5. 执行预测: 让模型对整个数据集进行预测, 得到所有样本的预测结果.
        predictions = trainer.predict(dataset) # predict()不会偷看标签
        # 6. 提取预测标签
        # 模型输出的是每个类别的"可能性分数"
        # 例如: [0.8, 0.2] -> 0(通用知识),  [0.3, 0.7] -> 1(专业咨询)
        # 对预测结果（取最后一维）进行"最大可能性下标"提取, 得到最终的预测标签.
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        # 8. 输出分类报告(日志记录): 详细说明模型对每个类别的预测能力.
        logger.info("分类报告:")
        # sklearn.metrics.classification_report
        report = classification_report(
            val_labels,         # 标准答案(真实标签)
            pred_labels,        # 模型预测的标签
            target_names=["通用知识", "专业咨询"],  # 把数字转成中文名称, 方便阅读.
        )
        logger.info(report)
        '''各项评估指标明细
        # precision(精准率)：预测为该类别的样本里，实际真的属于该类的比例
        # recall(召回率)：实际属于该类的样本里，被成功预测出来的比例
        # f1-score：精确率和召回率的综合分数，越接近1效果越好
        # support：该类别对应的样本数量
                precision recall f1-score support
        通用知识   1.00    0.96   0.98     23   # 共23条样本，精准度满分，召回略低，综合表现优秀
        专业咨询   0.99    1.00   0.99     77   # 共77条样本，召回满分，整体表现极佳
        accuracy                 0.99     100  # 整体准确率99%，总样本100条
       macro avg  0.99    0.98   0.99     100  # 宏平均：不按样本数量加权，各类指标取平均
    weighted avg  0.99    0.99   0.99     100  # 加权平均：按各类样本数量加权计算，贴合整体真实表现
        '''
        # 9. 输出混淆矩阵: 显示模型对每个类别的预测能力.
        logger.info("输出混淆矩阵:")
        # sklearn.metrics.confusion_matrix
        logger.info(confusion_matrix(val_labels, pred_labels))
        '''输出混淆矩阵:
        [[22  1]   # 通用知识共23条， 22条识别为通用知识 1条识别为专业咨询
         [ 0 77]]  # 专业咨询共77条， 0条识别为通用知识 77条识别为专业咨询
        # precision精准率计算：
            - 通用知识 22÷(22+0)=1.00 
            - 专业咨询 77÷(1+77)≈0.99
        # recall召回率计算：
            - 通用知识 22÷(22+1)≈0.96
            - 专业咨询 77÷(0+77)=1.00
        '''

    def _test_EVALUATE_MODEL(self, random_state=42, test_size=0.2):
        """评估已训练模型的性能（使用划分的验证集）
        Args:
            random_state: 随机种子，默认为42，确保结果可复现
            test_size: 验证集比例，默认为0.2（20%）
        """
        print('评估 ➡️ 已训练模型的性能')
        # 加载测试数据
        with open(Config.BERT_TRAIN_DATA, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]
        texts = [item['query'] for item in data]
        labels = [item['label'] for item in data]

        # 划分训练集和验证集（默认参数时与训练时保持一致）
        _, val_texts, _, val_labels = train_test_split(
            texts, # 完整文本
            labels, # 完整标签
            test_size=test_size, # 验证集比例
            random_state=random_state # 随机种子
        )

        print(f"使用随机种子: {random_state}")
        print(f"验证集比例: {test_size}")
        print(f"验证集大小: {len(val_texts)} 条")

        # 将字符串标签转换为数字标签
        # self.label_map == {"通用知识": 0, "专业咨询": 1}
        val_labels_numeric = [self.label_map[label] for label in val_labels]

        # 调用_evaluate_model评估函数评估验证集
        self._evaluate_model(val_texts, val_labels_numeric)


    # todo 7. --------意图识别API--------
    def predict_category(self, query):
        """用训练好的模型对单个查询进行分类, 判断它是"通用知识", 还是"专业咨询"
        :param query: 用户输入的查询文本, 例如: 什么是AI?  我的保险怎么理赔?
        :return: 意图识别的分类结果, 要么是"通用知识", 否则是"专业咨询"
        """
        # 检查模型是否加载
        if self.model is None:
            logger.error("模型未训练或加载")
            return "通用知识" # 默认返回通用知识

        # 对查询进行编码: 把用户输入的文本转成模型能看懂的格式(和训练时的处理方法一致)
        encoding = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        # print(encoding)
        # 将编码移到指定设备
        # 模型在GPU上, 数据也传到GPU上; 模型在CPU上, 数据必须传到CPU上
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # print(encoding)
        '''{
            'input_ids': tensor([[101, 2769, 4638,  ...]], device='cuda:0'),  # 文本的数字ID
            'attention_mask': tensor([[1, 1, 1, ...]], device='cuda:0')       # 注意力掩码
            'token_type_ids': tensor([[0, 0, 0, ...]], device='cuda:0')       # 句子标记
        }'''

        # 开始预测(不计算梯度，进行预测)
        with torch.no_grad():
            # 模型进行分类预测并输出
            outputs = self.model(**encoding)
            # print(outputs)
            # print(outputs.logits) # tensor([[-0.6638,  0.5243]], device='cuda:0')
            # 获取预测结果，返回张量中最后一个维度（dim=-1）的最大值的索引
            # item() 把PyTorch中的张量转成Python中的标量(数字)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            # print(prediction) # 1
        # 根据预测结果返回类别
        # self.label_map = {"通用知识": 0, "专业咨询": 1} → 构造反向字典
        reverse_label_map = {v: k for k, v in self.label_map.items()}
        return reverse_label_map[prediction]

    def _test_PREDICT_CATEGORY(self):
        print('测试➡️ 查询问题的意图识别')
        # 测试分类
        queries = [
            "768除以12是多少？",  # 通用知识
            "学习AI大模型课程就业薪资多少？",  # 专业咨询
            "什么是机器学习？",  # 通用知识
            "如何申请教育贷款？",  # 专业咨询
        ]
        query_classify = QueryClassifier()
        for q in queries:
            result = query_classify.predict_category(q)
            print(f"查询: {q}")
            print(f"分类: {result}\n")


if __name__ == '__main__':
    query_classify = QueryClassifier()
    print('测试开始 ... ')

    '''初始化'''
    # query_classify._test_INIT()

    '''训练集 验证集'''
    # query_classify._test_GET_TRAIN_DATA_SPLIT()

    '''文本编码成数字(列号)'''
    # query_classify._test_PREPROCESS_DATA()

    '''构建数据集'''
    # query_classify._test_CREATE_DATASET()

    '''模型训练参数'''
    # query_classify._test_GET_TRAINING_ARGS()

    '''模型训练对象'''
    # query_classify._test_GET_TRAINER()

    '''模型训练+评估'''
    # query_classify._train_model()

    '''模型评估'''
    # query_classify._test_EVALUATE_MODEL(
    #     random_state=77, test_size=0.3
    # )

    '''意图识别：预测问题类别'''
    query_classify._test_PREDICT_CATEGORY()

