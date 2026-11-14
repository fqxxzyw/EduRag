import re
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForTokenClassification
from langchain_text_splitters import CharacterTextSplitter

# 本地模型路径
model_path = '../models/nlp_bert_document-segmentation_chinese-base'


class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.device = device

        # 加载模型和分词器（只在初始化时加载一次）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        # 将模型移动到指定设备
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

    def _split_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 使用标点符号作为分割标志
        sentence_delimiters = ['。', '！', '？', '；', '：', '…']

        # 构建正则表达式模式
        pattern = '([' + ''.join(re.escape(d) for d in sentence_delimiters) + '])'

        # 分割文本
        parts = re.split(pattern, text)

        # 重组句子
        sentences = []
        current_sentence = ""
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                sentence = parts[i] + parts[i + 1]
            else:
                sentence = parts[i]

            if sentence.strip():
                sentences.append(sentence.strip())

        #  ['你好。', '世界！', '哈哈？', '拜拜']

        # 处理没有标点的情况
        if not sentences and text.strip():
            sentences = [text.strip()] # [你好世界哈哈拜拜,]

        return sentences

    def _segment_with_model(self, sentences: List[str], threshold: float = 0.5) -> List[str]:
        """使用BERT模型进行段落分割"""
        if not sentences:
            return []

        # 如果只有一个句子，直接返回
        if len(sentences) == 1:
            return [sentences[0]]

        # 获取每个句子是段落开头的概率
        boundary_probs = []

        # 逐句处理（避免内存溢出）
        with torch.no_grad():
            for i, sentence in enumerate(sentences):
                # 编码当前句子
                inputs = self.tokenizer(
                    sentence,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                )

                # 移动到相同设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 模型推理
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

                # 获取边界概率（假设label 1表示新段落开始）
                # 注意：需要根据实际模型输出的标签映射调整
                sentence_prob = probs[0, 0, 1].item()  # [batch, token, class]
                boundary_probs.append(sentence_prob)

        # 根据概率决定分割点
        paragraphs = []
        current_paragraph = []

        for i, sentence in enumerate(sentences):
            current_paragraph.append(sentence)

            # 判断是否需要在这里分割（下一个句子是段落开头）
            if i < len(sentences) - 1 and boundary_probs[i + 1] >= threshold:
                paragraphs.append(''.join(current_paragraph))
                current_paragraph = []

        # 添加最后一个段落
        if current_paragraph:
            paragraphs.append(''.join(current_paragraph))

        return paragraphs

    def split_text(self, text: str, threshold: float = 0.5) -> List[str]:
        """
        分割文本为段落

        Args:
            text: 输入文本
            threshold: 段落边界判断阈值（0-1之间，值越大越保守）

        Returns:
            分割后的段落列表
        """
        # PDF模式下的文本预处理
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub(r'\s', " ", text)
            text = re.sub(r"\n\n", "", text)

        # 先分割成句子
        sentences = self._split_sentences(text)

        # 使用模型进行语义分割
        if len(sentences) > 1:
            paragraphs = self._segment_with_model(sentences, threshold)
        else:
            # 如果句子太少，直接返回
            paragraphs = [text] if text.strip() else []

        # 清理输出格式
        cleaned_paragraphs = []
        for para in paragraphs:
            # 移除多余的换行符和制表符
            cleaned = re.sub(r'\n+', '\n', para.strip())
            if cleaned:
                cleaned_paragraphs.append(cleaned)

        return cleaned_paragraphs


# 为了兼容原代码的简单调用方式，提供一个简化版本
class SimpleAliTextSplitter:
    """简化版本，不依赖langchain"""

    def __init__(self, pdf: bool = False, device: str = "cpu"):
        self.pdf = pdf
        self.device = device

        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 创建实际的分割器
        self.splitter = AliTextSplitter(pdf=pdf, device=device)

    def split_text(self, text: str) -> List[str]:
        """简化的接口，直接调用分割器"""
        return self.splitter.split_text(text)



if __name__ == '__main__':
    # 测试文本
    test_text = '''移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。'''

    # 创建分割器实例
    model_split = AliTextSplitter(device="cpu")
    # 可以改为 "cuda" 使用GPU（需要额外安装gpu版本的torch）

    # 执行分割
    result = model_split.split_text(text=test_text, threshold=0.5)

    # 打印结果
    print("原始文本长度:", len(test_text))
    print("分割后段落数:", len(result))
    print("\n分割结果:")
    for i, para in enumerate(result, 1):
        print(f"\n段落 {i}:")
        print(para)
        print(f"段落长度: {len(para)} 字符")
        print("-" * 50)