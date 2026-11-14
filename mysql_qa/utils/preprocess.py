import jieba, traceback
from typing import Generator
from base import logger

def preprocess_text(
        text:str,
        cut_all=False, # 默认精准模式
        is_generator=True # True返回生成器 False列表
    ) -> Generator | list[str]:
    """对文本进行分词，返回 [] 或 【产生词的生成器】
    1. text.lower() 有英文的先转小写，再分词
    2. 分词返回生成器: jieba.cut(text, cut_all=cut_all)
    3. 分词返回list: jieba.lcut(text, cut_all=False)
    4. 分词模式cut_all参数
        默认False 精准模式，常用最准，计算机 → ['计算机']
        True 全模式，计算机 → ['计算', '算机', '计算机']
    """
    try:
        # logger.info(f"开始处理文本，原始文本长度：{len(text)}，全模式：{cut_all}，是否返回生成器：{is_generator}")
        text = text.lower() # 有英文就转小写
        if is_generator:
            result = jieba.cut(text, cut_all=cut_all)
            # logger.info(f"文本处理完成，返回分词生成器")
        else:
            result = jieba.lcut(text, cut_all=cut_all)
            # logger.info(f"文本处理完成，返回分词列表")
        return result
    except Exception as e:
        logger.error(f"文本处理异常：{traceback.format_exc()}")
        return []


if __name__ == '__main__':
    # 👇 测试
    test_text = """
    宙斯手持雷电法杖：我乃万神之王！执掌雷霆，统御诸神！
    玉皇大帝言出法随：你爱谁谁，掏20块钱，我下楼喝碗胡辣汤。
    职业法师刘海柱 我没开挂卢本伟 水晶吊坠周淑怡 宗门圣主蔡徐坤 
    雷电法王杨永信 落樱神斧华盛顿 不准道人海森堡 虐猫狂人薛定谔 
    屏蔽尊者法拉第 万法归一麦克韦 没有子弹燕双鹰 
    法外狂徒张三 无头骑士路易 潮汐海灵袁华 全程围观...
    """

    print("【测试1】👉 默认：精准模式 + 返回生成器")
    gen = preprocess_text(test_text)
    print("类型：", type(gen))
    print("转列表结果：", list(gen))

    print("【测试2】👉 精准模式 + 返回列表")
    lst = preprocess_text(test_text, is_generator=False)
    print("类型：", type(lst))
    print("结果：", lst)

    print("【测试3】👉 全模式 + 返回列表")
    lst_all = preprocess_text(test_text, cut_all=True, is_generator=False)
    print("结果：", lst_all)

    print("【测试4】👉 英文自动转小写")
    eng_text = "Hello WORLD 你好"
    res = preprocess_text(eng_text, is_generator=False)
    print("结果：", res)

    print("【测试5】👉 空文本/异常输入")
    res1 = preprocess_text("", is_generator=False) # []
    res2 = preprocess_text(None, is_generator=False) # 捕获异常，返回[]
    print("空字符串：", res1)
    print("None：", res2)