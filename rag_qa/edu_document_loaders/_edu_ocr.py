
'''
啥是 OCR(Optical Character Recognition)?   可以直接问 大模型, 让他给我们解释下.
    简单说就是"光学字符识别"技术，能让电脑"看懂"图片里的文字，把图片上的手写体、印刷体文字转换成可编辑的文本(比如照片里的笔记、扫描版 PDF 里的文字)

paddleocr：解析图片中的文字，也可以进行表格识别

rapidocr_paddle 和 rapidocr_onnxruntime 两种导入方式
    主要区别在于它们所使用的推理引擎和硬件支持
    选择哪种方式最合适取决于你的硬件环境和性能需求。

    当你有 GPU 且追求速度时：使用 rapidocr_paddle。
    PaddlePaddle 原生支持在 GPU 上推理 PaddleOCR 模型，速度更快。
    当只有 CPU 且需要高效推理时：使用 rapidocr_onnxruntime。
    它在 CPU 上进行了优化，资源占用较低.
'''

def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    """
    :return : paddleocr图像识别对象
    """
    try:
        from rapidocr_paddle import RapidOCR
        '''
        det_use_cuda=True：启用检测模型的GPU加速。cls_use_cuda=True：启用分类模型的GPU加速。rec_use_cuda=True：启用识别模型的GPU加速。
        '''
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr


