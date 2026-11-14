# 添加当前目录到系统路径
import sys, os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

from edu_docloader import OCRDOCLoader
from edu_pptloader import OCRPPTLoader
from edu_imgloader import OCRIMGLoader
from edu_pdfloader import OCRPDFLoader

__all__ = [
    "TextLoader",
    "UnstructuredMarkdownLoader",
    "OCRDOCLoader",
    "OCRPPTLoader",
    "OCRIMGLoader",
    "OCRPDFLoader",
]

