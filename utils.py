#!/usr/bin/env/ python
# -*- coding: utf-8 -*-
# @File  : util.py 
# @Author: 高歌
# @Date  : 2021/11/30
# @Desc  :
from PyQt5.QtWidgets import QMessageBox
import os

class util(object):
    def __init__(self):
        curPath = os.path.abspath(os.path.dirname(__file__))
        self.recycleBinPath = curPath+"//data//RecycleBin"

    def getDefaultModel(self):
        return "defultModel.pth.tar"

    def messageDialog(self,message):
        # 核心功能代码就两行，可以加到需要的地方
        msg_box = QMessageBox(QMessageBox.Warning, '警告', message)
        msg_box.exec_()

    def getModelList(self,path):
        listdir = os.listdir(path)
        print(type(listdir))
        for file in listdir:
            print(file)
        return listdir

    def setrecycleBinPath(self,path):
        self.recycleBinPath=path

    def getrecycleBinPath(self):
        return self.recycleBinPath


# if __name__ == '__main__':
#     u= util()
#     path = './data/model'
#     print(type(['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 2', 'Item 3', 'Item 4', 'Item 2', 'Item 3', 'Item 4'])==
#           type(u.getModelList(path)))
#     print(u.getModelList(path))

