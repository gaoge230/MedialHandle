# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'selectmodel.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QStringListModel
from utils import util

class Ui_SelectModel(object):
    def setupUi(self, SelectModel):
        SelectModel.setObjectName("SelectModel")
        SelectModel.resize(400, 300)
        self.listView = QtWidgets.QListView(SelectModel)
        self.listView.setGeometry(QtCore.QRect(10, 10, 301, 281))
        self.listView.setObjectName("listView")
        self.pushButton = QtWidgets.QPushButton(SelectModel)
        self.pushButton.setGeometry(QtCore.QRect(320, 10, 75, 23))
        self.pushButton.setObjectName("pushButton")
        slm = QStringListModel();  # 创建mode


        self.qList = util().getModelList('./data/model')
        slm.setStringList(self.qList)  # 将数据设置到model
        self.listView.setModel(slm)  ##绑定 listView 和 model

        self.retranslateUi(SelectModel)
        QtCore.QMetaObject.connectSlotsByName(SelectModel)

    def retranslateUi(self, SelectModel):
        _translate = QtCore.QCoreApplication.translate
        SelectModel.setWindowTitle(_translate("SelectModel", "Dialog"))
        self.pushButton.setText(_translate("SelectModel", "确定"))
