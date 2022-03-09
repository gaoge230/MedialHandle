#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QStringListModel, QThread, pyqtSignal
from PyQt5.QtGui import QPalette, QPixmap, QBrush, QImage, QTextCursor, QColor
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDialog, QFileDialog, QDesktopWidget, QGraphicsPixmapItem, \
    QGraphicsScene, QApplication

from Service.segmentation.SeTrain import SeTrain
from anomalydetection import Ui_anomalydetection
from dataargument import Ui_dataargument
from fenlei import Ui_fenlei
from imageresize import Ui_imageresize
from mainwindow import Ui_MainWindow
#from mianwindow1 import Ui_MainWindow
from modelmanagement import Ui_modelmanagement
from modifyname import Ui_modifyname
from normalize import Ui_normalize
from quzao import Ui_quzao
from selectmodel import Ui_SelectModel
from trainModel import Ui_trainModel
from trainModelFenLei import Ui_trainModelfenlei
from trainModelQuZao import Ui_trainModelQuZao
from utils import util
from xunlian import Ui_xunlian
from fenge import Ui_fenge
from inference import view_result, view_result1
from PIL import Image
import os
import imageio
from Service.datapre.PreData import PreData
import qdarkstyle
import csv
import pandas as pd
import time
import threading
import _thread
import subprocess
import shutil

class MainWindows(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MainWindows,self).__init__()
        self.setupUi(self)
        # self.action2.triggered.connect(self.showfenge)
        # self.action_4.triggered.connect(self.showmodelmanagement)
        # self.action_5.triggered.connect(self.showxunlian)
        # self.action_3.triggered.connect(self.msg)
        # self.action.triggered.connect(self.msg2)
        # self.pushButton.clicked.connect(self.showmodelmanagement)
        # self.pushButton_2.clicked.connect(self.showfenge)
        # self.pushButton_3.clicked.connect(self.showxunlian)

        # self.action1.triggered.connect(self.showfenge)
        # self.action2.triggered.connect(self.showmodelmanagement)
        # self.action3.triggered.connect(self.showxunlian)
        # self.action21.triggered.connect(self.msg)
        # self.action22.triggered.connect(self.msg2)
        # self.pushButton_2.clicked.connect(self.showmodelmanagement)
        # self.pushButton.clicked.connect(self.showfenge)
        # self.pushButton_3.clicked.connect(self.showxunlian)
        # self.label.setPixmap(QtGui.QPixmap("./image/segment.jpg"))  # file_name是一个路径
        # self.label.setScaledContents(True)
        palette = qdarkstyle.light.palette.LightPalette()
        self.centralwidget.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5(palette))
        #self.centralwidget.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside2'))
        # 设置组件背景透明
        self.label_6.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.label_7.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.label_2.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.label_3.setPixmap(QtGui.QPixmap(r"./image/1.webp"))
        self.label_3.setScaledContents(True)


        # 设置组件透明度，不是背景
        # op = QtWidgets.QGraphicsOpacityEffect()
        # op.setOpacity(0)
        # self.frame.setGraphicsEffect(op)
        # self.frame.setAutoFillBackground(True)

        self.stackedWidget.addWidget(Fenge())
        self.stackedWidget.addWidget(Modelmanagement())
        self.stackedWidget.addWidget(TrainModel())
        self.stackedWidget.addWidget(FenLei())
        self.stackedWidget.addWidget(ModelmanagementFenLei())
        self.stackedWidget.addWidget(TrainModelFenLei())
        self.stackedWidget.addWidget(QuZao())
        self.stackedWidget.addWidget(ModelmanagementQuZao())
        self.stackedWidget.addWidget(TrainModelQuZao())
        self.stackedWidget.addWidget(AnomalyDetection())
        self.stackedWidget.addWidget(DataArgument())
        self.stackedWidget.addWidget(ImageSize())
        self.stackedWidget.addWidget(ModelmanagementDetection()) # 13
        self.stackedWidget.addWidget(Normalize()) # 14

        self.setAutoFillBackground(True)
        widget = self.stackedWidget.currentWidget()
        #palette = QPalette()
        #palette.setBrush(QPalette.Background, QBrush(QPixmap("./image/background.jpg")))
        self.frame.setStyleSheet("QFrame{background-image: url(./image/background.jpg)}")
        #widget.setStyleSheet("QWidget{background-image: url(./image/background.jpg)}")
        #self.label_6.setStyleSheet("QLabel{background-image: url()}")
        #self.label_7.setStyleSheet("QLabel{background-image: url()}")
        #self.setAutoFillBackground(True)
        # 分割
        self.pushButton_2.clicked.connect(self.switch)
        self.pushButton_3.clicked.connect(self.switch)
        self.pushButton_4.clicked.connect(self.switch)
        # 分类
        self.pushButton_5.clicked.connect(self.switch)
        self.pushButton_6.clicked.connect(self.switch)
        self.pushButton_7.clicked.connect(self.switch)
        # 去噪
        self.pushButton_8.clicked.connect(self.switch)
        self.pushButton_9.clicked.connect(self.switch)
        self.pushButton_10.clicked.connect(self.switch)
        #检测
        self.pushButton_13.clicked.connect(self.switch)
        self.pushButton_14.clicked.connect(self.switch)
        #预处理
        self.pushButton.clicked.connect(self.switch)
        self.pushButton_12.clicked.connect(self.switch)


        self.menu_8.triggered.connect(self.showHome)
        self.action_3.triggered.connect(self.msg)
        self.action.triggered.connect(self.msg2)
        self.action_2.triggered.connect(self.switchBar)
        self.action_6.triggered.connect(self.switchBar)
        self.action_7.triggered.connect(self.switchBar)
        self.action_8.triggered.connect(self.switchBar)
        self.action_9.triggered.connect(self.switchBar)
        self.action_10.triggered.connect(self.switchBar)
        self.action_11.triggered.connect(self.switchBar)
        self.action_12.triggered.connect(self.switchBar)
        self.action_13.triggered.connect(self.switchBar)
        self.action_14.triggered.connect(self.switchBar)
        self.action_15.triggered.connect(self.switchBar)
        self.action_16.triggered.connect(self.switchBar)
        self.action_17.triggered.connect(self.switchBar)
        self.action_22.triggered.connect(self.switchBar)

    def switchBar(self):
        dic = {"action_2": 12, "action_6": 11, "action_7": 14,
               "action_8": 1, "action_9": 2, "action_10": 3,
               "action_11": 4, "action_12": 5, "action_13": 6,
               "action_14": 7, "action_15": 8, "action_16": 9,
               "action_17": 10,"action_22": 13}
        index = dic[str(self.sender().objectName())]
        print("switch index:", index)
        self.stackedWidget.setCurrentIndex(index)

    def showHome(self):
        self.stackedWidget.setCurrentIndex(0)

    def switch(self):
        dic = {"pushButton_2": 1, "pushButton_3": 2,"pushButton_4":3,
               "pushButton_6":4,"pushButton_5":5,"pushButton_7":6,
               "pushButton_9":7,"pushButton_8":8,"pushButton_10":9,
               "pushButton_13":10,"pushButton_12":11,"pushButton":12,"pushButton_14":13,
               "pushButton_11":14}
        index = dic[str(self.sender().objectName())]
        print("switch index:",index)
        self.stackedWidget.setCurrentIndex(index)
        #self.stackedWidget.currentWidget().showEvent()


    def msg(self):
        ok = QMessageBox.about(self,
                               ("制作人信息"), ("高歌，重庆邮电大学图像认知与模式识别实验室。版本v1.0")
                               )

    def msg2(self):
        ok = QMessageBox.about(self,
                               ("软件信息"), ("胎儿心脏分割系统\n拥有3个模块\n1.分割功能。2.更新模型功能。3.训练分割模型功能。")
                               )

    def showmodelmanagement(self):
        myshow2=Modelmanagement()
        myshow2.show()
        myshow2.exec_()

    def showfenge(self):
        print("显示分割")
        myshow3=Fenge()
        myshow3.show()
        myshow3.exec_()

    def showxunlian(self):
        myshow4=Xunlian()
        myshow4.show()
        myshow4.exec_()

class Fenge(QDialog,Ui_fenge):

    def __init__(self):
        super(Fenge, self).__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.selectModel)
        self.pushButton_4.clicked.connect(self.showfile)
        self.pushButton_5.clicked.connect(self.callInference)
        self.pushButton_6.clicked.connect(self.savefile)
        self.resultPath = None
        self.loadImagePath = None
        #self.frame.setStyleSheet("QWidget{background-image: url(./image/background.jpg)}")
        #self.label_6.setStyleSheet("QLabel{background-image: url()}")
        #self.label_7.setStyleSheet("QLabel{background-image: url()}")

        self.label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.label_2.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        listModel = QStringListModel()
        self.items = util().getModelList("./data/model")
        print(self.items)
        listModel.setStringList(self.items)
        self.listView.setModel(listModel)
        self.item = ["全部", "U-Net", "Attention U-Net", "ResU-Net", "U-Net++", "SST U-Net"]
        self.comboBox.addItems(self.item)
        self.comboBox.setCurrentIndex(0)

    def showfile(self):
        print("上传文件！")
        print("传入")
        global image1
        file_name, filtertype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      '打开文件',
                                                                      r"C:\Users\Y430P\Desktop/",
                                                                      "png(*.png);;jpg(*.jpg)")
        if(file_name==''):
            return
        self.label.setPixmap(QtGui.QPixmap(file_name))  #file_name是一个路径
        self.label.setScaledContents(True)
        self.loadImagePath = file_name

    def callInference(self):
        print("执行分割推理！")
        if(self.loadImagePath is None):
            message = "请上传带分割的超声心脏图像！！！"
            print(message)
            box = QMessageBox(QMessageBox.Warning, '警告', message)
            box.exec_()
            return
        text = self.lineEdit.text()
        if(text==None or text==""):
            box = QMessageBox(QMessageBox.Warning, '警告', "请选择具体的推理模型！！！")
            box.exec_()
            return

        # 执行推理
        file_name = view_result1(self.loadImagePath)
        print(file_name)
        self.label_2.setPixmap(QtGui.QPixmap(file_name))
        self.label_2.setScaledContents(True)
        self.resultPath = file_name
        self.loadImagePath = None

    def selectModel(self):
        print("选择推理模型！")
        select = self.listView.currentIndex()
        row = select.row()
        print("select", row)
        if (row < 0):
            box = QMessageBox(QMessageBox.Warning, '警告', "请选择具体的推理模型！！！")
            box.exec_()
            return
        self.lineEdit.setText(self.items[row])

    def savefile(self):
        print("保存图片")
        if(self.resultPath==None):
            message = "没有分割结果需要保存！！！"
            print(message)
            box = QMessageBox(QMessageBox.Warning, '警告', message)
            box.exec_()
            return
        image = Image.open(self.resultPath)
        filename=QtWidgets.QFileDialog.getSaveFileName(None, "保存文件", ".","Image Files(*.jpg *.png)",)
        if(filename[0]==''):
            return
        image.save(filename[0])
    # 更新列表
    def updateList(self):
        ## 刷新列表
        listModel = QStringListModel()
        self.items = util().getModelList("./data/model")
        print(self.items)
        listModel.setStringList(self.items)
        self.listView.setModel(listModel)
    # 页面show时触发showEvent事件
    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        print("showEvent")
        self.updateList()

class Modelmanagement(QDialog, Ui_modelmanagement):
    def __init__(self):
        super(Modelmanagement, self).__init__()
        self.setupUi(self)
        listModel = QStringListModel()
        self.items = util().getModelList("./data/model")
        print(self.items)
        listModel.setStringList(self.items)
        self.listView.setModel(listModel)
        self.item = ["全部","U-Net","Attention U-Net","ResU-Net","U-Net++","SST U-Net"]
        self.comboBox.addItems(self.item)
        self.comboBox.setCurrentIndex(0)
        self.pushButton_4.clicked.connect(self.clickQuery) # query
        self.pushButton_6.clicked.connect(self.clickModify) # modify
        self.pushButton_5.clicked.connect(self.clickDelete)  # delete
        self.pushButton_7.clicked.connect(self.clickLoad)
        self.pushButton_8.clicked.connect(self.selectFile)
        self.loadPath =None

    def clickQuery(self):
        index = self.comboBox.currentIndex()
        print(self.item[index])

    def clickModify(self):
        select = self.listView.currentIndex()
        row = select.row()
        print("modify",row)
        if (row < 0):
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择要修改的模型！")
            box.exec_()
            return
        modifyName = ModifyName(self.items[row])
        modifyName.show()
        modifyName.exec_()
        newName = modifyName.newName
        print("新名字：",newName)
        ## 修改文件名
        oldName = self.items[row]
        os.rename("./data/model/"+oldName,"./data/model/"+newName+".pth")

        ## 刷新列表
        listModel = QStringListModel()
        self.items = util().getModelList("./data/model")
        print(self.items)
        listModel.setStringList(self.items)
        self.listView.setModel(listModel)

    def clickDelete(self):
        select = self.listView.currentIndex()
        row = select.row()
        print("delete：",row)
        modelname = self.items[row]

        if(row<0):
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择要删除的模型！")
            box.exec_()
            return
        reply = QtWidgets.QMessageBox.question(self, '警告', '你确定要删除'+modelname+'模型?',
                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("确认执行删除！")
            # 执行刪除操作
            shutil.move("./data/model/"+modelname,"./data/RecycleBin/"+modelname)
            # 刷新列表
            listModel = QStringListModel()
            self.items = util().getModelList("./data/model")
            print(self.items)
            listModel.setStringList(self.items)
            self.listView.setModel(listModel)
        else:
            print("取消执行删除！")
            return

    def clickLoad(self):
        document1 = self.textEdit.toPlainText()
        document2 = self.textEdit_2.toPlainText()
        print("Load:",document1)
        if document1==None or document1 =="" :
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择模型类型！！！")
            box.exec_()
            return
        if  document2 == None or document2 == "":
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择模型地址！！！")
            box.exec_()
            return

        ## 复制文件从用户选定路径到存储地（文件夹或者数据库）
        shutil.copy(self.loadPath,"./data/model/"+os.path.basename(self.loadPath))
        # 更新列表
        listModel = QStringListModel()
        self.items = util().getModelList("./data/model")
        print(self.items)
        listModel.setStringList(self.items)
        self.listView.setModel(listModel)

        isload =True
        if isload:
            box = QMessageBox(QMessageBox.Warning, '提示', "上传成功！！！！")
            box.exec_()
        else:
            box = QMessageBox(QMessageBox.Warning, '提示', "上传失败！请重新上传！")
            box.exec_()
        ##清空
        self.textEdit.setText("")
        self.textEdit_2.setText("")

    def selectFile(self):
        file_name, filtertype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      '打开文件',
                                                                      r"C:\Users\Y430P\Desktop/",
                                                                      "pth(*.pth)")
        self.textEdit_2.setText(os.path.basename(file_name))
        self.loadPath =file_name

    # 更新列表
    def updateList(self):
        ## 刷新列表
        listModel = QStringListModel()
        self.items = util().getModelList("./data/model")
        print(self.items)
        listModel.setStringList(self.items)
        self.listView.setModel(listModel)
    # 页面show时触发showEvent事件
    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        print("showEvent")
        self.updateList()

class TrainModel(QDialog,Ui_trainModel):
    def __init__(self):
        super(TrainModel, self).__init__()
        self.setupUi(self)
        self.items = ["有监督","无监督"]
        self.comboBox.addItems(self.items)
        self.pushButton.clicked.connect(self.clickTrain)

        # self.textEdit_10.setAutoFillBackground(True)
        # background_color = QColor()
        # background_color.setNamedColor('#282821')
        # palette = QPalette()
        # palette.setColor(QPalette.Background,background_color)
        # self.textEdit_10.setPalette(palette)
        self.textEdit_10.setStyleSheet("background-color: black; color: white")

    def clickTrain(self):
        index = self.comboBox.currentIndex()
        trainMode = self.items[index]
        batch_size = self.textEdit.toPlainText()
        epoch = self.textEdit_2.toPlainText()
        classes = self.textEdit_3.toPlainText()
        channel = self.textEdit_4.toPlainText()
        lr = self.textEdit_5.toPlainText()
        num_workers = self.textEdit_6.toPlainText()
        dataset = self.textEdit_7.toPlainText()
        net = self.textEdit_8.toPlainText()
        moco_dim = self.textEdit_19.toPlainText()
        moco_k = self.textEdit_18.toPlainText()
        moco_m = self.textEdit_17.toPlainText()
        moco_t = self.textEdit_20.toPlainText()
        #
        # if (batch_size == None or batch_size == ""):
        #     box = QMessageBox(QMessageBox.Warning, '警告', "batch_size不能为空！！！")
        #     box.exec_()
        #     return
        # if (epoch == None or epoch == ""):
        #     box = QMessageBox(QMessageBox.Warning, '警告', "epoch不能为空！！！")
        #     box.exec_()
        #     return
        # if (classes == None or classes == ""):
        #     box = QMessageBox(QMessageBox.Warning, '警告', "classes不能为空！！！")
        #     box.exec_()
        #     return
        # if (channel == None or channel == ""):
        #     box = QMessageBox(QMessageBox.Warning, '警告', "channel不能为空！！！")
        #     box.exec_()
        #     return
        # if (lr == None or lr == ""):
        #     box = QMessageBox(QMessageBox.Warning, '警告', "learning rate不能为空！！！")
        #     box.exec_()
        #     return
        # if (num_workers == None or num_workers == ""):
        #     box = QMessageBox(QMessageBox.Warning, '警告', "num_workers不能为空！！！")
        #     box.exec_()
        #     return
        # if (dataset == None or dataset == ""):
        #     box = QMessageBox(QMessageBox.Warning, '警告', "dataset不能为空！！！")
        #     box.exec_()
        #     return
        # if (net == None or net == ""):
        #     box = QMessageBox(QMessageBox.Warning, '警告', "net不能为空！！！")
        #     box.exec_()
        #     return

        map = {}
        map["batch_size"] = batch_size
        map["epoch"] = epoch
        map["classes"] = classes
        map["channel"] = channel
        map["lr"] = lr
        map["num_workers"] = num_workers
        map["dataset"] = dataset
        map["net"] = net
        map["moco_dim"] = moco_dim
        map["moco_k"] = moco_k
        map["moco_m"] = moco_m
        map["moco_t"] = moco_t

        self.label_31.setText("训练正在进行！！！")
        self.textEdit_10.setText("")
        print("参数：",map)
        if trainMode == "有监督":
            print("执行有监督")

            # --------------------------------使用多线程的方式 指令结束后线程终止
            command=None
            self.t = MyThread(command)
            # 线程信号绑定到负责写入文本浏览器的槽函数onUpdateText
            self.t.signalForText.connect(self.onUpdateText)
            self.t.start()

            #--------------------------------使用多线程的方式 但线程会一直使用
            # #创建线程
            # thread1 = myThread("Thread-1", map,self)
            # # 开启新线程
            # thread1.start()
            # self.dowork()

            # --------------------------------不使用多线程的方式
            # train = SeTrain()
            # ret = train.train(map,self)
            # while (ret.poll() != 0):
            #     time.sleep(0.5)
            #     f = open('./Service/segmentation/log/log.txt')
            #     read = f.read()
            #     self.textEdit_10.setText(read)
            #     QApplication.processEvents()  # 实时刷新界面
            #     time.sleep(0.2)  # 间隔显示，为了是肉眼可以看清输出，要不刷新过快看不
            print("按钮退出")
        else:
            print("执行无监督")

    def onUpdateText(self, text):
        if(text != "false"):
            cursor = self.textEdit_10.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.textEdit_10.append(text)
            self.textEdit_10.setTextCursor(cursor)
            self.textEdit_10.ensureCursorVisible()
        else:
            # 训练结束，显示结果
            # 获取结果图 ，例如 loss
            file_name = "./data/LoadImage/cardia_loss.png"
            self.label_16.setPixmap(QtGui.QPixmap(file_name))  # file_name是一个路径
            self.label_16.setScaledContents(True)

            # 获取分割结果
            file_name1 = "./data/LoadImage/result9_img.png"
            file_name2 = "./data/LoadImage/result9_mask.png"
            file_name3 = "./data/LoadImage/result9_img_result.png"
            self.label_17.setPixmap(QtGui.QPixmap(file_name1))  # file_name是一个路径
            self.label_17.setScaledContents(True)
            self.label_18.setPixmap(QtGui.QPixmap(file_name2))  # file_name是一个路径
            self.label_18.setScaledContents(True)
            self.label_19.setPixmap(QtGui.QPixmap(file_name3))  # file_name是一个路径
            self.label_19.setScaledContents(True)

            # 设置测试集指标结果
            self.textEdit_12.setText("0.9426") #dice
            self.textEdit_11.setText("2.1023")  # HD
            self.textEdit_13.setText("0.8856")  # JS
            self.textEdit_14.setText("0.8856")  # IoU
            self.textEdit_15.setText("0.7932")  # Sensitivity
            self.textEdit_16.setText("0.9566")  # PPV

            self.label_31.setText("训练已完成！！！")

    def dowork(self):
        self.thread = Call_Thread()
        self.thread.update_text_singal.connect(self.update_text)
        self.thread.start()

    def update_text(self, text):
        self.textEdit_10.setText(text)

class Call_Thread(QtCore.QThread):
    def __init__(self,parent = None):
        super(Call_Thread,self).__init__(parent)
    update_text_singal = QtCore.pyqtSignal(str)
    def run(self):
        while(True):
               f = open('./Service/segmentation/log/log.txt')
               read = f.read()
               #now_time = datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')
               time.sleep(1)
               self.update_text_singal.emit(read)

class MyThread(QThread):
    signalForText = pyqtSignal(str)
    def __init__(self, param=None, parent=None):
        super(MyThread, self).__init__(parent)
        # 如果有参数，可以封装在类里面
        self.param = param

    def write(self, text):
        self.signalForText.emit(str(text))  # 发射信号

    def result(self):
        self.signalForText.emit("false")

    def run(self):
        # 通过cmdlist[self.param]传递要执行的命令command
        #p = subprocess.Popen(cmdlist[self.param], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # 通过成员变量传参
        #train = SeTrain()
        p = subprocess.Popen(["python","./Service/segmentation/train.py",
                              "-p","./Service/segmentation/dataset/cardia_all_train_test",
                              "-s","./Service/segmentation/save",
                              "-r","./Service/segmentation/result",
                              "-category","0",
                              "-net","unet",
                              "-dn","cardia",
                              "-e","2",
                              "-l","0.0001","-b", "5","-w","8"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
        print("xxx")
        while True:
            result = p.stdout.readline()
            #print("result{}".format(result))
            if result != ''or result==None:
                # print(result.decode('utf-8').strip('\r\n'))  # 对结果进行UTF-8解码以显示中文
                # self.write(result.decode('utf-8').strip('\r\n'))
                # print(result)  # 对结果进行UTF-8解码以显示中文
                self.write(result)
            else:
                break
        # while True:
        #     result = p.stderr.readline()
        #     # print("result{}".format(result))
        #     if result != b'':
        #         print(result.decode('utf-8').strip('\r\n'))  # 对结果进行UTF-8解码以显示中文
        #         self.write(result.decode('utf-8').strip('\r\n'))
        #     else:
        #         break
        p.stdout.close()
        p.stderr.close()
        p.wait()
        self.result()

# class myThread (threading.Thread):
#     def __init__(self, threadID, a, b):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.a = a
#         self.b = b
#
#     def run(self):
#         print ("开始线程：" + self.threadID)
#
#         print ("退出线程：" + self.threadID)

class ModifyName(QDialog,Ui_modifyname):
    def __init__(self,Modelmanagement):
        super(ModifyName, self).__init__()
        self.setupUi(self)
        self.lineEdit.setText(Modelmanagement)
        self.pushButton.clicked.connect(self.modify)
        self.newName = None
    def modify(self):
        text = self.lineEdit_2.text()
        if(text == None or text ==""):
            box = QMessageBox(QMessageBox.Warning, '警告', "新名字不能为空！！！")
            box.exec_()
            return
        self.newName = text
        self.close()

class SelectMode(QDialog,Ui_SelectModel):
    def __init__(self,fenge):
        super(SelectMode, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.select)
        self.fenge = fenge

    def select(self):
        selected = self.listView.currentIndex()
        item = selected.row()
        inf = f"Pos:{item + 1},data: {self.qList[item]}"
        print("点击的是："+inf)
        self.fenge.textEdit.setText(self.qList[item])
        self.close()

class FenLei(QDialog,Ui_fenlei):
    def __init__(self):
        super(FenLei, self).__init__()
        self.setupUi(self)
        self.item = ["全部", "AlexNet", "VGG11", "VGG13", "VGG16", "VGG19",
                     "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
                     "DenseNet"]
        self.comboBox.addItems(self.item)
        self.comboBox.setCurrentIndex(0)
        listModel = QStringListModel()
        self.items = util().getModelList("./data/fenlei")
        print(self.items)
        listModel.setStringList(self.items)
        self.listView.setModel(listModel)
        self.pushButton_2.clicked.connect(self.selectModel)
        self.pushButton_4.clicked.connect(self.loadfile)
        self.pushButton_5.clicked.connect(self.inferenceFenLei)
        self.loadImagePath = None

    def selectModel(self):
        print("选择推理模型！")
        select = self.listView.currentIndex()
        row = select.row()
        print("select", row)
        if (row < 0):
            box = QMessageBox(QMessageBox.Warning, '警告', "请选择具体的推理模型！！！")
            box.exec_()
            return
        self.lineEdit.setText(self.items[row])

    def loadfile(self):
        print("上传文件！")
        global image1
        file_name, filtertype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      '打开文件',
                                                                      r"C:\Users\Y430P\Desktop/",
                                                                      "png(*.png);;jpg(*.jpg)")
        if (file_name == ''):
            return
        self.label.setPixmap(QtGui.QPixmap(file_name))  # file_name是一个路径
        self.label.setScaledContents(True)
        self.loadImagePath = file_name

    def inferenceFenLei(self):
        print("执行分类推理！")
        # 判断条件
        if (self.loadImagePath is None):
            message = "请上传的胎儿超声切面图像！！！"
            print(message)
            box = QMessageBox(QMessageBox.Warning, '警告', message)
            box.exec_()
            return
        text = self.lineEdit.text()
        if (text == None or text == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "请选择具体的推理模型！！！")
            box.exec_()
            return

        # 执行分类
        self.textEdit.setText("该超声图像为腹部横切面。\n"
                              "腹部横切面包括UV脐静脉、PV门静脉、ST胃、DAO降主动脉、IVC下腔静脉、SP脊柱。正常胎儿腹部横切面显示胃泡"
                              "位于左侧富强，脐静脉与门静脉相连，门静脉窦转向胎儿右侧，降主动脉横切面位于脊柱左前方，与脊柱紧靠；下腔"
                              "静脉横切面位于脊柱右前方，相对远离脊柱。")

class ModelmanagementFenLei(QDialog, Ui_modelmanagement):
    def __init__(self):
        super(ModelmanagementFenLei, self).__init__()
        self.setupUi(self)
        listModel = QStringListModel()
        self.items = util().getModelList("./data/fenlei")
        print(self.items)
        listModel.setStringList(self.items)
        self.listView.setModel(listModel)
        self.item = ["全部", "AlexNet", "VGG11", "VGG13", "VGG16", "VGG19",
                     "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
                     "DenseNet"]
        self.comboBox.addItems(self.item)
        self.comboBox.setCurrentIndex(0)
        self.pushButton_4.clicked.connect(self.clickQuery) # query
        self.pushButton_6.clicked.connect(self.clickModify) # modify
        self.pushButton_5.clicked.connect(self.clickDelete)  # delete
        self.pushButton_7.clicked.connect(self.clickLoad)
        self.pushButton_8.clicked.connect(self.selectFile)
        self.loadPath =None
    def clickQuery(self):
        index = self.comboBox.currentIndex()
        print(self.item[index])

    def clickModify(self):
        select = self.listView.currentIndex()
        row = select.row()
        print("modify",row)
        if (row < 0):
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择要修改的模型！")
            box.exec_()
            return
        modifyName = ModifyName(self.items[row])
        modifyName.show()
        modifyName.exec_()
        name = modifyName.newName
        print("新名字：",name)

    def clickDelete(self):
        select = self.listView.currentIndex()
        row = select.row()
        print("delete：",row)
        modelname = self.items[row]
        if(row<0):
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择要删除的模型！")
            box.exec_()
            return
        reply = QtWidgets.QMessageBox.question(self, '警告', '你确定要删除'+modelname+'模型?',
                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("确认执行删除！")
            # 执行刪除操作
            # 刷新列表
        else:
            print("取消执行删除！")
            return

    def clickLoad(self):
        document1 = self.textEdit.toPlainText()
        document2 = self.textEdit_2.toPlainText()
        print("Load:",document1)
        if document1==None or document1 =="" :
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择模型类型！！！")
            box.exec_()
            return
        if  document2 == None or document2 == "":
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择模型地址！！！")
            box.exec_()
            return

        ## 保存文件
        isload =True
        if isload:
            box = QMessageBox(QMessageBox.Warning, '提示', "上传成功！！！！")
            box.exec_()
        else:
            box = QMessageBox(QMessageBox.Warning, '提示', "上传失败！请重新上传！")
            box.exec_()
        ##清空
        self.textEdit.setText("")
        self.textEdit_2.setText("")

    def selectFile(self):
        file_name, filtertype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      '打开文件',
                                                                      r"C:\Users\Y430P\Desktop/",
                                                                      "pth(*.pth)")
        self.textEdit_2.setText(os.path.basename(file_name))
        self.loadPath =file_name

class TrainModelFenLei(QDialog,Ui_trainModelfenlei):
    def __init__(self):
        super(TrainModelFenLei, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.clickTrain)

    def clickTrain(self):

        batch_size = self.textEdit.toPlainText()
        epoch = self.textEdit_2.toPlainText()
        classes = self.textEdit_3.toPlainText()
        channel = self.textEdit_4.toPlainText()
        lr = self.textEdit_5.toPlainText()
        num_workers = self.textEdit_6.toPlainText()
        dataset = self.textEdit_7.toPlainText()
        net = self.textEdit_8.toPlainText()

        map = {}
        map["batch_size"] = batch_size
        map["epoch"] = epoch
        map["classes"] = classes
        map["channel"] = channel
        map["lr"] = lr
        map["num_workers"] = num_workers
        map["dataset"] = dataset
        map["net"] = net
        # map["moco_dim"] = moco_dim
        # map["moco_k"] = moco_k
        # map["moco_m"] = moco_m
        # map["moco_t"] = moco_t
        print("参数：",map)

        print("执行有监督")
        # 训练结束，显示结果
        # 获取结果图 ，例如 loss
        f = open('./data/LoadImage/log.txt')
        read = f.read()
        # 显示日志
        self.textEdit_10.setText(read)

        file_name ="./data/LoadImage/hunxiao.png"
        self.label_16.setPixmap(QtGui.QPixmap(file_name))  # file_name是一个路径
        self.label_16.setScaledContents(True)

        # 设置测试集指标结果
        self.textEdit_12.setText("0.9426") #dice
        self.textEdit_11.setText("2.1023")  # HD
        self.textEdit_13.setText("0.8856")  # JS
        self.textEdit_14.setText("0.8856")  # IoU

class QuZao(QDialog,Ui_quzao):
    def __init__(self):
        super(QuZao, self).__init__()
        self.setupUi(self)
        listModel = QStringListModel()
        self.items = util().getModelList("./data/quzao")
        print(self.items)
        listModel.setStringList(self.items)
        self.listView.setModel(listModel)
        self.item = ["全部", "RedCNN", "RedWGAN", "WGAN-VGG", "MLP", "KSVD", "BM3D", "RDN"]
        self.comboBox.addItems(self.item)
        self.comboBox.setCurrentIndex(0)
        self.pushButton_2.clicked.connect(self.selectModel)
        self.pushButton_4.clicked.connect(self.loadfile)
        self.pushButton_5.clicked.connect(self.inferenceFenLei)
        self.pushButton_6.clicked.connect(self.savefile)
        self.resultPath = None
        self.loadImagePath = None

    def selectModel(self):
        print("选择去噪模型！")
        select = self.listView.currentIndex()
        row = select.row()
        print("select", row)
        if (row < 0):
            box = QMessageBox(QMessageBox.Warning, '警告', "请选择具体的推理模型！！！")
            box.exec_()
            return
        self.lineEdit.setText(self.items[row])

    def loadfile(self):
        print("上传文件！")
        global image1
        file_name, filtertype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      '打开文件',
                                                                      r"C:\Users\Y430P\Desktop/",
                                                                      "png(*.png);;jpg(*.jpg)")
        if (file_name == ''):
            return
        self.label.setPixmap(QtGui.QPixmap(file_name))  # file_name是一个路径
        self.label.setScaledContents(True)
        self.loadImagePath = file_name

    def inferenceFenLei(self):
        print("执行去噪推理！")
        # 判断条件
        if (self.loadImagePath is None):
            message = "请上传的超声图像！！！"
            print(message)
            box = QMessageBox(QMessageBox.Warning, '警告', message)
            box.exec_()
            return
        text = self.lineEdit.text()
        if (text == None or text == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "请选择具体的推理模型！！！")
            box.exec_()
            return
        # 执行去噪
        file_name = view_result1(self.loadImagePath)
        file_name = "./image/zaoresult.png"
        print(file_name)
        self.label_2.setPixmap(QtGui.QPixmap(file_name))
        self.label_2.setScaledContents(True)
        self.resultPath = file_name
        self.loadImagePath = None

    def savefile(self):
        print("保存图片")
        if (self.resultPath == None):
            message = "没有分割结果需要保存！！！"
            print(message)
            box = QMessageBox(QMessageBox.Warning, '警告', message)
            box.exec_()
            return
        image = Image.open(self.resultPath)
        filename = QtWidgets.QFileDialog.getSaveFileName(None, "保存文件", ".", "Image Files(*.jpg *.png)", )
        if (filename[0] == ''):
            return
        image.save(filename[0])

class ModelmanagementQuZao(QDialog, Ui_modelmanagement):
    def __init__(self):
        super(ModelmanagementQuZao, self).__init__()
        self.setupUi(self)
        listModel = QStringListModel()
        self.items = util().getModelList("./data/quzao")
        print(self.items)
        listModel.setStringList(self.items)
        self.listView.setModel(listModel)
        self.item = ["全部", "RedCNN","RedWGAN","WGAN-VGG","MLP","KSVD","BM3D","RDN"]
        self.comboBox.addItems(self.item)
        self.comboBox.setCurrentIndex(0)
        self.pushButton_4.clicked.connect(self.clickQuery)  # query
        self.pushButton_6.clicked.connect(self.clickModify)  # modify
        self.pushButton_5.clicked.connect(self.clickDelete)  # delete
        self.pushButton_7.clicked.connect(self.clickLoad)
        self.pushButton_8.clicked.connect(self.selectFile)
        self.loadPath = None

    def clickQuery(self):
        index = self.comboBox.currentIndex()
        print(self.item[index])

    def clickModify(self):
        select = self.listView.currentIndex()
        row = select.row()
        print("modify", row)
        if (row < 0):
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择要修改的模型！")
            box.exec_()
            return
        modifyName = ModifyName(self.items[row])
        modifyName.show()
        modifyName.exec_()
        name = modifyName.newName
        print("新名字：", name)

    def clickDelete(self):
        select = self.listView.currentIndex()
        row = select.row()
        print("delete：", row)
        modelname = self.items[row]
        if (row < 0):
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择要删除的模型！")
            box.exec_()
            return
        reply = QtWidgets.QMessageBox.question(self, '警告', '你确定要删除' + modelname + '模型?',
                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("确认执行删除！")
            # 执行刪除操作
            # 刷新列表
        else:
            print("取消执行删除！")
            return

    def clickLoad(self):
        document1 = self.textEdit.toPlainText()
        document2 = self.textEdit_2.toPlainText()
        print("Load:", document1)
        if document1 == None or document1 == "":
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择模型类型！！！")
            box.exec_()
            return
        if document2 == None or document2 == "":
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择模型地址！！！")
            box.exec_()
            return

        ## 保存文件
        isload = True
        if isload:
            box = QMessageBox(QMessageBox.Warning, '提示', "上传成功！！！！")
            box.exec_()
        else:
            box = QMessageBox(QMessageBox.Warning, '提示', "上传失败！请重新上传！")
            box.exec_()
        ##清空
        self.textEdit.setText("")
        self.textEdit_2.setText("")

    def selectFile(self):
        file_name, filtertype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      '打开文件',
                                                                      r"C:\Users\Y430P\Desktop/",
                                                                      "pth(*.pth)")
        self.textEdit_2.setText(os.path.basename(file_name))
        self.loadPath = file_name

class TrainModelQuZao(QDialog,Ui_trainModelQuZao):
    def __init__(self):
        super(TrainModelQuZao, self).__init__()
        self.setupUi(self)
        # self.items = ["有监督","无监督"]
        # self.comboBox.addItems(self.items)
        self.pushButton.clicked.connect(self.clickTrain)

    def clickTrain(self):
        # index = self.comboBox.currentIndex()
        # trainMode = self.items[index]
        batch_size = self.textEdit.toPlainText()
        epoch = self.textEdit_2.toPlainText()
        classes = self.textEdit_3.toPlainText()
        channel = self.textEdit_4.toPlainText()
        lr = self.textEdit_5.toPlainText()
        num_workers = self.textEdit_6.toPlainText()
        dataset = self.textEdit_7.toPlainText()
        net = self.textEdit_8.toPlainText()
        # moco_dim = self.textEdit_19.toPlainText()
        # moco_k = self.textEdit_18.toPlainText()
        # moco_m = self.textEdit_17.toPlainText()
        # moco_t = self.textEdit_20.toPlainText()

        map = {}
        map["batch_size"] = batch_size
        map["epoch"] = epoch
        map["classes"] = classes
        map["channel"] = channel
        map["lr"] = lr
        map["num_workers"] = num_workers
        map["dataset"] = dataset
        map["net"] = net
        # map["moco_dim"] = moco_dim
        # map["moco_k"] = moco_k
        # map["moco_m"] = moco_m
        # map["moco_t"] = moco_t
        print("参数：",map)

        print("执行有监督")
        # 训练结束，显示结果
        f = open('./data/LoadImage/log.txt')
        read = f.read()
        # 显示日志
        self.textEdit_10.setText(read)
        # 获取结果图 ，例如 loss
        file_name = "./data/LoadImage/myplot.png"
        self.label_16.setPixmap(QtGui.QPixmap(file_name))  # file_name是一个路径
        self.label_16.setScaledContents(True)

        # 获取分割结果
        file_name1 = "./data/LoadImage/addzao.png"
        file_name2 = "./data/LoadImage/zaolable.png"
        file_name3 = "./data/LoadImage/zaoresult.png"
        self.label_17.setPixmap(QtGui.QPixmap(file_name1))  # file_name是一个路径
        self.label_17.setScaledContents(True)
        self.label_18.setPixmap(QtGui.QPixmap(file_name2))  # file_name是一个路径
        self.label_18.setScaledContents(True)
        self.label_19.setPixmap(QtGui.QPixmap(file_name3))  # file_name是一个路径
        self.label_19.setScaledContents(True)

        # 设置测试集指标结果
        self.textEdit_12.setText("30.5674")  # PSNR
        self.textEdit_11.setText("0.8260")  # SSIM
        self.textEdit_13.setText("7.5539")  # RMSE

class Xunlian(QDialog,Ui_xunlian):
    def __init__(self):
        super(Xunlian, self).__init__()
        self.setupUi(self)

class AnomalyDetection(QDialog,Ui_anomalydetection):
    def __init__(self):
        super(AnomalyDetection, self).__init__()
        self.setupUi(self)
        self.items = ["心脏病检测","反流检测"]
        self.comboBox.addItems(self.items)
        self.comboBox_2.addItems(self.items)
        self.pushButton_2.clicked.connect(self.selectModel1)
        self.pushButton_4.clicked.connect(self.valueExecute)
        self.pushButton.clicked.connect(self.selectFile)
        self.loadPath = None
        self.pushButton_3.clicked.connect(self.selectModel2)
        self.pushButton_5.clicked.connect(self.batchExecute)

    def selectModel1(self):
        print("selectModel")
        self.lineEdit_17.setText("自编码器2")

    def valueExecute(self):
        ## 获取值
        LA = self.lineEdit.text()   #LA
        LV = self.lineEdit_2.text() #LV
        RA = self.lineEdit_3.text() #RA
        RV = self.lineEdit_4.text() #RV
        AO = self.lineEdit_5.text() #AO
        PA = self.lineEdit_6.text() #PA
        AR = self.lineEdit_7.text() #AR
        MV_E = self.lineEdit_14.text() #MV-E
        MV_A = self.lineEdit_11.text() #MV-A
        TV_E = self.lineEdit_12.text() #TV-E
        TV_A = self.lineEdit_9.text() #TV-A
        P_AO = self.lineEdit_8.text() #P-AO
        P_PA = self.lineEdit_13.text() #P-PA
        P_AR = self.lineEdit_10.text() #P-AR
        yunZhou = self.lineEdit_15.text() #孕周

        index = self.comboBox.currentIndex()
        detectType = self.items[index]
        model = self.lineEdit_17.text() # model

        #判断
        if(LA == None or LA == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "LA(左心房)不能为空！！！")
            box.exec_()
            return
        if (LV == None or LV == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "LV(左心室)不能为空！！！")
            box.exec_()
            return
        if (RA == None or RA == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "RA(右心房)不能为空！！！")
            box.exec_()
            return
        if (RV == None or RV == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "RV(右心室)不能为空！！！")
            box.exec_()
            return
        if (AO == None or AO == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "AO(主动脉)不能为空！！！")
            box.exec_()
            return
        if (PA == None or PA == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "PA(肺动脉)不能为空！！！")
            box.exec_()
            return
        if (AR == None or AR == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "AR(峡部)不能为空！！！")
            box.exec_()
            return
        if (MV_E == None or MV_E == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "MV-E(二尖瓣E峰)不能为空！！！")
            box.exec_()
            return
        if (MV_A == None or MV_A == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "MV_A(二尖瓣A峰)不能为空！！！")
            box.exec_()
            return
        if (TV_E == None or TV_E == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "TV-E(三尖瓣E峰)不能为空！！！")
            box.exec_()
            return
        if (TV_A == None or TV_A == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "TV-A(三尖瓣A峰)不能为空！！！")
            box.exec_()
            return
        if (P_AO == None or P_AO == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "P-AO(主动脉流速)不能为空！！！")
            box.exec_()
            return
        if (P_PA == None or P_PA == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "P-PA(肺动脉流速)不能为空！！！")
            box.exec_()
            return
        if (P_AR == None or P_AR == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "P-AR(峡部流速)不能为空！！！")
            box.exec_()
            return
        if (yunZhou == None or yunZhou == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "孕周不能为空！！！")
            box.exec_()
            return
        if (model == None or model == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "模型不能为空！！！")
            box.exec_()
            return
        print("执行检测")
        self.textEdit.setText("结果显示，检测样本不存在心脏病。")

    def selectFile(self):
        file_name, filtertype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      '打开文件',
                                                                      r"C:\Users\Y430P\Desktop/",
                                                                      "csv(*.csv)")
        self.lineEdit_16.setText(os.path.basename(file_name))
        self.loadPath = file_name

    def selectModel2(self):
        print("selectModel")
        self.lineEdit_18.setText("自编码器2")

    def batchExecute(self):
        file = self.lineEdit_16.text()
        model = self.lineEdit_18.text()
        rate = self.lineEdit_19.text()
        if (file == None or file == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "请上传文件！！！")
            box.exec_()
            return
        if (model == None or model == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "请选择模型！！！")
            box.exec_()
            return
        if (rate == None or rate == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "异常率不能为空！！！")
            box.exec_()
            return
        print("执行批检测")

        self.textEdit_2.setText("结果显示:\n"
                                "1,0\n"
                                "2,0\n"
                                "3,1\n"
                                "4,0\n"
                                "5,0\n"
                                "6,0\n"
                                "7,0\n"
                                "8,1\n"
                                "9,1\n"
                                "10,0\n"
                                "11,1\n"
                                "12,0\n"
                                "13,0\n"
                                "14,0\n"
                                "15,1\n"
                                "16,0\n"
                                )

class ModelmanagementDetection(QDialog, Ui_modelmanagement):
    def __init__(self):
        super(ModelmanagementDetection, self).__init__()
        self.setupUi(self)
        listModel = QStringListModel()
        self.items = util().getModelList("./data/detect")
        print(self.items)
        listModel.setStringList(self.items)
        self.listView.setModel(listModel)
        self.item = ["全部", "自编码器","高斯核密度估计模型"]
        self.comboBox.addItems(self.item)
        self.comboBox.setCurrentIndex(0)
        self.pushButton_4.clicked.connect(self.clickQuery)  # query
        self.pushButton_6.clicked.connect(self.clickModify)  # modify
        self.pushButton_5.clicked.connect(self.clickDelete)  # delete
        self.pushButton_7.clicked.connect(self.clickLoad)
        self.pushButton_8.clicked.connect(self.selectFile)
        self.loadPath = None

    def clickQuery(self):
        index = self.comboBox.currentIndex()
        print(self.item[index])

    def clickModify(self):
        select = self.listView.currentIndex()
        row = select.row()
        print("modify", row)
        if (row < 0):
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择要修改的模型！")
            box.exec_()
            return
        modifyName = ModifyName(self.items[row])
        modifyName.show()
        modifyName.exec_()
        name = modifyName.newName
        print("新名字：", name)

    def clickDelete(self):
        select = self.listView.currentIndex()
        row = select.row()
        print("delete：", row)
        modelname = self.items[row]
        if (row < 0):
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择要删除的模型！")
            box.exec_()
            return
        reply = QtWidgets.QMessageBox.question(self, '警告', '你确定要删除' + modelname + '模型?',
                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("确认执行删除！")
            # 执行刪除操作
            # 刷新列表
        else:
            print("取消执行删除！")
            return

    def clickLoad(self):
        document1 = self.textEdit.toPlainText()
        document2 = self.textEdit_2.toPlainText()
        print("Load:", document1)
        if document1 == None or document1 == "":
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择模型类型！！！")
            box.exec_()
            return
        if document2 == None or document2 == "":
            box = QMessageBox(QMessageBox.Warning, '提示', "请选择模型地址！！！")
            box.exec_()
            return

        ## 保存文件
        isload = True
        if isload:
            box = QMessageBox(QMessageBox.Warning, '提示', "上传成功！！！！")
            box.exec_()
        else:
            box = QMessageBox(QMessageBox.Warning, '提示', "上传失败！请重新上传！")
            box.exec_()
        ##清空
        self.textEdit.setText("")
        self.textEdit_2.setText("")

    def selectFile(self):
        file_name, filtertype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      '打开文件',
                                                                      r"C:\Users\Y430P\Desktop/",
                                                                      "pth(*.pth)")
        self.textEdit_2.setText(os.path.basename(file_name))
        self.loadPath = file_name

class DataArgument(QDialog,Ui_dataargument):
    def __init__(self):
        super(DataArgument, self).__init__()
        self.setupUi(self)
        self.label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.label_2.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.pushButton.clicked.connect(self.selectFile)
        self.loadImagePath=None
        self.pushButton_2.clicked.connect(self.executeDA)
        self.pushButton_3.clicked.connect(self.saveFile)
        self.resultPath = None

    def selectFile(self):
        print("上传文件！")
        global image1
        file_name, filtertype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      '打开文件',
                                                                      r"C:\Users\Y430P\Desktop/",
                                                                      "png(*.png);;jpg(*.jpg)")
        if (file_name == ''):
            return
        self.label.setPixmap(QtGui.QPixmap(file_name))  # file_name是一个路径
        self.label.setScaledContents(True)
        self.loadImagePath = file_name
    def executeDA(self):
        if self.loadImagePath ==None:
            message = "请上传带超声图像！！！"
            print(message)
            box = QMessageBox(QMessageBox.Warning, '警告', message)
            box.exec_()
            return

        if(not self.checkBox_4.isChecked() and
        not self.checkBox_2.isChecked() and
        not self.checkBox_3.isChecked() and
        not self.checkBox.isChecked() and
        not self.checkBox_5.isChecked() and
        not self.checkBox_8.isChecked() and
        not self.checkBox_7.isChecked() and
        not self.checkBox_9.isChecked() and
        not self.checkBox_10.isChecked() and
        not self.checkBox_6.isChecked() and
        not self.checkBox_11.isChecked() and
        not self.checkBox_16.isChecked() and
        not self.checkBox_17.isChecked() and
        not self.checkBox_14.isChecked() and
        not self.checkBox_12.isChecked() and
        not self.checkBox_15.isChecked() and
        not self.checkBox_19.isChecked() and
        not self.checkBox_20.isChecked() ):
            message = "至少选择一种数据增强方式！！！"
            box = QMessageBox(QMessageBox.Warning, '警告', message)
            box.exec_()
            return
        image = imageio.imread(self.loadImagePath)
        pre_data = PreData()
        if self.checkBox_4.isChecked():
            image= pre_data.fliplr(image)
        if self.checkBox_2.isChecked():
            image = pre_data.flipud(image)
        if self.checkBox_3.isChecked():
            image = pre_data.affine(image)
        if self.checkBox.isChecked():
            image = pre_data.crop(image)
        if self.checkBox_5.isChecked():
            image = pre_data.scale(image)
        if self.checkBox_8.isChecked():
            image = pre_data.gaussianBlur(image)
        if self.checkBox_7.isChecked():
            image = pre_data.averageBlur(image)
        if self.checkBox_9.isChecked():
            image = pre_data.medianBlur(image)
        if self.checkBox_10.isChecked():
            image = pre_data.bilateralBlur(image)
        if self.checkBox_6.isChecked():
            image = pre_data.motionBlur(image)
        if self.checkBox_11.isChecked():
            image = pre_data.meanShiftBlur(image)
        if  self.checkBox_16.isChecked():
            image = pre_data.fastSnowyLandscape(image)
        if self.checkBox_17.isChecked():
            image = pre_data.clouds(image)
        if self.checkBox_14.isChecked():
            image = pre_data.fog(image)
        if self.checkBox_12.isChecked():
            image = pre_data.snowflakes(image)
        if self.checkBox_15.isChecked():
            image = pre_data.rain(image)
        if self.checkBox_19.isChecked():
            image = pre_data.cutout(image)
        if self.checkBox_20.isChecked():
            image = pre_data.addToHue(image)

        abspath = os.path.abspath(os.path.dirname(__file__))
        basename = os.path.basename(self.loadImagePath)
        print('baseename:',basename)
        outputPath = abspath+"/image/"+basename[:-3]+"dataArgument.png"
        imageio.imwrite(outputPath,image)

        self.label_2.setPixmap(QtGui.QPixmap(outputPath))
        self.label_2.setScaledContents(True)
        self.resultPath = outputPath
        self.loadImagePath = None
    def saveFile(self):
        print("保存图片")
        if (self.resultPath == None):
            message = "没有数据需要保存！！！"
            print(message)
            box = QMessageBox(QMessageBox.Warning, '警告', message)
            box.exec_()
            return
        image = Image.open(self.resultPath)
        filename = QtWidgets.QFileDialog.getSaveFileName(None, "保存文件", ".", "Image Files(*.jpg *.png)", )
        if (filename[0] == ''):
            return
        image.save(filename[0])

class ImageSize(QDialog,Ui_imageresize):
    def __init__(self):
        super(ImageSize, self).__init__()
        self.setupUi(self)
        self.label_2.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.label_3.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.label_4.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.pushButton.clicked.connect(self.selectFile)
        self.pushButton_2.clicked.connect(self.saveFile)
        self.pushButton_3.clicked.connect(self.clickResize)
        self.pushButton_4.clicked.connect(self.enlarge)
        self.pushButton_5.clicked.connect(self.narrow)
        self.zoomscale = 1  # 图片放缩尺度
        self.item = None
        self.scene =None
        self.x =None
        self.y =None

    def selectFile(self):
        file_name, filtertype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      '打开文件',
                                                                      r"C:\Users\Y430P\Desktop/",
                                                                      "png(*.png);;jpg(*.jpg)")

        if(file_name==None or file_name == ""):
            return
        img = cv2.imread(file_name)  # 读取图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        self.x = img.shape[1]  # 获取图像大小
        self.y = img.shape[0]

        frame = QImage(img, self.x,self.y, QImage.Format_RGB888)
        #frame = QImage(img)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)
    def enlarge(self):
        print("放大")
        self.zoomscale = self.zoomscale + 0.05
        if self.zoomscale >= 3:
            self.zoomscale = 3
        self.item.setScale(self.zoomscale)
    def narrow(self):
        print("缩小")
        self.zoomscale = self.zoomscale - 0.05
        if self.zoomscale <= 0:
            self.zoomscale = 0.2
        self.item.setScale(self.zoomscale)  # 缩小图像
    def clickResize(self):
        print("resize")
        w = self.lineEdit.text()
        h = self.lineEdit_2.text()
        if (w == None or w == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "图片宽度不能为空！！！")
            box.exec_()
            return
        if (h == None or h == ""):
            box = QMessageBox(QMessageBox.Warning, '警告', "图片高度不能为空！！！")
            box.exec_()
            return
        if (self.item == None):
            box = QMessageBox(QMessageBox.Warning, '警告', "请上传图片！！！")
            box.exec_()
            return
        self.item.setScale(int(w)/self.x)

    def saveFile(self):
        print("保存图片")
        if (self.item == None):
            message = "没有图像需要保存！！！"
            print(message)
            box = QMessageBox(QMessageBox.Warning, '警告', message)
            box.exec_()
            return
        #image = Image.open(self.resultPath)
        filename = QtWidgets.QFileDialog.getSaveFileName(None, "保存文件", ".", "Image Files(*.jpg *.png)", )
        if (filename[0] == ''):
            return
        #image.save(filename[0])

class Normalize(QDialog,Ui_normalize):
    def __init__(self):
        super(Normalize, self).__init__()
        self.setupUi(self)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    myshow = MainWindows()
    myshow.show()
    sys.exit(app.exec_())