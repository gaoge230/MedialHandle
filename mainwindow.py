# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1118, 917)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("image/src=http___img95.699pic.com_xsj_0q_mh_r9.jpg!_fw_700_watermark_url_L3hzai93YXRlcl9kZXRhaWwyLnBuZw_align_southeast&refer=http___img95.699pic.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(6, -2, 151, 51))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 60, 111, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(4, 180, 171, 51))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 250, 111, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 290, 111, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setEnabled(True)
        self.pushButton_5.setGeometry(QtCore.QRect(10, 450, 111, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setDefault(False)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setEnabled(True)
        self.pushButton_6.setGeometry(QtCore.QRect(10, 410, 111, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setDefault(False)
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setEnabled(False)
        self.pushButton_7.setGeometry(QtCore.QRect(280, 930, 91, 31))
        self.pushButton_7.setDefault(False)
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setEnabled(True)
        self.pushButton_8.setGeometry(QtCore.QRect(10, 610, 141, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton_8.setFont(font)
        self.pushButton_8.setDefault(False)
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_9.setEnabled(True)
        self.pushButton_9.setGeometry(QtCore.QRect(10, 570, 141, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton_9.setFont(font)
        self.pushButton_9.setDefault(False)
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_10.setEnabled(False)
        self.pushButton_10.setGeometry(QtCore.QRect(380, 930, 91, 31))
        self.pushButton_10.setDefault(False)
        self.pushButton_10.setObjectName("pushButton_10")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(190, 20, 901, 821))
        self.stackedWidget.setObjectName("stackedWidget")
        self.stackedWidgetPage1 = QtWidgets.QWidget()
        self.stackedWidgetPage1.setObjectName("stackedWidgetPage1")
        self.label_6 = QtWidgets.QLabel(self.stackedWidgetPage1)
        self.label_6.setGeometry(QtCore.QRect(190, 120, 841, 141))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.stackedWidgetPage1)
        self.label_7.setGeometry(QtCore.QRect(20, 50, 841, 141))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.frame = QtWidgets.QFrame(self.stackedWidgetPage1)
        self.frame.setGeometry(QtCore.QRect(-11, -11, 921, 841))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.stackedWidget.addWidget(self.stackedWidgetPage1)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(180, 0, 921, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(170, 9, 21, 841))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.pushButton_12 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_12.setGeometry(QtCore.QRect(10, 100, 111, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton_12.setFont(font)
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_13 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_13.setEnabled(True)
        self.pushButton_13.setGeometry(QtCore.QRect(10, 730, 111, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton_13.setFont(font)
        self.pushButton_13.setDefault(False)
        self.pushButton_13.setObjectName("pushButton_13")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(6, 0, 181, 20))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(-10, 170, 181, 16))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_8 = QtWidgets.QFrame(self.centralwidget)
        self.line_8.setGeometry(QtCore.QRect(1090, 10, 21, 841))
        self.line_8.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.line_9 = QtWidgets.QFrame(self.centralwidget)
        self.line_9.setGeometry(QtCore.QRect(10, 840, 1091, 20))
        self.line_9.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.pushButton_14 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_14.setEnabled(True)
        self.pushButton_14.setGeometry(QtCore.QRect(10, 770, 111, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton_14.setFont(font)
        self.pushButton_14.setDefault(False)
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(180, 930, 91, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(-20, -10, 1141, 891))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("background-color: rgb(206, 235, 255);")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(-10, 340, 181, 16))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(0, 350, 171, 51))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.line_6 = QtWidgets.QFrame(self.centralwidget)
        self.line_6.setGeometry(QtCore.QRect(-10, 500, 181, 16))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(0, 510, 261, 51))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(0, 670, 261, 51))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.line_7 = QtWidgets.QFrame(self.centralwidget)
        self.line_7.setGeometry(QtCore.QRect(-10, 660, 180, 20))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.label_3.raise_()
        self.pushButton_14.raise_()
        self.pushButton_9.raise_()
        self.pushButton_5.raise_()
        self.pushButton_13.raise_()
        self.pushButton_10.raise_()
        self.pushButton_6.raise_()
        self.pushButton_8.raise_()
        self.pushButton_7.raise_()
        self.label.raise_()
        self.pushButton.raise_()
        self.label_2.raise_()
        self.pushButton_2.raise_()
        self.pushButton_3.raise_()
        self.stackedWidget.raise_()
        self.line.raise_()
        self.line_2.raise_()
        self.pushButton_12.raise_()
        self.line_3.raise_()
        self.line_4.raise_()
        self.line_8.raise_()
        self.line_9.raise_()
        self.pushButton_4.raise_()
        self.line_5.raise_()
        self.label_4.raise_()
        self.line_6.raise_()
        self.label_5.raise_()
        self.label_8.raise_()
        self.line_7.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1118, 23))
        self.menubar.setObjectName("menubar")
        self.menu23 = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.menu23.setFont(font)
        self.menu23.setObjectName("menu23")
        self.menu = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.menu.setFont(font)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.menu_2.setFont(font)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.menu_3.setFont(font)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.menu_4.setFont(font)
        self.menu_4.setObjectName("menu_4")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.menu_5.setFont(font)
        self.menu_5.setObjectName("menu_5")
        self.menu_6 = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.menu_6.setFont(font)
        self.menu_6.setObjectName("menu_6")
        self.menu_7 = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.menu_7.setFont(font)
        self.menu_7.setObjectName("menu_7")
        self.menu_8 = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.menu_8.setFont(font)
        self.menu_8.setObjectName("menu_8")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action2 = QtWidgets.QAction(MainWindow)
        self.action2.setObjectName("action2")
        self.action = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action.setFont(font)
        self.action.setObjectName("action")
        self.action_3 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_3.setFont(font)
        self.action_3.setObjectName("action_3")
        self.action_4 = QtWidgets.QAction(MainWindow)
        self.action_4.setObjectName("action_4")
        self.action_5 = QtWidgets.QAction(MainWindow)
        self.action_5.setObjectName("action_5")
        self.action_2 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_2.setFont(font)
        self.action_2.setObjectName("action_2")
        self.action_6 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_6.setFont(font)
        self.action_6.setObjectName("action_6")
        self.action_7 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_7.setFont(font)
        self.action_7.setObjectName("action_7")
        self.action_8 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_8.setFont(font)
        self.action_8.setObjectName("action_8")
        self.action_9 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_9.setFont(font)
        self.action_9.setObjectName("action_9")
        self.action_10 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_10.setFont(font)
        self.action_10.setObjectName("action_10")
        self.action_11 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_11.setFont(font)
        self.action_11.setObjectName("action_11")
        self.action_12 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_12.setFont(font)
        self.action_12.setObjectName("action_12")
        self.action_13 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_13.setFont(font)
        self.action_13.setObjectName("action_13")
        self.action_14 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_14.setFont(font)
        self.action_14.setObjectName("action_14")
        self.action_15 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_15.setFont(font)
        self.action_15.setObjectName("action_15")
        self.action_16 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_16.setFont(font)
        self.action_16.setObjectName("action_16")
        self.action_17 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_17.setFont(font)
        self.action_17.setObjectName("action_17")
        self.action_18 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_18.setFont(font)
        self.action_18.setObjectName("action_18")
        self.action_19 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_19.setFont(font)
        self.action_19.setObjectName("action_19")
        self.action_20 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_20.setFont(font)
        self.action_20.setObjectName("action_20")
        self.action_21 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_21.setFont(font)
        self.action_21.setObjectName("action_21")
        self.action_22 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.action_22.setFont(font)
        self.action_22.setObjectName("action_22")
        self.menu23.addAction(self.action)
        self.menu23.addAction(self.action_3)
        self.menu.addAction(self.action_8)
        self.menu.addAction(self.action_9)
        self.menu.addAction(self.action_10)
        self.menu_2.addAction(self.action_11)
        self.menu_2.addAction(self.action_12)
        self.menu_2.addAction(self.action_13)
        self.menu_3.addAction(self.action_14)
        self.menu_3.addAction(self.action_15)
        self.menu_3.addAction(self.action_16)
        self.menu_4.addAction(self.action_17)
        self.menu_4.addAction(self.action_22)
        self.menu_5.addAction(self.action_18)
        self.menu_5.addAction(self.action_19)
        self.menu_6.addAction(self.action_20)
        self.menu_7.addAction(self.action_2)
        self.menu_7.addAction(self.action_6)
        self.menu_7.addAction(self.action_7)
        self.menu_8.addAction(self.action_21)
        self.menubar.addAction(self.menu_8.menuAction())
        self.menubar.addAction(self.menu_7.menuAction())
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())
        self.menubar.addAction(self.menu_6.menuAction())
        self.menubar.addAction(self.menu23.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "胎儿心脏分割原型系统"))
        self.label.setText(_translate("MainWindow", "图像预处理"))
        self.pushButton.setText(_translate("MainWindow", "尺寸修改"))
        self.label_2.setText(_translate("MainWindow", "超声图像分割"))
        self.pushButton_2.setText(_translate("MainWindow", "胎儿心脏分割"))
        self.pushButton_3.setText(_translate("MainWindow", "分割模型管理"))
        self.pushButton_5.setText(_translate("MainWindow", "分类模型管理"))
        self.pushButton_6.setText(_translate("MainWindow", "超声切面分类"))
        self.pushButton_7.setText(_translate("MainWindow", "分类模型训练"))
        self.pushButton_8.setText(_translate("MainWindow", "报告生成模型管理"))
        self.pushButton_9.setText(_translate("MainWindow", "医学诊断报告生成"))
        self.pushButton_10.setText(_translate("MainWindow", "去噪模型训练"))
        self.label_6.setText(_translate("MainWindow", "胎儿心脏分割原型系统V1.0"))
        self.label_7.setText(_translate("MainWindow", "欢迎使用"))
        self.pushButton_12.setText(_translate("MainWindow", "图像增强"))
        self.pushButton_13.setText(_translate("MainWindow", "异常检测"))
        self.pushButton_14.setText(_translate("MainWindow", "检测模型管理"))
        self.pushButton_4.setText(_translate("MainWindow", "分割模型训练"))
        self.label_4.setText(_translate("MainWindow", "超声图像分类"))
        self.label_5.setText(_translate("MainWindow", "诊断报告生成"))
        self.label_8.setText(_translate("MainWindow", "胎儿异常检测"))
        self.menu23.setTitle(_translate("MainWindow", "帮助"))
        self.menu.setTitle(_translate("MainWindow", "超声图像分割"))
        self.menu_2.setTitle(_translate("MainWindow", "超声切面分类"))
        self.menu_3.setTitle(_translate("MainWindow", "诊断报告生成"))
        self.menu_4.setTitle(_translate("MainWindow", "胎儿异常检测"))
        self.menu_5.setTitle(_translate("MainWindow", "用户管理"))
        self.menu_6.setTitle(_translate("MainWindow", "设置"))
        self.menu_7.setTitle(_translate("MainWindow", "图像预处理"))
        self.menu_8.setTitle(_translate("MainWindow", "主页"))
        self.action2.setText(_translate("MainWindow", "图像分割"))
        self.action.setText(_translate("MainWindow", "关于软件"))
        self.action_3.setText(_translate("MainWindow", "关于作者"))
        self.action_4.setText(_translate("MainWindow", "图像转换"))
        self.action_5.setText(_translate("MainWindow", "颜色迁移"))
        self.action_2.setText(_translate("MainWindow", "尺寸修改"))
        self.action_6.setText(_translate("MainWindow", "图像增强"))
        self.action_7.setText(_translate("MainWindow", "图像归一化"))
        self.action_8.setText(_translate("MainWindow", "胎儿心脏分割"))
        self.action_9.setText(_translate("MainWindow", "分割模型管理"))
        self.action_10.setText(_translate("MainWindow", "分割模型训练"))
        self.action_11.setText(_translate("MainWindow", "超声切面分类"))
        self.action_12.setText(_translate("MainWindow", "分类模型管理"))
        self.action_13.setText(_translate("MainWindow", "分类模型训练"))
        self.action_14.setText(_translate("MainWindow", "医学诊断报告生成"))
        self.action_15.setText(_translate("MainWindow", "报告生成模型管理"))
        self.action_16.setText(_translate("MainWindow", "去噪模型训练"))
        self.action_17.setText(_translate("MainWindow", "异常检测"))
        self.action_18.setText(_translate("MainWindow", "注册"))
        self.action_19.setText(_translate("MainWindow", "个人信息"))
        self.action_20.setText(_translate("MainWindow", "属性"))
        self.action_21.setText(_translate("MainWindow", "主页"))
        self.action_22.setText(_translate("MainWindow", "检测模型管理"))
