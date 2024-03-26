import os
import sys
import time
import tkinter
from tkinter import messagebox

import numpy as np
# import pyqt lib
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtSql import QSqlDatabase
from PyQt5.QtSql import QSqlQuery
from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem, QMenu, QAbstractItemView, QStyleFactory
from PyQt5.QtWidgets import QApplication, QMainWindow
# import matplot lib
from sympy.plotting import plot as symplot
from matplotlib import pyplot as pl
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import datetime
from scipy.io import loadmat
# import user's lib
from NetMain import main
from test2 import Ui_MainWindow


class AppWidget(QMainWindow, Ui_MainWindow):
    """程序界面设定控制类"""

    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        self.setupUi(self)
        # 对Sqlite数据库进行相关初始化
        self.init_db()
        # 调用界面初始化方法（一般会将UI界面的代码封装到另外一个方法函数中，而不直接放到__init__）
        self.init_ui()
        # 加载文件中存储的所有运行信息
        self.load_all_infos()
        self.model_name_list = []

    def init_db(self):
        """对数据库初始化"""
        # 指定要操作的数据库类型
        #成功与SQLite库连接。
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        # 指定要使用的数据库（如果数据库不存在则会建立新的）
        self.db.setDatabaseName("./infos.db")
        # 打开数据库
        if not self.db.open():
            exit("无法建立数据库")
        # 创建一个用来执行sql语句的对象
        self.query = QSqlQuery()
        # 建立数据表

        sql = "create table if not exists person(" \
              "id integer primary key, " \
              "`model_name` varchar(50), " \
              "resume varchar(10), " \
              "data_load_path varchar(50), " \
              "num_receiver varchar(10), " \
              "num_frequency varchar(10), " \
              "batchsize varchar(10), " \
              "batchsize_v varchar(10), " \
              "batchsize_exp varchar(10), " \
              "test_only varchar(10), " \
              "exp_only varchar(10), " \
              "mini_epoch varchar(10), " \
              "max_epoch varchar(10), " \
              "load_model_path varchar(50), " \
              "learning_rate varchar(20), " \
              "save_file_path varchar(100), " \
              "save_file_name varchar(50), " \
              "run_date_time varchar(50), " \
              "note varchar(200))"
        # 这个函数用于执行一个查询语句并返回结果。具体的作用取决于所使用的数据库库和上下文。
        self.query.exec(sql)

    def init_ui(self):
        # 初始化tab_run
        # 添加选择lineEdit_01_model_name
        #在test2中，下面四个量已经提供一个下拉列表供用户选择
        self.lineEdit_01_model_name.addItems(['mtl_cnn', 'mtl_unet', 'mtl_unet_cbam', 'xception', '~all~'])
        self.lineEdit_02_resume.addItems(['False', 'True', ])
        self.lineEdit_03_test_only.addItems(['False', 'True', ])
        self.lineEdit_04_exp_only.addItems(['False', 'True', ])

        # RUN BUTTON 事件绑定
        #8.self.button1.clicked.connect(self.click):是一个将信号与槽链接的方法（method），它的作用是在点击 button 时，执行（self.click）里的方法。
        self.pushButton_01_run.clicked.connect(self.add_new_run_info)
        self.pushButton_04_runM.clicked.connect(self.add_new_run_info)

        # 信息表
        self.tableWidget_01_run_info.setColumnCount(18)
        self.tableWidget_01_run_info.setHorizontalHeaderLabels([
            '模型名称', '重复训练', '数据集路径', '阵元数',
            '频点数', '批次大小(训练)', '批次大小（验证）', '批次大小（实测）',
            '仅测试', '仅实测', '小循环周期', '大循环周期',
            '权重文件路径', '学习率', '保存路径', '保存文件名', '运行时间', '备注',
        ])

        # 禁用双击编辑单元格
        self.tableWidget_01_run_info.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 改为选择一行
        self.tableWidget_01_run_info.setSelectionBehavior(QAbstractItemView.SelectRows)
        # 添加右击菜单
        self.tableWidget_01_run_info.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget_01_run_info.customContextMenuRequested.connect(self.generate_menu)

        # 按类型查找
        # 下拉菜单，查找类型
        find_type = ['模型名称', '重复训练', '数据集路径', '阵元数',
                     '频点数', '批次大小(训练)', '批次大小（验证）', '批次大小（实测）',
                     '仅测试', '仅实测', '小循环周期', '大循环周期',
                     '权重文件路径', '学习率', '保存路径', '保存文件名', '运行日期范围', '备注', ]
        #enumerate将可以遍历的量组合为一个索引序列。
        for i, type_temp in enumerate(find_type):
            self.lineEdit_17_seek_type.addItem("")
            self.lineEdit_17_seek_type.setItemText(i, type_temp)
        # 事件绑定
        self.lineEdit_17_seek_type.currentIndexChanged.connect(self.change_search_type)

        # 查找输入框
        # 查找日期范围（默认不显示，只有当选择查询日期范围时才显示）
        self.line_edit_star_time = QtWidgets.QDateTimeEdit(self.TableFrame)
        # self.line_edit_star_time = QDateTimeEdit(self)
        # self.line_edit_star_time.setGeometry(550, 290, 100, 20)
        self.line_edit_star_time.setCalendarPopup(True)
        self.line_edit_star_time.setDisplayFormat("yyyy-MM-dd")
        self.line_edit_star_time.setVisible(False)
        self.line_edit_end_time = QtWidgets.QDateTimeEdit(self.TableFrame)
        # self.line_edit_end_time = QDateTimeEdit(self)
        # self.line_edit_end_time.setGeometry(655, 290, 100, 20)
        self.line_edit_end_time.setCalendarPopup(True)
        self.line_edit_end_time.setDisplayFormat("yyyy-MM-dd")
        self.line_edit_end_time.setVisible(False)

        # 初始化tab_plot
        # 事件绑定
        #pushButton_02可以看到所有保存的训练模型。
        self.pushButton_02_checkall.clicked.connect(self.get_all_infos)
        # self.plot_button.clicked.connect(self.mapping)
        self.plot_button.clicked.connect(self.plot_from_files)
        self.pushButton_03_seek.clicked.connect(self.search_info_from_files)
        self.lineEdit_19_seek_type.addItems(['数据集路径', '保存路径', '保存文件名', '备注', ])
        # 查询信息表
        #setColumnCount()设置列数
        self.tableWidget_02_plot_info.setColumnCount(5)
        self.tableWidget_02_plot_info.setHorizontalHeaderLabels([
            '模型名称', '数据集路径', '保存路径', '保存文件名', '备注',
        ])
        # 结果分析信息表
        self.tableWidget_03_results.setColumnCount(7)
        self.tableWidget_03_results.setHorizontalHeaderLabels([
            '模型名称', '训练误差-距离', '训练误差-深度', '测试平均准确度-距离', '测试平均准确度-深度',
            '实测平均绝对误差-距离', '实测平均绝对误差-深度',
        ])
        # 事件绑定
        self.pushButton_04_seek.clicked.connect(self.check_search_info_from_files)
        # 使checkBox 不可见
        self.checkBox_01_mtl_cnn.setVisible(False)
        self.checkBox_02_mtl_unet.setVisible(False)
        self.checkBox_03_mtl_unet_cbam.setVisible(False)
        self.checkBox_04_xception.setVisible(False)
        # 事件绑定
        self.pushButton_05_select_all.clicked.connect(self.check_select_all)
        self.pushButton_06_plot.clicked.connect(self.plot_from_files)

        # 初始化绘图区大小
        try:
            train_range_background = QtWidgets.QLabel(self)
            train_range_background.setPixmap(QtGui.QPixmap("./picture/查找学习.png").scaled(628, 518))
            train_depth_background = QtWidgets.QLabel(self)
            train_depth_background.setPixmap(QtGui.QPixmap("./picture/查找学习.png").scaled(628, 518))
            test_range_background = QtWidgets.QLabel(self)
            test_range_background.setPixmap(QtGui.QPixmap("./picture/查找学习.png").scaled(628, 518))
            test_depth_background = QtWidgets.QLabel(self)
            test_depth_background.setPixmap(QtGui.QPixmap("./picture/查找学习.png").scaled(628, 518))
            exp_range_background = QtWidgets.QLabel(self)
            exp_range_background.setPixmap(QtGui.QPixmap("./picture/查找学习.png").scaled(628, 518))
            exp_depth_background = QtWidgets.QLabel(self)
            exp_depth_background.setPixmap(QtGui.QPixmap("./picture/查找学习.png").scaled(628, 518))

            self.train_range_layout.addWidget(train_range_background)
            self.train_depth_layout.addWidget(train_depth_background)
            self.test_range_layout.addWidget(test_range_background)
            self.test_depth_layout.addWidget(test_depth_background)
            self.exp_range_layout.addWidget(exp_range_background)
            self.exp_depth_layout.addWidget(exp_depth_background)
        except:
            pass

    def closeEvent(self, *args):
        """重写此方法，在关闭窗口时要关闭数据库"""
        self.db.close()
        super().closeEvent(*args)

    def check_select_all(self):
        if 'mtl_cnn' in self.model_name_list:
            self.checkBox_01_mtl_cnn.setChecked(True)
        if 'mtl_unet' in self.model_name_list:
            self.checkBox_02_mtl_unet.setChecked(True)
        if 'mtl_unet_cbam' in self.model_name_list:
            self.checkBox_03_mtl_unet_cbam.setChecked(True)
        if 'xception' in self.model_name_list:
            self.checkBox_04_xception.setChecked(True)

    def change_search_type(self):
        """更改搜索类型"""
        # .currentText()可以显示当前文本你内容
        search_type = self.lineEdit_17_seek_type.currentText()
        if search_type == '运行日期范围':
            self.line_edit_star_time.setVisible(True)
            self.line_edit_end_time.setVisible(True)
            self.horizontalLayout.addWidget(self.line_edit_star_time)
            self.horizontalLayout.addWidget(self.line_edit_end_time)
            self.lineEdit_18_seek_info.setVisible(False)
        else:
            self.line_edit_star_time.setVisible(False)
            self.line_edit_end_time.setVisible(False)
            self.lineEdit_18_seek_info.setVisible(True)

    def get_all_infos(self):
        """查询所有人员信息并显示"""
        self.load_all_infos()
        # 清理查询输入框内容
        self.lineEdit_18_seek_info.setText("")

    def search_info_from_files(self):
        # 查询类型
        search_type = self.lineEdit_17_seek_type.currentText()

        if search_type == '运行日期范围':
            start_time = self.line_edit_star_time.text().split('-')
            stop_time = self.line_edit_end_time.text().split('-')
            start_time = datetime.date(int(start_time[0]), int(start_time[1]), int(start_time[2]))
            stop_time = datetime.date(int(stop_time[0]), int(stop_time[1]), int(stop_time[2]))
            if (stop_time - start_time).days < 0:
                QMessageBox.information(self, '提示', '开始日期不能小于结束日期')
                return

        # 获取要搜索的内容
        search_content = self.lineEdit_18_seek_info.text()
        search_type_map = [
            ('模型名称', 'model_name'),
            ('重复训练', 'resume'),
            ('数据集路径', 'data_load_path'),
            ('阵元数', 'num_receiver'),
            ('频点数', 'num_frequency'),
            ('批次大小(训练)', 'batchsize'),
            ('批次大小（验证）', 'batchsize_v'),
            ('批次大小（实测）', 'batchsize_exp'),
            ('仅测试', 'test_only'),
            ('仅实测', 'exp_only'),
            ('小循环周期', 'mini_epoch'),
            ('大循环周期', 'max_epoch'),
            ('权重文件路径', 'load_model_path'),
            ('学习率', 'learning_rate'),
            ('保存路径', 'save_file_path'),
            ('保存文件名', 'save_file_name'),
            ('运行日期范围', 'run_date_time'),
            ('备注', 'note'),
        ]
        for temp in search_type_map:
            if search_type == temp[0]:
                sql = "select count(*) from person where `{}`='{}'".format(temp[1], search_content)
                self.query.exec(sql)
                ##query.next()将指针往后移动一位
                self.query.next()
                self.tableWidget_01_run_info.setRowCount(self.query.value(0))
                sql = "select * from person where `{}`='{}'".format(temp[1], search_content)
                self.query.exec(sql)
                i = 0
                while self.query.next():  # 每次查询一行记录
                    for j in range(1, 19):
                        new_item = QTableWidgetItem(self.query.value(j))
                        new_item.setTextAlignment(Qt.AlignHCenter)
                        self.tableWidget_01_run_info.setItem(i, j - 1, new_item)

                    i += 1
                break
        else:
            if search_type == "运行日期范围":
                sql = "select count(*) from person where birthday>='{}' and birthday<='{}'".format(start_time,
                                                                                                   stop_time)
                self.query.exec(sql)
                self.query.next()
                self.tableWidget_01_run_info.setRowCount(self.query.value(0))
                sql = "select * from person where birthday>='{}' and birthday<='{}'".format(start_time, stop_time)
                self.query.exec(sql)
                i = 0
                while self.query.next():  # 每次查询一行记录
                    for j in range(1, 19):
                        new_item = QTableWidgetItem(self.query.value(j))
                        new_item.setTextAlignment(Qt.AlignHCenter)
                        self.tableWidget_01_run_info.setItem(i, j - 1, new_item)

                    i += 1

    def check_search_info_from_files(self):
        # 查询类型
        search_type = self.lineEdit_19_seek_type.currentText()

        # 获取要搜索的内容
        search_content = self.lineEdit_20_seek_info.text()
        search_type_map = [
            ('数据集路径', 'data_load_path'),
            ('保存路径', 'save_file_path'),
            ('保存文件名', 'save_file_name'),
            ('备注', 'note'),
        ]

        for temp in search_type_map:
            if search_type == temp[0]:
                search_content = search_content.replace("\\", '/')
                sql = "select count(*) from person where `{}`='{}'".format(temp[1], search_content)
                self.query.exec(sql)
                #query.next()将指针往后移动一位
                self.query.next()
                self.tableWidget_02_plot_info.setRowCount(self.query.value(0))
                sql = "select * from person where `{}`='{}'".format(temp[1], search_content)
                self.query.exec(sql)
                i = 0
                col_id = [1, 3, 15, 16, 18]
                while self.query.next():  # 每次查询一行记录
                    for j in range(1, len(col_id) + 1):
                        new_item = QTableWidgetItem(self.query.value(col_id[j - 1]))
                        new_item.setTextAlignment(Qt.AlignHCenter)
                        self.tableWidget_02_plot_info.setItem(i, j - 1, new_item)

                    i += 1
                break
        self.model_name_list = []
        model = self.tableWidget_02_plot_info.model()
        for j in range(0, i):
            model_name = model.data(model.index(j, 0))
            self.model_name_list.append(model_name)
        self.checkBox_01_mtl_cnn.setVisible(False)
        self.checkBox_02_mtl_unet.setVisible(False)
        self.checkBox_03_mtl_unet_cbam.setVisible(False)
        self.checkBox_04_xception.setVisible(False)
        if 'mtl_cnn' in self.model_name_list:
            self.checkBox_01_mtl_cnn.setVisible(True)
        if 'mtl_unet' in self.model_name_list:
            self.checkBox_02_mtl_unet.setVisible(True)
        if 'mtl_unet_cbam' in self.model_name_list:
            self.checkBox_03_mtl_unet_cbam.setVisible(True)
        if 'xception' in self.model_name_list:
            self.checkBox_04_xception.setVisible(True)

    def save_change_info(self):
        """修改数据"""
        model_name = self.lineEdit_01_model_name.currentText()
        resume = self.lineEdit_02_resume.currentText()
        test_only = self.lineEdit_03_test_only.currentText()
        exp_only = self.lineEdit_04_exp_only.currentText()
        num_receiver = self.lineEdit_05_num_receiver.text()
        num_frequency = self.lineEdit_09_num_frequency.text()
        learning_rate = self.lineEdit_13_learning_rate.text()
        load_model_path = self.lineEdit_06_load_model_path.text()
        data_load_path = self.lineEdit_10_data_load_path.text()
        save_file_path = self.lineEdit_14_save_file_path.text()
        batchsize = self.lineEdit_07_batchsize.text()
        batchsize_v = self.lineEdit_11_batchsize_v.text()
        batchsize_exp = self.lineEdit_15_batchsize_exp.text()
        mini_epoch = self.lineEdit_08_mini_epoch.text()
        max_epoch = self.lineEdit_12_max_epoch.text()
        note = self.lineEdit_16_note.toPlainText()

        infos = [model_name, resume, data_load_path,
                 num_receiver, num_frequency, batchsize, batchsize_v,
                 batchsize_exp, test_only, exp_only, mini_epoch,
                 max_epoch, load_model_path, learning_rate,
                 save_file_path, note]
        if "" in infos:
            QMessageBox.information(self, '提示', '修改的信息不能为空')

        sql = "update operating set " \
              "resume='{1}'," \
              "data_load_path='{2}'," \
              "num_receiver='{3}'," \
              "num_frequency='{4}'," \
              "batchsize='{5}'," \
              "batchsize_v='{6}'," \
              "batchsize_exp='{7}'," \
              "test_only='{8}'," \
              "exp_only='{9}'," \
              "mini_epoch='{10}'," \
              "max_epoch='{11}'," \
              "load_model_path='{12}'," \
              "learning_rate='{13}'," \
              "save_file_path='{14}'," \
              "note='{15}' " \
              "where " \
              "model_name='{0}'".format(*infos)
        self.query.exec(sql)

        # 清空文本框的内容
        self.person_no.setText('')
        self.lineEdit_01_model_name.setText('')
        self.lineEdit_02_resume.setText('')
        self.lineEdit_03_test_only.setText('')
        self.lineEdit_04_exp_only.setText('')
        self.lineEdit_05_num_receiver.setText('')
        self.lineEdit_09_num_frequency.setText('')
        self.lineEdit_13_learning_rate.setText('')
        self.lineEdit_06_load_model_path.setText('')
        self.lineEdit_10_data_load_path.setText('')
        self.lineEdit_14_save_file_path.setText('')
        self.lineEdit_07_batchsize.setText('')
        self.lineEdit_11_batchsize_v.setText('')
        self.lineEdit_15_batchsize_exp.setText('')
        self.lineEdit_08_mini_epoch.setText('')
        self.lineEdit_12_max_epoch.setText('')
        self.lineEdit_16_note.setPlainText('')

        # 重新加载所有信息
        self.reload_all_infos()

        QMessageBox.information(self, '提示', '修改成功')

    def generate_menu(self, pos):
        """右键菜单"""
        menu = QMenu()
        # time.sleep的作用是，按照给定的秒数暂停后再执行程序。
        time.sleep(0.2)
        item1 = menu.addAction("修改")
        item2 = menu.addAction("删除")
        action = menu.exec_(self.tableWidget_01_run_info.mapToGlobal(pos))
        if action == item1:
            print("选择了'修改'操作")

            # 显示“修改信息按钮”
            # self.btn_change.setVisible(True)

            # 从表格中提取需要的数据
            table_selected_index = self.tableWidget_01_run_info.currentIndex().row()
            model = self.tableWidget_01_run_info.model()
            model_name = model.data(model.index(table_selected_index, 0))
            resume = model.data(model.index(table_selected_index, 1))
            data_load_path = model.data(model.index(table_selected_index, 2))
            num_receiver = model.data(model.index(table_selected_index, 3))
            num_frequency = model.data(model.index(table_selected_index, 4))
            batchsize = model.data(model.index(table_selected_index, 5))
            batchsize_v = model.data(model.index(table_selected_index, 6))
            batchsize_exp = model.data(model.index(table_selected_index, 7))
            test_only = model.data(model.index(table_selected_index, 8))
            exp_only = model.data(model.index(table_selected_index, 9))
            mini_epoch = model.data(model.index(table_selected_index, 10))
            max_epoch = model.data(model.index(table_selected_index, 11))
            load_model_path = model.data(model.index(table_selected_index, 12))
            learning_rate = model.data(model.index(table_selected_index, 13))
            save_file_path = model.data(model.index(table_selected_index, 14))
            note = model.data(model.index(table_selected_index, 17))

            # 将这些数据设置到对应的文本框
            if model_name == 'mtl_cnn':
                self.lineEdit_01_model_name.setItemText(0, model_name)
                self.lineEdit_01_model_name.setItemText(1, 'mtl_unet')
                self.lineEdit_01_model_name.setItemText(2, 'mtl_unet_cbam')
                self.lineEdit_01_model_name.setItemText(3, 'xception')
                self.lineEdit_01_model_name.setItemText(4, '~all~')
            elif model_name == 'mtl_unet':
                self.lineEdit_01_model_name.setItemText(0, model_name)
                self.lineEdit_01_model_name.setItemText(1, 'mtl_cnn')
                self.lineEdit_01_model_name.setItemText(2, 'mtl_unet_cbam')
                self.lineEdit_01_model_name.setItemText(3, 'xception')
                self.lineEdit_01_model_name.setItemText(4, '~all~')
            elif model_name == 'mtl_unet_cbam':
                self.lineEdit_01_model_name.setItemText(0, model_name)
                self.lineEdit_01_model_name.setItemText(1, 'mtl_unet')
                self.lineEdit_01_model_name.setItemText(2, 'mtl_cnn')
                self.lineEdit_01_model_name.setItemText(3, 'xception')
                self.lineEdit_01_model_name.setItemText(4, '~all~')
            elif model_name == 'xception':
                self.lineEdit_01_model_name.setItemText(0, model_name)
                self.lineEdit_01_model_name.setItemText(1, 'mtl_unet')
                self.lineEdit_01_model_name.setItemText(2, 'mtl_unet_cbam')
                self.lineEdit_01_model_name.setItemText(3, 'mtl_cnn')
                self.lineEdit_01_model_name.setItemText(4, '~all~')
            if resume == 'True':
                self.lineEdit_02_resume.setItemText(0, resume)
                self.lineEdit_02_resume.setItemText(1, 'False')
            else:
                self.lineEdit_02_resume.setItemText(0, resume)
                self.lineEdit_02_resume.setItemText(1, 'True')
            # 用一个隐藏的文本框记录要修改的人员编号
            # self.person_no.setText(person_no)
            self.lineEdit_10_data_load_path.setText(data_load_path)
            self.lineEdit_05_num_receiver.setText(num_receiver)
            self.lineEdit_09_num_frequency.setText(num_frequency)
            self.lineEdit_07_batchsize.setText(batchsize)
            self.lineEdit_11_batchsize_v.setText(batchsize_v)
            self.lineEdit_15_batchsize_exp.setText(batchsize_exp)
            if test_only == 'True':
                self.lineEdit_03_test_only.setItemText(0, test_only)
                self.lineEdit_03_test_only.setItemText(1, 'False')
            else:
                self.lineEdit_03_test_only.setItemText(0, test_only)
                self.lineEdit_03_test_only.setItemText(1, 'True')
            if exp_only == 'True':
                self.lineEdit_04_exp_only.setItemText(0, exp_only)
                self.lineEdit_04_exp_only.setItemText(1, 'False')
            else:
                self.lineEdit_04_exp_only.setItemText(0, exp_only)
                self.lineEdit_04_exp_only.setItemText(1, 'True')
            self.lineEdit_12_max_epoch.setText(max_epoch)
            self.lineEdit_08_mini_epoch.setText(mini_epoch)
            self.lineEdit_06_load_model_path.setText(load_model_path)
            self.lineEdit_13_learning_rate.setText(learning_rate)
            self.lineEdit_14_save_file_path.setText(save_file_path)
            # self.line_edit_birthday.setDate(QDate.fromString(birthday, "yyyy-MM-dd"))
            self.lineEdit_16_note.setPlainText(note)
        elif action == item2:
            print("选择了'删除'操作")
            reply2 = QMessageBox.warning(self, "警告", "您选择了删除，请谨慎操作，防止记录丢失！", QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)
            if not reply2 == 65536:
                table_selected_index = self.tableWidget_01_run_info.currentIndex().row()
                # 获取表格数据模型对象
                model = self.tableWidget_01_run_info.model()
                # 通过模型对象提取第X行第16列（即运行时间）单元格对象
                table_selected_first_cell = model.index(table_selected_index, 16)
                # 提取编号数据
                run_date_time = model.data(table_selected_first_cell)
                # person_no = table_selected_index
                print("要删除的编号:", run_date_time)

                sql = "delete from person where run_date_time='{}'".format(run_date_time)
                self.query.exec(sql)

                # 重新加载所有信息
                self.reload_all_infos()

                QMessageBox.information(self, '提示', '删除成功')

    def reload_all_infos(self):
        """重新加载所有信息"""
        self.load_all_infos()

    def add_new_run_info(self):
        # oper_no = self.person_no.text()
        # currentText()可以得到当前文本的内容
        model_name = self.lineEdit_01_model_name.currentText()
        resume = self.lineEdit_02_resume.currentText()
        test_only = self.lineEdit_03_test_only.currentText()
        exp_only = self.lineEdit_04_exp_only.currentText()
        data_load_path = self.lineEdit_10_data_load_path.text()
        num_receiver = self.lineEdit_05_num_receiver.text()
        num_frequency = self.lineEdit_09_num_frequency.text()
        batchsize = self.lineEdit_07_batchsize.text()
        batchsize_v = self.lineEdit_11_batchsize_v.text()
        batchsize_exp = self.lineEdit_15_batchsize_exp.text()
        mini_epoch = self.lineEdit_08_mini_epoch.text()
        max_epoch = self.lineEdit_12_max_epoch.text()
        load_model_path = self.lineEdit_06_load_model_path.text()
        learning_rate = self.lineEdit_13_learning_rate.text()
        save_file_path = self.lineEdit_14_save_file_path.text()
        note = self.lineEdit_16_note.toPlainText()

        load_model_path = load_model_path.replace("\\", '/')
        save_file_path = save_file_path.replace("\\", '/')
        data_load_path = data_load_path.replace("\\", '/')

        if model_name == '~all~':
            for model_name in ['mtl_cnn', 'mtl_unet', 'mtl_unet_cbam', 'xception']:
                run_date_time = datetime.datetime.now()
                save_file_name = main(data_loader_path=data_load_path, length_freq=int(num_frequency),
                                      num_of_receiver=int(num_receiver),
                                      receiver_depth=0, cpu=1, rest_time=5, seed=0,
                                      num_of_sources=32, model_name=model_name,
                                      batch_size=int(batchsize), batch_size_v=int(batchsize_v),
                                      batch_size_exp=int(batchsize_exp),
                                      mini_epoch=int(mini_epoch), max_epoch=int(max_epoch), test_only=eval(test_only),
                                      plot_only=False, exp_only=eval(exp_only),
                                      resume=eval(resume),
                                      load_model_path=load_model_path, lr=float(learning_rate),
                                      save_file=save_file_path)
                # save_file_name = model_name + '_2.1datasetTest8_epoch_10.00'
                infos = [model_name, resume, data_load_path,
                         num_receiver, num_frequency, batchsize, batchsize_v,
                         batchsize_exp, test_only, exp_only, mini_epoch,
                         max_epoch, load_model_path, learning_rate,
                         save_file_path, save_file_name, run_date_time, note]
                self.info_wr(infos)
        else:
            # 判断有没有填写的信息，如果有则提示，如果没有则存储到文件
            run_date_time = datetime.datetime.now()
            save_file_name = main(data_loader_path=data_load_path, length_freq=int(num_frequency),
                                  num_of_receiver=int(num_receiver),
                                  receiver_depth=0, cpu=1, rest_time=5, seed=0,
                                  num_of_sources=32, model_name=model_name,
                                  batch_size=int(batchsize), batch_size_v=int(batchsize_v),
                                  batch_size_exp=int(batchsize_exp),
                                  mini_epoch=int(mini_epoch), max_epoch=int(max_epoch), test_only=eval(test_only),
                                  plot_only=False, exp_only=eval(exp_only),
                                  resume=eval(resume),
                                  load_model_path=load_model_path, lr=float(learning_rate), save_file=save_file_path)
            # save_file_name = model_name + '_2.2datasetTest9_epoch_100.00'
            infos = [model_name, resume, data_load_path,
                     num_receiver, num_frequency, batchsize, batchsize_v,
                     batchsize_exp, test_only, exp_only, mini_epoch,
                     max_epoch, load_model_path, learning_rate,
                     save_file_path, save_file_name, run_date_time, note]
            self.info_wr(infos)
        # 重新加载所有信息
        self.reload_all_infos()

        QMessageBox.information(self, '提示', '提交成功')

    def info_wr(self, infos):
        if "" in infos:
            QMessageBox.information(self, '提示', '输入的信息不能为空')
        else:
            sql = "insert into person(" \
                  "model_name," \
                  "resume," \
                  "data_load_path," \
                  "num_receiver," \
                  "num_frequency," \
                  "batchsize," \
                  "batchsize_v," \
                  "batchsize_exp," \
                  "test_only," \
                  "exp_only," \
                  "mini_epoch," \
                  "max_epoch," \
                  "load_model_path," \
                  "learning_rate," \
                  "save_file_path," \
                  "save_file_name," \
                  "run_date_time," \
                  "note) " \
                  "values(" \
                  "'{0}'," \
                  "'{1}'," \
                  "'{2}'," \
                  "'{3}'," \
                  "'{4}'," \
                  "'{5}'," \
                  "'{6}'," \
                  "'{7}'," \
                  "'{8}'," \
                  "'{9}'," \
                  "'{10}'," \
                  "'{11}'," \
                  "'{12}'," \
                  "'{13}'," \
                  "'{14}'," \
                  "'{15}'," \
                  "'{16}'," \
                  "'{17}');".format(*infos)
            self.query.exec(sql)

    def load_all_infos(self):
        """加载所有的人员信息"""
        # 查询信息条数
        sql = "select count(*) from person"
        self.query.exec(sql)
        self.query.next()
        self.tableWidget_01_run_info.setRowCount(self.query.value(0))
        # 查询所有信息（如果数据量过大，切记要limit）
        sql = "select * from person"
        self.query.exec(sql)
        i = 0
        while self.query.next():  # 每次查询一行记录
            for j in range(1, 19):
                new_item = QTableWidgetItem(self.query.value(j))
                new_item.setTextAlignment(Qt.AlignHCenter)
                self.tableWidget_01_run_info.setItem(i, j - 1, new_item)

            i += 1

    def plot_from_files(self):
        self.clear_draw()
        # 创建动画图形并且显示在窗口右侧
        # .rcParams['font.sans-serif'] = 'SimHei'说明是显示中文
        pl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        # .rcParams['axes.unicode_minus'] = False用来显示字符
        pl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        pl.rcParams['font.size'] = '16'  # 设置字体大小
        # 图 1 1
        self.fig_train_range = pl.figure()  # type:figure.Figure
        # pl.subplot()函数用于直接指定划分方式和位置进行绘图。
        self.ax_train_range = pl.subplot()  # type:axes.Axes
        #创造界面组件
        self.canvans_train_range = FigureCanvas(self.fig_train_range)
        #创建NavigationToolbar工具栏时，传递一个FigureCanvas对象作为参数，关联FigureCanvas类对象。
        self.toolbar_train_range = NavigationToolbar(self.canvans_train_range, self)
        self.train_range_layout.addWidget(self.canvans_train_range)
        self.train_range_layout.addWidget(self.toolbar_train_range)
        # self.ax_train_range.plot(xy[0], xy[1])

        # 图 1 2
        self.fig_train_depth = pl.figure()  # type:figure.Figure
        self.ax_train_depth = pl.subplot()  # type:axes.Axes
        self.canvans_train_depth = FigureCanvas(self.fig_train_depth)
        self.toolbar_train_depth = NavigationToolbar(self.canvans_train_depth, self)
        #.addWidget()把括号内的小部件加入
        self.train_depth_layout.addWidget(self.canvans_train_depth)
        self.train_depth_layout.addWidget(self.toolbar_train_depth)
        # self.ax_train_depth.plot(xy[0], xy[1])

        # 图 2 1
        self.fig_test_range = pl.figure()  # type:figure.Figure
        self.ax_test_range = pl.subplot()  # type:axes.Axes
        self.canvans_test_range = FigureCanvas(self.fig_test_range)
        self.toolbar_test_range = NavigationToolbar(self.canvans_test_range, self)
        self.test_range_layout.addWidget(self.canvans_test_range)
        self.test_range_layout.addWidget(self.toolbar_test_range)
        # self.ax_test_range.plot(xy[0], xy[1])

        # 图 2 2
        self.fig_test_depth = pl.figure()  # type:figure.Figure
        self.ax_test_depth = pl.subplot()  # type:axes.Axes
        self.canvans_test_depth = FigureCanvas(self.fig_test_depth)
        self.toolbar_test_depth = NavigationToolbar(self.canvans_test_depth, self)
        self.test_depth_layout.addWidget(self.canvans_test_depth)
        self.test_depth_layout.addWidget(self.toolbar_test_depth)
        # self.ax_test_depth.plot(xy[0], xy[1])

        # 图 3 1
        self.fig_exp_range = pl.figure()  # type:figure.Figure
        self.ax_exp_range = pl.subplot()  # type:axes.Axes
        self.canvans_exp_range = FigureCanvas(self.fig_exp_range)
        self.toolbar_exp_range = NavigationToolbar(self.canvans_exp_range, self)
        self.exp_range_layout.addWidget(self.canvans_exp_range)
        self.exp_range_layout.addWidget(self.toolbar_exp_range)
        # self.ax_exp_range.plot(xy[0], xy[1])

        # 图 3 2
        self.fig_exp_depth = pl.figure()  # type:figure.Figure
        self.ax_exp_depth = pl.subplot()  # type:axes.Axes
        self.canvans_exp_depth = FigureCanvas(self.fig_exp_depth)
        self.toolbar_exp_depth = NavigationToolbar(self.canvans_exp_depth, self)
        self.exp_depth_layout.addWidget(self.canvans_exp_depth)
        self.exp_depth_layout.addWidget(self.toolbar_exp_depth)
        # self.ax_exp_depth.plot(xy[0], xy[1])

        #isChecked()是用来复选框的。即CheckBox对象。区分CheckBox是否被选中，选中则返回1，反之返回0.
        mtl_cnn_plt = self.checkBox_01_mtl_cnn.isChecked()
        mtl_unet_plt = self.checkBox_02_mtl_unet.isChecked()
        mtl_unet_cbam_plt = self.checkBox_03_mtl_unet_cbam.isChecked()
        xception_plt = self.checkBox_04_xception.isChecked()
        checkbox_plt_list = [mtl_cnn_plt, mtl_unet_plt, mtl_unet_cbam_plt, xception_plt]
        # model_name_list = ['mtl_cnn', 'mtl_unet', 'mtl_unet_cbam', 'xception', ]
        model = self.tableWidget_02_plot_info.model()
        save_file = model.data(model.index(0, 2))

        # training loss
        dpi = 200
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        line_style = ['-', '--', '-.', ':']
        self.tableWidget_03_results.setRowCount(len(self.model_name_list))
        model_name = []
        j = 0

        for i in range(0, len(checkbox_plt_list)):
            if checkbox_plt_list[i]:

                save_file_name = model.data(model.index(i, 3))
                mat_path = os.path.abspath(save_file + '/b.training_results/data/' + save_file_name +
                                           '_training_process' + '.mat')
                plot_data = loadmat(mat_path)
                # plt.plot(x.loc[i], y.loc[j], color[i * 4 + j] + line_style[j])
                x = np.squeeze(plot_data['epoch'])
                y = np.squeeze(plot_data['MAE_r'])
                model_name.append((str(plot_data['model_name']))[2:-2])
                #ax_train_range可以认为是提供作图的空间。python中.plot是为了可视化作图
                self.ax_train_range.plot(x, y, color[j] + line_style[j],
                                         label=model_name[j])
                #
                new_item = QTableWidgetItem(model_name[j])
                new_item.setTextAlignment(Qt.AlignHCenter)
                self.tableWidget_03_results.setItem(j, 0, new_item)
                train_loss_r_str = '%.3f km' % y[-1]
                new_item = QTableWidgetItem(train_loss_r_str)
                #Qt.AlignHCenter让文字水平居中
                new_item.setTextAlignment(Qt.AlignHCenter)
                self.tableWidget_03_results.setItem(j, 1, new_item)
                j = j + 1
        self.tableWidget_03_results.update()

        self.ax_train_range.legend()
        self.ax_train_range.set_xlabel("epoch")
        self.ax_train_range.set_ylabel("MAE_r (km)")
        j = 0
        for i in range(0, len(checkbox_plt_list)):
            if checkbox_plt_list[i]:
                save_file_name = model.data(model.index(i, 3))
                mat_path = os.path.abspath(save_file + '/b.training_results/data/' + save_file_name +
                                           '_training_process' + '.mat')
                plot_data = loadmat(mat_path)
                # plt.plot(x.loc[i], y.loc[j], color[i * 4 + j] + line_style[j])
                x = np.squeeze(plot_data['epoch'])
                y = np.squeeze(plot_data['MAE_d'])
                self.ax_train_depth.plot(x, y, color[j] + line_style[j],
                                         label=model_name[j])
                train_loss_d_str = '%.3f m' % y[-1]
                new_item = QTableWidgetItem(train_loss_d_str)
                new_item.setTextAlignment(Qt.AlignHCenter)
                self.tableWidget_03_results.setItem(j, 2, new_item)
                j = j + 1
        self.ax_train_depth.legend()
        self.ax_train_depth.set_xlabel("epoch")
        self.ax_train_depth.set_ylabel("MAE_d (m)")

        # test accuracy
        j = 0
        for i in range(0, len(checkbox_plt_list)):
            if checkbox_plt_list[i]:
                save_file_name = model.data(model.index(i, 3))
                mat_path = os.path.abspath(save_file + '/c.test_results/data/' + save_file_name +
                                           '_test_accuracy' + '.mat')
                plot_data = loadmat(mat_path)
                # plt.plot(x.loc[i], y.loc[j], color[i * 4 + j] + line_style[j])
                x = np.squeeze(plot_data['para'])
                y = np.squeeze(plot_data['PRPA']) * 100
                self.ax_test_range.plot(x, y, color[j] + line_style[j],
                                        label=model_name[j])
                test_m_acc_r_str = '%.1f %%' % np.mean(y)
                new_item = QTableWidgetItem(test_m_acc_r_str)
                new_item.setTextAlignment(Qt.AlignHCenter)
                self.tableWidget_03_results.setItem(j, 3, new_item)
                j = j + 1
        self.ax_test_range.legend()
        self.ax_test_range.set_xlabel("Parameter")
        self.ax_test_range.set_ylabel("Proportion (%)")

        j = 0
        for i in range(0, len(checkbox_plt_list)):
            if checkbox_plt_list[i]:
                save_file_name = model.data(model.index(i, 3))
                mat_path = os.path.abspath(save_file + '/c.test_results/data/' + save_file_name +
                                           '_test_accuracy' + '.mat')
                plot_data = loadmat(mat_path)
                # plt.plot(x.loc[i], y.loc[j], color[i * 4 + j] + line_style[j])
                x = np.squeeze(plot_data['para'])
                y = np.squeeze(plot_data['PDPA']) * 100
                self.ax_test_depth.plot(x, y, color[j] + line_style[j],
                                        label=model_name[j])
                test_m_acc_d_str = '%.1f %%' % np.mean(y)
                new_item = QTableWidgetItem(test_m_acc_d_str)
                new_item.setTextAlignment(Qt.AlignHCenter)
                self.tableWidget_03_results.setItem(j, 4, new_item)
                j = j + 1
        self.ax_test_depth.legend()
        self.ax_test_depth.set_xlabel("Parameter")
        self.ax_test_depth.set_ylabel("Proportion (%)")

        # exp estimation
        j = 0
        for i in range(0, len(checkbox_plt_list)):
            if checkbox_plt_list[i]:
                save_file_name = model.data(model.index(i, 3))
                mat_path = os.path.abspath(save_file + '/d.exp_results/data/' + save_file_name +
                                           '_exp_estimation' + '.mat')
                plot_data = loadmat(mat_path)
                # plt.plot(x.loc[i], y.loc[j], color[i * 4 + j] + line_style[j])
                # x = np.squeeze(plot_data['epoch'])
                y = np.squeeze(plot_data['Err_r'])
                x = np.arange(0, np.size(y))
                self.ax_exp_range.plot(x, y, color[j] + line_style[j],
                                       label=model_name[j])
                exp_mae_r_str = '%.3f km' % np.mean(y)
                new_item = QTableWidgetItem(exp_mae_r_str)
                new_item.setTextAlignment(Qt.AlignHCenter)
                self.tableWidget_03_results.setItem(j, 5, new_item)
                j = j + 1
        self.ax_exp_range.legend()
        self.ax_exp_range.set_xlabel("Sample index")
        self.ax_exp_range.set_ylabel("AE_r (km)")

        j = 0
        for i in range(0, len(checkbox_plt_list)):
            if checkbox_plt_list[i]:
                save_file_name = model.data(model.index(i, 3))
                mat_path = os.path.abspath(save_file + '/d.exp_results/data/' + save_file_name +
                                           '_exp_estimation' + '.mat')
                plot_data = loadmat(mat_path)
                # plt.plot(x.loc[i], y.loc[j], color[i * 4 + j] + line_style[j])
                y = np.squeeze(plot_data['Err_d'])
                x = np.arange(0, np.size(y))
                self.ax_exp_depth.plot(x, y, color[j] + line_style[j],
                                       label=model_name[j])
                exp_mae_d_str = '%.3f m' % np.mean(y)
                new_item = QTableWidgetItem(exp_mae_d_str)
                new_item.setTextAlignment(Qt.AlignHCenter)
                self.tableWidget_03_results.setItem(j, 6, new_item)
                j = j + 1
        self.ax_exp_depth.legend()
        self.ax_exp_depth.set_xlabel("Sample index")
        self.ax_exp_depth.set_ylabel("AE_d (m)")
        # for i in range(0, len(checkbox_plt_list)):
        #     for j in range(1, 4):
        #         new_item = QTableWidgetItem(self.query.value(j))
        #         new_item.setTextAlignment(Qt.AlignHCenter)
        #         self.tableWidget_01_run_info.setItem(i, j - 1, new_item)

        # pl.figure(dpi=dpi, figsize=(10, 3))
        # pl.subplot(131)
        # bp(x, MAE_of_range, x_tick_off=False, legend_label='Train', color='b')
        # bp(x, MAE_of_range_v, title='', x_label='epoch',
        #    y_label=r'$MAE_r$ (km)', x_tick_off=False, legend_label='Validation', color='r',
        #    font_size=20, loc='upper right')
        # pl.subplot(132)
        # bp(x, MAE_of_depth, x_tick_off=False, legend_label='Train', color='b')
        # bp(x, MAE_of_depth_v, title='', x_label='epoch',
        #    y_label=r'$MAE_d$ (m)', x_tick_off=False, legend_label='Validation', color='r'
        #    , font_size=20, loc='upper right')

    def clear_draw(self):
        try:
            # 清除右侧画布内容
            for i in range(self.train_range_layout.count()):
                self.train_range_layout.itemAt(i).widget().deleteLater()
            for i in range(self.train_depth_layout.count()):
                self.train_depth_layout.itemAt(i).widget().deleteLater()
            for i in range(self.test_range_layout.count()):
                self.test_range_layout.itemAt(i).widget().deleteLater()
            for i in range(self.test_depth_layout.count()):
                self.test_depth_layout.itemAt(i).widget().deleteLater()
            for i in range(self.exp_range_layout.count()):
                self.exp_range_layout.itemAt(i).widget().deleteLater()
            for i in range(self.exp_depth_layout.count()):
                self.exp_depth_layout.itemAt(i).widget().deleteLater()
        except:
            pass


if __name__ == '__main__':
    # 创建应用程序对象
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    # 设置窗口窗口风格
    QApplication.setStyle(QStyleFactory.create("Fusion"))
    print(QStyleFactory.keys())

    # 创建一个要显示的窗口对象
    app_widget = AppWidget()
    app_widget.show()

    # 让应用程序一直运行，在这期间会显示UI界面、检测事件等操作
    sys.exit(app.exec())
