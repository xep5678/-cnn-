from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from test import Ui_MainWindow


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.pushButton_menu1.clicked.connect(self.menu1_method)
        # self.pushButton_menu2.clicked.connect(self.menu2_method)
        # self.pushButton_menu3.clicked.connect(self.menu3_method)

    def menu1_method(self):
        self.lineEdit.setText('Hello World! Message from menu1.')

    def menu2_method(self):
        self.lineEdit.setText('Hello World! Message from menu2.')

    def menu3_method(self):
        self.lineEdit.setText('Hello World! Message from menu3.')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    sys.exit(app.exec_())
