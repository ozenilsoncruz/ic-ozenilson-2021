#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:53:16 2019

@author: gabriel
"""

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter, QIcon
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, qApp, QFileDialog, \
QWidget
import PyQt5.QtWidgets as QtWidgets
from DataBase import DataBase
import json
from os.path import expanduser
from os import sep
import os

win_title = "PathoSpotter-K Search"
class Application(QMainWindow):
    
#    path_db = '/home/gabriel/bd_.json'
    name_representations = ['vgg16', 'inception', 'pspotter', 'vgg16ft']
    metrics = ["euclidian", "cosine"]
    quantities = ['10', '20', '25', '30', '35']
    padding = "QListWidget {padding: 10px;} QListWidget::item { margin: 10px; }"
    file_config = expanduser("~")+sep+'.pssearch'+sep+'.config_pssearch.json'
    def __init__(self):
        self.load_configurations()
        self.init_db()
        super().__init__()
        #seta icone
        self.setWindowIcon(QIcon('lente.jpeg'))
        
        self.createActions()
        self.createMenus()
        
        self.setWindowTitle(win_title)
        self.resize(1024, 600)
        self._main = QWidget()
        self.setCentralWidget(self._main)
        
        #botoes
        self.search_button = QtWidgets.QPushButton("search")
        self.search_button.clicked.connect(self.search_images)
        self.search_button.setDisabled(True)
        self.selectImage_button = QtWidgets.QPushButton("select image")
        self.selectImage_button.clicked.connect(self.open_image)
        
        #labels
        self.imagePath_label = QLabel()
        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.quantity_label = QLabel("Quantity: ")
        self.engine_label = QLabel("Engine: ")
        self.distance_label = QLabel("Metric: ")
        self.results_label = QLabel("Results:")
        self.results_label.setVisible(False)
        
        #comboboxes
        self.engine_cbox = QtWidgets.QComboBox()
        self.engine_cbox.addItems(self.name_representations)
        self.distance_cbox = QtWidgets.QComboBox()
        self.distance_cbox.addItems(self.metrics)
        self.quantity_cbox = QtWidgets.QComboBox()
        self.quantity_cbox.addItems(self.quantities)
        
        #listViews
        self.listView = QtWidgets.QListWidget()
        self.listView.setVisible(False)
        self.listView.setStyleSheet(self.padding)
        
        #layouts
        self.layout = QtWidgets.QHBoxLayout(self._main)
        options_layout = QtWidgets.QHBoxLayout()
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.selectImage_button)
#        left_layout.addWidget(QtWidgets.QSplitter())
        left_layout.addWidget(self.image_label, 1)
        
        options_layout.addWidget(self.quantity_label)
        options_layout.addWidget(self.quantity_cbox)
        options_layout.addWidget(self.engine_label)
        options_layout.addWidget(self.engine_cbox)
        options_layout.addWidget(self.distance_label)
        options_layout.addWidget(self.distance_cbox)
        
        left_layout.addLayout(options_layout)
        left_layout.addWidget(self.search_button)
        
        
        right_layout = QtWidgets.QVBoxLayout()
#        right_layout.addWidget(self.image_label, 1)
        right_layout.addWidget(self.results_label)
        right_layout.addWidget(self.listView, 1)
        #right_layout.addWidget(self.imagePath_label, 0)
        
        
        
        
        self.layout.addLayout(left_layout, 1);
        self.layout.addLayout(right_layout, 1);
        
        #seta status bar para mostrar qual banco de dados está sendo utilizando
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.update_statusBar()
    
    def init_db(self):
        print('initializing database, loading file:%s'%(self.path_db))
        self.db = DataBase()
        self.db.load_representations(self.path_db, self.name_representations)
        
    def load_configurations(self):
        home = expanduser("~")
        file = home+sep+'.pssearch'+sep+'.config_pssearch.json'
        
        print("loading config file...")
        if not os.path.exists(home+sep+'.pssearch'+sep):
            os.mkdir(home+sep+'.pssearch'+sep)
            
        if os.path.isfile(file):
            with open(self.file_config, 'r') as f:
                config_dict = json.load(f)
                f.close()
        else:
            with open(self.file_config, 'w') as f:
                aux = {}
                aux["path_db"] = home+sep+'.pssearch'+sep+'kpath_bd.json'
                aux["name_representations"] = self.name_representations
                aux["metrics"] = self.metrics
                f.write(json.dumps(aux))
                f.close()
                
                config_dict = aux
        
        self.path_db = config_dict["path_db"]
        self.name_representations = config_dict["name_representations"]
        self.metrics = config_dict["metrics"]
        
    
    def update_statusBar(self):
        self.statusBar.showMessage("utilizing database: "+self.path_db+ ", number of images: "+str(len(self.db)))
    
    
    def search_images(self):
        self.results_label.setVisible(False)
        self.listView.setVisible(False)
        print('starting search for %s ..' % (self.imageFileName))
        heap = self.db.search(self.imageFileName, model=self.engine_cbox.currentIndex(), k=int(self.quantities[self.quantity_cbox.currentIndex()]), distance=self.metrics[self.distance_cbox.currentIndex()])
        print('Done. Showing results..')
        
        self.listView.clear()
        for image_representation in heap:
            item_list = QtWidgets.QListWidgetItem()
            image_label = QLabel()
            image = QImage(image_representation.path)
            image_label.setPixmap(QPixmap.fromImage(image))
            image_label.setScaledContents(True)
            (width, height) = self.get_new_dimensions(image.width(),image.height())
#            image_label.setFixedSize(width, height)
            item_list.setSizeHint(QSize(400, 400))
            self.listView.addItem(item_list)
            self.listView.setItemWidget(item_list, image_label)
        
        self.results_label.setVisible(True)
        self.listView.setVisible(True)
        
        
    def createActions(self):
        self.generateFeaturesAct = QAction('&Generate Features...', self, shortcut="Ctrl+F", triggered=self.generate_db)
        self.aboutAct = QAction('&About', self, triggered=self.about)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.setDBAct = QAction("Set DB", self, shortcut="Ctrl+B", triggered=self.set_db)
    
    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.configMenu = QMenu("&Config", self)
        self.fileMenu.addAction(self.generateFeaturesAct)
        self.fileMenu.addAction(self.aboutAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)
        self.configMenu.addAction(self.setDBAct)
        
        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.configMenu)
        
    def about(self):
        QMessageBox.about(self, "About PathoSpotter-K Search",
                          "The PathoSpotter project emerged through the desire and enthusiasm of the pathologist Dr. Washington Luís, from the Oswald Cruz Foundation in Brazil, for improving the clinical-pathological diagnoses for renal diseases. In 2014, Dr. Washington met Dr. Angelo Duarte from State University of Feira de Santana (Brazil) and both start the building of the project PathoSpotter. In the beginning, Pathospotter aimed to assist pathologists in the identification of glomerular lesions from kidney biopsies. As the work evolved, the aim shifted to make large-scale correlations of lesions with clinical and historical data of patients, to enlarge the knowledge in kidney pathology. Currently, the project Pathospotter intends to offer computational tools to: - Facilitate the consensus of clinicians and pathologists, helping to achieve more accurate and faster diagnoses; - Facilitate large-scale clinical-pathological correlations studies; - Help pathologists and students in classifying lesions in kidney biopsies. Developed by Gabriel Antonio Carneiro (gabri14el@gmail.com)")
    def open_image(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(self, 'Select an image to open', '',
                                                  'Images (*.png *.jpeg *.jpg *.bmp *.gif *.tif)', options=options)
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(self, win_title, "Cannot load %s." % fileName)
                return
            
            self.imagePath_label.setText(fileName)
            self.imageFileName = fileName
            
            self.image_label.setPixmap(QPixmap.fromImage(image))
            (width, height) = self.get_new_dimensions(image.width(),image.height())
            self.image_label.setFixedSize(width, height)
            self.search_button.setDisabled(False)
            
    
    def generate_db(self):
        path_in = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        path_out = QFileDialog.getSaveFileName(self, 'Save File', 'pssearch_database', 'JSON (*.json)')
        
        self.db.generate_representations(path_in=path_in, path_out=path_out[0])
        
    def get_new_dimensions(self, width, height):
        big_side_size = None
        big_side_img = 400
        if width > height:
            big_side_size = width
        else:
            big_side_size = height
        
        scale_percent = big_side_img/big_side_size
        
        width = int(width*scale_percent)
        height = int(height*scale_percent)
        return (width, height)
    
    
    def __generate_json_configuration__(self):
        
        conf = {}
        conf["path_db"] = self.path_db
        conf["name_representations"] = self.name_representations
        conf["metrics"] = self.metrics
        
        return json.dumps(conf)
    
    def set_db(self):
        '''
        Metodo usado para setar banco de dados que será utilizado.
        '''
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, 'Select an image to open', '',
                                                  'Databases (*.json)', options=options)
        
        if fileName:
            with open(self.file_config, 'w') as f:
                self.path_db = fileName
                f.write(self.__generate_json_configuration__())
                f.close()
                self.init_db()
                print('db modified to %s' %(fileName))
                self.update_statusBar()

        
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    imageViewer = Application()
    imageViewer.show()
    sys.exit(app.exec_())