# -*- coding: utf-8 -*-

import json
import os
import pathlib
from os import sep

from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QWidget, QLabel, QInputDialog, QProgressBar

from DataBase import DataBase
from resultList import ResultList


class MyWindow(QMainWindow):
    name_representations = ['vgg16', 'inception', 'pspotter']  # Define a configuração esperada de arquivo.
    metrics = ["euclidian", "cosine"]  # Define as métricas disponíveis.
    directory = pathlib.Path(__file__).parent.absolute()  # Pega o diretório da pasta atual
    directory = str(directory)  # Transforma o diretório em string
    file_config = directory + sep + 'config_pssearch.json'

    def __init__(self):  # Inicializando a classe.
        super(MyWindow, self).__init__()  # Inicializa a clase pai
        self.db = DataBase()  # Cria para si uma instância da classe Database
        self.result = ResultList()  # Cria para si uma instância da classe resultList
        self.setGeometry(200, 200, 320, 350)  # Tamanho da tela.
        self.setFixedSize(320, 350)
        self.setWindowTitle("PathoSpotter")  # Nome da janela.
        self.setWindowIcon(QIcon('icon.png'))  # Ícone da janela.
        self._main = QWidget()  # Definindo a classe principal da interface
        self.setCentralWidget(self._main)  # Colocando a classe no centro da interface
        self.k = 10  # Definindo o número padrão de resultados a serem exibidos.
        '''
        self.configMenu = QMenu("&Configurações", self)  # Menu de Configurações
        self.setKAct = QAction("Quantidade de resultados", self, shortcut="Ctrl+K", triggered=self.change_k)
        self.setDBAct = QAction("Definir dataset", self, shortcut="Ctrl+D", triggered=self.open_database)
        self.configMenu.addAction(self.setKAct)  # Adiciona a opção de "Quantidade de resultados" ao menu
        self.configMenu.addAction(self.setDBAct)  # Adiciona a opção de "Definir dataset" ao menu
        self.menuBar().addMenu(self.configMenu)  # Adiciona o menu à tela
        '''
        self.icon = QLabel(self)  # Logo do LACAD
        self.pixmap = QPixmap('logo.png')  # Este processo é padrão do PyQt5
        self.icon.setPixmap(self.pixmap)  # Para adicionar uma imagem na tela, é preciso uma label
        self.icon.resize(self.pixmap.width(), self.pixmap.height())  # Colocamos a imagem dentro da label
        self.icon.move(10, 20)  # Posicionando elemento na interface "pra ficar bonito"

        self.welcomeLabel = QtWidgets.QLabel(self)  # Label de Bem-vindo!
        self.welcomeLabel.setText("Bem-vindo ao PathoSpotter!")  # Essa label é amigável
        self.welcomeLabel.resize(150, 30)  # Mudando a label de tamanho
        self.welcomeLabel.move(75, 135)  # Posicionando elemento na interface "pra ficar bonito"

        self.userLabel = QtWidgets.QLabel(self)  # Label de instruções ao usuário.
        self.userLabel.setText("Por favor, diga-nos o diretório de suas imagens.")  # Essa label muda!
        self.userLabel.resize(250, 30)  # Mudando a label de tamanho
        self.userLabel.move(50, 175)  # Posicionando elemento na interface "pra ficar bonito"

        self.pbar = QProgressBar(self)  # Barra de Progresso
        self.pbar.move(10, 215)  # Posicionando elemento na interface "pra ficar bonito"
        self.pbar.setGeometry(20, 215, 260, 20)  # Mudando o tamanho da barra de progresso
        self.pbar.setMinimum(0)  # Coloca o número mínimo da barra como 0
        self.pbar.setFormat(" Imagem %v de %m.")  # Coloca o texto da barra como imagem (atual) de (total)
        self.pbar.setVisible(False)  # Esconde a barra até ela ser necessária

        self.selectDatabase_button = QtWidgets.QPushButton(self)  # Botão de processar diretório
        self.selectDatabase_button.setText("Definir diretório.")  # Coloca um texto no botão
        self.selectDatabase_button.resize(150, 30)  # Mudando o tamanho do botão
        self.selectDatabase_button.move(70, 245)  # Posicionando elemento na interface "pra ficar bonito"
        self.selectDatabase_button.clicked.connect(self.generate_db)

        self.search_button = QtWidgets.QPushButton(self)  # Botão da busca por imagem
        self.search_button.setText("Buscar por imagem.")  # Coloca um texto no botão
        self.search_button.resize(150, 30)  # Mudando o tamanho do botão
        self.search_button.move(70, 285)  # Posicionando elemento na interface "pra ficar bonito"
        self.search_button.clicked.connect(self.search_images)  # Conecta o botão à ação que ele realiza
        self.search_button.setDisabled(True)  # Desativa o botão até o usuário possuir um dataset processado

        self.heap = []  # Heap de resultados que serão exibidos na tela filha
        self.window = QtWidgets.QMainWindow()  # Faz uma janela pra colocar todas essas coisas.
        self.load_configurations()  # Chama o método que lê as configurações iniciais

    def generate_db(self):
        self.pbar.setVisible(True)
        self.userLabel.setText("Selecione um diretório na caixa de diálogo.")
        path_in = str(QFileDialog.getExistingDirectory(self, "Selecione o Diretório"))
        #path_out = QFileDialog.getSaveFileName(self, 'Salve o Dataset', 'pssearch_dataset', 'JSON (*.json)')
        if path_in:
            self.userLabel.setText("Estamos processando seus dados.")
            directory = pathlib.Path(__file__).parent.absolute()  # Pega o diretório da pasta atual
            directory = str(directory)  # Transforma o diretório em string
            path_out = directory + sep + 'pssearch_dataset.json'  # Adiciona o nome do arquivo ao diretório
            self.db.generate_representations(bar=self.pbar, path_in=path_in, path_out=path_out)

            self.path_db = path_out
            print(path_out)
            with open(self.file_config, 'w') as f:
                f.write(self.__generate_json_configuration__())
                f.close()
            self.init_db(path_out)
            self.pbar.setVisible(False)
        else:
            self.userLabel.setText("Por favor, selecione um diretório válido.")
            self.pbar.setVisible(False)

    def open_database(self):
        '''
        Metodo usado para setar banco de dados que será utilizado.
        '''
        options = QFileDialog.Options()  # Supostamente, aqui se abre o arquivo.
        fileName, _ = QFileDialog.getOpenFileName(self, 'Selecione um dataset', '',
                                                  'Databases (*.json)', options=options)
        if fileName:  # Zero ideias.
            with open(self.file_config, 'w') as f:
                self.path_db = fileName
                f.write(self.__generate_json_configuration__())
                f.close()
                self.init_db(fileName)

    def __generate_json_configuration__(self):  # Gera o JSON das configurações do banco na máquina do usuário.
        conf = {}
        conf["path_db"] = self.path_db
        conf["name_representations"] = self.name_representations
        conf["metrics"] = self.metrics

        return json.dumps(conf)

    def init_db(self, path_db):  # Inicializa o banco de dados.
        print('initializing database, loading file:%s' % path_db)
        retorno = self.db.load_representations(path_db, self.name_representations)
        if retorno == 0:
            self.userLabel.setText("Sinto muito, não conseguimos ler seu dataset.")
        else:
            self.search_button.setDisabled(False)
            self.userLabel.setText("Por favor, realize sua busca.")

    def change_k(self):
        i, okPressed = QInputDialog.getInt(self, "Defina a quantidade de resultados", "K:", 10, 0, 100, 1)
        if okPressed:
            self.k = i
            print(self.k)

    def search_images(self):
        tempk, ok = QInputDialog.getInt(self, 'Definir Resultados', 'Quantos resultados deseja obter no máximo?', 10)
        if ok:
            self.userLabel.setText("Por favor, selecione a imagem.")
            self.k = tempk
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getOpenFileName(self, 'Selecione uma imagem', '',
                                                      'Images (*.png *.jpeg *.jpg *.bmp *.gif *.tif)', options=options)
            if fileName:
                image = QImage(fileName)
                if image.isNull():
                    QMessageBox.information(self, "PathoSpotter", "Cannot load %s." % fileName)
                    return
                self.imageFileName = fileName
                self.userLabel.setText("Estamos buscando a imagem.")
                print('starting search for %s ..' % self.imageFileName)
                self.heap = self.db.search(self.imageFileName, model=int("1"), k=self.k, distance=self.metrics[int("1")])
                self.userLabel.setText("Sua busca foi concluída.")
                self.result.show()
                self.result.show_pictures(self.heap)
            else:
                self.userLabel.setText("Por favor, busque uma imagem válida.")
        else:
            self.userLabel.setText("Número informado inválido. Tente novamente.")

    def load_configurations(self):
        directory = pathlib.Path(__file__).parent.absolute()  # Pega o diretório da pasta atual
        directory = str(directory)  # Transforma o diretório em string
        file = directory + sep + 'config_pssearch.json'
        print("loading config file...")
        if os.path.isfile(file):
            with open(self.file_config, 'r') as f:
                config_dict = json.load(f)
                f.close()
        else:
            with open(self.file_config, 'w') as f:
                aux = {}
                aux["path_db"] = directory + sep + 'kpath_bd.json'
                aux["name_representations"] = self.name_representations
                aux["metrics"] = self.metrics
                f.write(json.dumps(aux))
                f.close()

                config_dict = aux

        self.path_db = config_dict["path_db"]
        if os.path.isfile(self.path_db):
            print('Já havia um arquivo: %s' % self.path_db)
            self.init_db(self.path_db)
        else:
            self.userLabel.setText("Por favor, diga-nos o diretório de suas imagens.")

        self.name_representations = config_dict["name_representations"]
        self.metrics = config_dict["metrics"]

#Método que inicia a interface.
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()