from os import sep
from os.path import expanduser

from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QWidget, QGridLayout, QScrollArea

from ImageLabel import ImageLabel


class ResultList(QMainWindow):
    padding = "QListWidget {padding: 10px;} QListWidget::item { margin: 10px; } "
    file_config = expanduser("~") + sep + '.pssearch' + sep + '.config_pssearch.json'

    def __init__(self):  # Inicializando a classe.
        super(ResultList, self).__init__()  # Inicializando a classe pai
        self.setWindowIcon(QIcon('icon.png'))  # Ícone da janela.
        layout = QGridLayout()  # Layout de Grade
        layout.setSpacing(10)
        layout.setHorizontalSpacing(10)
        self.setGeometry(300, 100, 1040, 520)  # Tamanho da tela.
        self.setWindowTitle("Resultados da Pesquisa")  # Nome da janela.
        self._main = QWidget(self)  # Cria um Widget que receberá o Layout
        self._main.setLayout(layout)  # Coloca o Layout no Widget
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self._main)
        self.scrollArea.setLayout(layout)
        self.setCentralWidget(self.scrollArea)  # Coloca o Widget no meio da tela

    def show_pictures(self, heap):
        row = col = 0  # Inicia a tela sem linhas e colunas
        quant = len(heap)
        count = 1
        if quant < 9:
            colQ = 4
        elif quant < 13:
            colQ = 5
        else:
            colQ = 6
        for image_representation in heap:  # Para cada image nos resultados:
            label = ImageLabel(image_representation.path, count)  # Cria uma instância da classe ImageLabel
            image = QImage(image_representation.path)  # Pega a imagem do caminho
            pixmap = QPixmap(image)  # Coloca a imagem num pixmap
            pixmap = pixmap.scaled(250, 250)  # Coloca uma nova escala na imagem
            label.setPixmap(pixmap)  # Coloca o pixmap na imagem
            self.scrollArea.layout().addWidget(label, row, col)  # Coloca a imagem na tela
            col += 1  # Anda uma coluna
            count += 1  # Anda o contador
            if col % colQ == 0:  # Se passar de 4 imagens por coluna
                row += 1  # Passa para a próxima linha
                col = 0  # Volta para a primeira coluna


# Método que inicia a interface. Padrão do PyQt.
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = ResultList()
    window.show()
    app.exec_()
