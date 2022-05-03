import subprocess

from PyQt5.QtWidgets import QLabel


class ImageLabel(QLabel):

    def __init__(self, path, count):  # Inicializando a classe.
        super(ImageLabel, self).__init__()  # Inicializando a clase pai.
        self.setMouseTracking(True)  # Começa a acompanhar o mouse do usuário.
        self.path = path  # Guarda o caminho da imagem.
        self.tooltipText = "Número " + str(count) + " em similaridade."
        self.setToolTip(self.tooltipText)

    def mousePressEvent(self, event):  # Quando o mouse for pressionado
        subprocess.run(r'"{}"'.format(self.path), shell=True)  # Chama o SO para abrí-lo no programa correspondente
