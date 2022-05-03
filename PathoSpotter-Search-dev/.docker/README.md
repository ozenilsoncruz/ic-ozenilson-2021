# Ambiente com docker

## Pré-requisitos
* Instalar o `Docker`.

[Instalação oficial](https://docs.docker.com/get-docker/)
[Instalação no Fedora](https://computingforgeeks.com/how-to-install-docker-on-fedora/)

* Instalar o `Docker Compose`.

[Instalação oficial](https://docs.docker.com/compose/install/)
[Instalação no Fedora](https://computingforgeeks.com/install-and-use-docker-compose-on-fedora/)

* Baixar o `Kimia-Path-960`
O dataset escolhido e organizado se encontra no driver, na pasta PathoSpotter Search. Caso nao tenha acesso, solicite e baixe o arquivo `KIMIA_Path_960.tar.gz` e extraia o arquivo para a o diretorio do projeto.

## Construindo a imagem
A imagen para treinamento e fine tuning e o arquivo `.docker/train.Dockerfile`. Caso necessite de outras bibliotecas python que nao estao instaladas, adicione ao arquivo `.docker/train-requirements.txt`.

Execute o comando abaixo para gerar a imagem a ser utilizada pelo container.

``` bash
docker build -f .docker/train.Dockerfile -t ps-search .
```

## Levantando o ambiente
* Criar o container a partir da imagem `ps-search` gerada anteriormente executando o comando abaixo.

``` bash
docker run --gpus all -it -v {{base-path}}/pathospotter-search/PathoSpotter-Search-dev:/usr/src/app -w /usr/src/app ps-search bash
```

Dessa forma o terminal do container estara integrado com o terminal do host permitindo a execucao de comandos.

## Scripts ajustados para rodar no container
Atualmente apenas o arquivo `vgg16_finetuning.py` esta atualizado para ser executado na nova versao do tensorflow em que a imagem docker esta baseada.

Estando no terminal integrado do container com o host execute o comando
``` bash
python vgg16_finetuning.py
```