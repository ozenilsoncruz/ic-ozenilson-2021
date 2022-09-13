<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/LACAD/pathospottersearch-engine/">
    <img src="/PathoSpotter-Search-app/logo.png" alt="Logo" height="80">
  </a>

  <h3 align="center">PathoSpotter</h3>

  <p align="center">
    PathoSpotter is a computational tool built to help pathologists. A diagnosis based only in the confidence of the pathologist in his or hers experience and vision is, in most times, enough. However, to some very difficult cases, it is nice to have another specialist glance. This is why we train neural networks to serve as second opinion to pathologists diagnosis.<br>
Nevertheless, it is still difficult to make a computer understand what the human eye sees. For that reason, we are constantly making new experiments to improve the confidence of diagnosis these neural networks can offer. This is the project where our experiments are stored.</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The PathoSpotter project emerged from the desire and enthusiasm of the pathologist Dr. Washington Luís, from the Oswald Cruz Foundation (Brazil), for improving the clinical-pathological diagnoses for renal diseases. In 2014, Dr. Washington met Dr. Angelo Duarte from the State University of Feira de Santana (Brazil) and both start the building of the project PathoSpotter.

Initially, the project aimed to assist pathologists in the identification of glomerular lesions from kidney biopsies. As the work evolved, the goal shifted to make large-scale correlations of lesions with clinical and historical data of patients and to create a set of tools to aid the pathologists' daily practices.

Currently, the project Pathospotter intends to offer computational tools to:
* To facilitate the consensus of clinicians and pathologists, helping to achieve more accurate and faster diagnoses;
* To facilitate large-scale clinical-pathological correlations studies;
* To help pathologists and students in classifying lesions in kidney biopsies.

### Built With

* [Python 3.7.9](https://www.python.org/downloads/release/python-379/)
* [Tensorflow](https://pypi.org/project/tensorflow/2.2.0/)
* [Pandas](https://pypi.org/project/pandas/)
* [Scikit-Learn](https://pypi.org/project/scikit-learn/)

## Getting Started

In these folders, there are several experiments. Each will have a unique dependency and version of library. However, it is almost sure you can run any of these files independently using the requirements file in our dev folder. Remember: these experiments were made to train large networks with hundreds of images. Remember to have your dataset and patience ready when pressing enter.

### Installation

This tutorial assumes you already have a valid Python instalation. If not, please check the website above for tutorials.

1. Clone the repo
   ```sh
   git clone https://github.com/LACAD/pathospottersearch-engine.git
   ```
2. Inside the dev folder, create a new environment.
   ```sh
   python3 -m venv env
   ```

3. Use the new environment.
  * On Windows, you can:
     ```sh
     env\Scripts\activate.bat
     ```
   * On Linux, you can:
     ```sh
     source env/bin/activate
     ```

4. Inside the dev folder, install the libraries specified by us in the requirement.txt file.
   ```sh
   pip install -r requirements.txt
   ```

5. Run the desired script.
   ```sh
   python3 that_awesome_script.py
   ```

## Usage

First, follow the steps defined in installation. Then, just use the results wherever you want to. _Ta-da!_

## Tips

If there is a missing library, you can probably find in this [website](https://pypi.org).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Ellen C. Aguiar - chalegreaguiar@gmail.com  
Ângelo A. Duarte - angeloduarte@uefs.br  
Project Link: [here](https://github.com/LACAD/pathospottersearch-engine/).  

<!--## Acknowledgements -->

<!--* []()
* []()
* []()
-->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username


