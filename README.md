<div id="top"></div>





<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">ARMOR</h3>

  <p align="center">
    GAN-Based Mitigation of Edge-Case BackdoorAttacks in Federated Learning

  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
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
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>

  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


This project contains the code for armor, a FL defense mechanism against edge case backdoors attacks, based on GAN, the software also contains implementations for the state of the art works Multi-Krum, NDC and Trimmed-mean. The software has been implemented in python, using the pytorch package.

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With



* [Python 3.7](https://www.python.org/downloads/release/python-370/)
* [Pytorch](https://pytorch.org/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started



### Prerequisites

All the used packages are in requirements.txt file
* python 3.7
  ```sh
  apt-get install python3.7
  ```

### Installation



1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/ARMOR.git
   ```
3. Install the requirements
   ```sh
   pip3 install -r requirements.txt
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

to run the code use:
   ```sh
   python3 main.py -detector .. -start_round ..
   ```
   
   All the arguments are in the file <a href="/src/parser.py">parser</a>


<p align="right">(<a href="#top">back to top</a>)</p>




