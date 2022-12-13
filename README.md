<div id="top"></div>





<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">ARMOR</h3>

  <p align="center">
    GAN-Based Mitigation of Edge-Case Backdoor Attacks in Federated Learning

  </p>


</div>


 Software of the paper submitted to [IMWUT 2023]

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


This project contains the code for ARMOR, a FL defense mechanism against edge case backdoors attacks, that leverages adversarial learning to uncover edge-case backdoors. More precisely, ARMOR relies on GANs to extract data features from model updates and uses the generated samples to activate potential backdoors in the model.  Our experimental evaluations with various datasets and neural network models show that ARMOR can counter edge-case backdoors with 95% resilience against attacks, and without impacting model quality.

<!-- PROJECT LOGO  <p align="right">(<a href="#top">back to top</a>)</p>  -->



### Built With



* [Python 3.7](https://www.python.org/downloads/release/python-370/)
* [Pytorch](https://pytorch.org/)

<!-- PROJECT LOGO  <p align="right">(<a href="#top">back to top</a>)</p>  -->


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

<!-- PROJECT LOGO  <p align="right">(<a href="#top">back to top</a>)</p>  -->



<!-- USAGE EXAMPLES -->
## Usage

to run the code use:
   ```sh
   python3 main.py -detector .. -start_round ..
   ```
   
   All the arguments can be found in the file src/parser.py




## Structure

The code for state-of-the-art defense mechanisms can be found on the repository src/StateOfTheArtDefenses

The code for armor can be found on the repository src/ARMOR 

The code for edge-case backdoor can be found on the repository src/train

