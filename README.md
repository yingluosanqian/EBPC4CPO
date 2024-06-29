# Exploiting Booster Pass Chain for Compiler Phase Ordering

## 1. Introduction

This repository is currently used to store essential materials for our submission to ASPLOS'25.

## 2. Result

We have placed the descriptions of BPC coreset, BPC set, and related passes from the paper in the quick result directory for easy reference.

- The `quick_result/pass_description.txt` file contains descriptions of the 124 transform passes from LLVM 10.0.0 used in our experiments.
- The quick_result/`BPC_set.txt` file stores the BPC generated at the termination of our iterative algorithm. The passes are described using numbers, each ranging from 0 to 123, corresponding sequentially to the passes listed in the quick_result/`pass_description.txt` file.
- The `quick_result/BPC_coreset.txt` file contains the BPC coreset generated at the termination of our iterative algorithm, consisting of 50 pass sequences. The passes are also described using numbers, each ranging from 0 to 123, corresponding sequentially to the passes listed in the `quick_result/pass_description.txt` file.

## 3. Start

### 3.1 Environment Setup

To simplify the setup process, we provide a Docker image which you can download from the following link:

[Download Docker Image](https://link-to-your-docker-image)

Use the following command to create a Docker container from our provided Docker image:

```sh
docker run -it --name ebpc4cpo ebpc4cpo
```

We use Conda for Python package management, so after creating the Docker container, please activate the Conda environment with the following command:

```sh
conda activate compiler_gym
```

We have not described how to build the project from scratch because the environment setup is quite complex and has encountered various issues across different operating systems. To ensure you can run our code smoothly, we recommend using the provided Docker image.

### 3.2 RQ1: Generating and Evaluating Coreset

#### 3.2.1 Generating BPC Coreset

Use the following command to generate the BPC coreset (ours). Ensure you are in the xxx working directory:

```sh
# Command to generate BPC coreset
```

The results will be stored in the xxx directory.

#### 3.2.2 Generating NVP1 Coreset

Use the following command to generate the NVP-1 coreset. Ensure you are in the xxx working directory:

```sh
# Command to generate NVP-1 coreset
```

The results will be stored in the xxx directory.

The NVP-2 coreset uses the results published in the original paper directly.

#### 3.2.3 Generating ICMC Coreset

Use the following command to generate the ICMC coreset. Ensure you are in the xxx working directory:

```sh
# Command to generate ICMC coreset
```

The results will be stored in the xxx directory.

#### 3.2.4 Generating GENS Coreset

Use the following command to generate the GENS coreset. Ensure you are in the xxx working directory:

```sh
# Command to generate GENS coreset
```

The results will be stored in the xxx directory.

####  3.2.6 Moving All Coresets to a Single Directory for Comparison

After generating all the coresets, you need to move them to the same directory for comparison. Execute the following command:

```sh
# Command to move all coresets to the comparison directory
```

Then execute the following command to evaluate them.

``` sh

```

The results will be stored in the xxx directory.

### 3.3 RQ2: Training and Evaluation of MLP

#### 3.3.1 Training and Evaluating MLP

Use the following command to complete the training and evaluation of MLP: 

```sh
```

The results will be saved in the xxx directory.

#### 3.3.2 Plotting Results

Use the following command to plot the results. The images will be saved in the xxx directory:

``` sh
```

### 3.3 RQ3: Ablation Study

#### 3.3.1 Ablation Studyfor Refine Module

Use the following command to complete the ablation experiment for the Refine module:

``` sh
```

The results will be saved in the xxx directory.

#### 3.3.2 Ablation Experiment for Coreset Generation Module

Use the following command to complete the ablation experiment for the Coreset generation module:

``` sh
```

The results will be saved in the xxx directory, and the related charts will be generated. The output for the Refine module ablation will be printed on the console, while the output for the Coreset generation module ablation will be saved in `ablation_2.pdf` in the current directory.