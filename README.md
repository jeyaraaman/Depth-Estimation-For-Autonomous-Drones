# Wire Detection Project

This project implements a YOLO-based wire detection system. 

## Setup Instructions

1. Clone this repository.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## External Assets

This repository relies on large external assets (datasets and model weights) which are not stored in version control due to their size.

### Datasets

The datasets folder contains the training and validation images/labels.

**Download link:** `[INSERT GOOGLE DRIVE / ONEDRIVE LINK HERE]`

Instructions:
- Download the ZIP file from the link above.
- Extract the contents directly into a folder named `datasets/` in the root of this project.

### Model Weights

The pre-trained model weights (`best.pt` or similar `*.pt` files) are hosted using GitHub Releases.

**Download link:** `[INSERT GITHUB RELEASE LINK HERE]`

Instructions:
- Download the required model weights (`.pt` files).
- Place them in the root of this project.

## Running the Code

Ensure the data and weights are properly placed before running the training or prediction scripts.
