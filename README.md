# FastAPI Project Setup Guide

This guide outlines the steps to set up and run the FastAPI project locally.

## Prerequisites

-   Python 3.9+
-   pip (Python package installer)
-   A CUDA-enabled GPU with CUDA installed. NVDIA GPUs come with CUDA by default. To install or 
    upgrade CUDA version, click [here](https://developer.nvidia.com/cuda-downloads).


## Setup Instructions
1. **Installing Pytorch:**

    Based on your Operating System and Version of CUDA, you need to install PyTorch for the model to run. Based on your OS and CuDA version, 
    please visit [here](https://pytorch.org/get-started/locally/#start-locally) to get the install instructions. 

2. **Install Ollama:**
    In order to download the model locally to your hardware, you must install Ollama using the following command:

    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

3. **Download the Model:**
    Use the `ollama run` command to download the model as shown below:
    ```bash
    ollama run gemma3:4b-it-qat
    ```

    To confirm that the model is up and running run the `ollama list` command.
    You should see the following: 

    NAME             ID           SIZE  PROCESSOR   UNTIL
    gemma3:4b-it-qat d01ad…       6.6 GB 100% GPU   4 minutes from now

4.  **Create a virtual environment:**

    Navigate to the project directory in your terminal and create a virtual environment using the following command:

    ```bash
    python -m venv venv
    ```

5.  **Activate the virtual environment:**

    -   On Windows:

        ```bash
        venv\Scripts\activate
        ```

    -   On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

6.  **Install dependencies:**

    Install the required packages from the `requirements.txt` file using pip:

    ```bash
    pip install -r requirements.txt

    ```
    

7.  **Run the FastAPI application:**

    Start the FastAPI server using uvicorn:

    ```bash
    python server.py
    ```

8.  **Access the application:**

    Congratulations, the app is now running on http://0.0.0.0:8080. 

## Additional Notes

-   Ensure that you have the necessary environment variables configured as required by the application.
-   Refer to the FastAPI documentation for more details and advanced usage: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)