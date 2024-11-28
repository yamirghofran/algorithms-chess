# Algorithms Project - Chess
Yousef Amirghofran, Lea Aboujaoudé, Diana Cordovez, Kareem Ramil Jamil

# Python Backend
Documentation is provided in comments in the code. (Time complexity is provided for each function.)
## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - For Windows:
     ```
     venv\Scripts\activate
     ```
   - For macOS and Linux:
     ```
     source venv/bin/activate
     ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Flask Backend

1. Make sure the virtual environment is activated.

2. Navigate to the `python-backend` directory:
   ```
   cd python-backend
   ```

3. Run the Flask application:
   ```
   python main.py
   ```

   The Flask backend will start running on `http://localhost:5000`.


# React Frontend
## Setup

1. Install Bun:
   - Visit the Bun website: https://bun.sh/
   - Follow the installation instructions for your operating system

2. Navigate to the `react-frontend` directory:
   ```
   cd react-frontend
   ```

3. Install the project dependencies using Bun:
   ```
   bun install
   ```

## Running the React Frontend

1. Make sure you are in the `react-frontend` directory.

2. Start the development server:
   ```
   bun run dev
   ```

   The React frontend will start running on `http://localhost:5173`.

3. Open your web browser and visit `http://localhost:5173` to see the application.

The reset board button is buggy and has to be pressed twice.