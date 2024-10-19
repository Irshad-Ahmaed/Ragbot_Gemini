# Chat with Multiple PDFs using Gemini 💁
This project is a Streamlit-based web application that allows users to upload multiple PDF files and ask questions about the content within them. It uses Google's Generative AI embeddings and FAISS for similarity search to find answers within the provided PDFs.

# Features
- Upload multiple PDF files
- Extract text from the PDFs
- Chunk the text into manageable sections
- Perform similarity searches using FAISS and Google Generative AI embeddings
- Generate responses using Google's Gemini model
- Simple and interactive web interface built with Streamlit

# Live Link:
### Hosted on Striemlit
    - https://q-a-ragbot-with-gemini.streamlit.app

# Requirements
## To run this project, you'll need the following:

- Python 3.7 or higher
- Google Cloud API Key (for Google Generative AI)
- Streamlit for the web interface
- FAISS for similarity search

# Installation
- Clone the repository to your local machine:
    ```sh
    git clone https://github.com/yourusername/chat-with-pdfs-gemini.git
    cd chat-with-pdfs-gemini

- Create a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

- Install the required Python packages:
    ```sh
    pip install -r requirements.txt

- Set up your environment variables:
    Create a .env file in the root directory and add your Google Cloud API key:
    ```sh
    GOOGLE_API_KEY=your_google_api_key

- Run the programme:
    ```sh
    streamlit run ragbot_gemini.py

- Open the app in your browser at http://localhost:8501.

# Application Workflow

- Upload PDFs: Use the sidebar to upload multiple PDF files.
- Ask Questions: After uploading, type a question related to the PDF content in the text input field.
- Get Answers: The app will perform similarity searches on the PDF content and generate a response using Google's Gemini model.


## Example Usage
- Upload a few PDF files related to your topic.
- Ask a question in the input box such as, "What is the main topic of the second document?"
- The app will process the content and provide a relevant response.