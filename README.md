# enron-search
Simple CLI for keyword searching over a subset of the Enron documents

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/enron-search.git
    cd enron-search
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install uv
    uv sync
    ```

## Usage

1. Ensure you have the `enron-search-data.json` file in the project directory.

2. Run the main script:
    ```sh
    python main.py
    ```

3. Enter your search query when prompted and view the top results.

## Example

```
Your query: energy crisis
You entered: energy crisis
Here are the top 10 results for your query:

Doc ID: 12345
Score: 0.85
Extracted Text: 
... (text from the document) ...
================================================================================
...
```