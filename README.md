# enron-search
Simple CLI for keyword searching over a subset of the Enron documents

## Installation

Open a terminal and follow these steps to set up the project:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/enron-search.git
   cd enron-search
   ```

2. Install uv and set up the environment
   ```sh
   pip install uv # You might need to do `pip3 install uv` on Linux or Mac
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