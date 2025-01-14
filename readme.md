## Install

**Install**

```shell
# clone this repo first
cd text2graph
pip install -e .
```



> **Please set API key and BASE_URL in environment: `export API_KEY="sk-..."` `export BASE_URL="https://..."`.**


### Function: `extraction`

**Description:**
This function takes a list of strings representing the documents and returns a json object representing the result of the extraction.

**Parameters:**
- `chunk_list` (list[str]): A list of strings representing the documents to be inserted.

**Returns:**
- `dict`: A json object representing the result of the extraction.

**Example Usage:**
```python
from text2graph import extraction
with open("book.txt", encoding="utf-8-sig") as f:
    scope = f.read()
result = extraction([scope])
print(result)

```
**run demo**
```shell
cd demo
python main.py
```