# Linux PC Expert

Linux PC Expert is a command-line assistant that answers questions using local
Linux documentation and, when needed, information from the current computer.
It uses a LangGraph workflow to classify questions, retrieve relevant document
content, draft an answer, and have a second model review it.

## Features

- Retrieves up to four relevant passages from a Chroma database built from
  `files/linuxfun.pdf`.
- Generates and classifies answers with Google Gemini, then reviews drafts with
  Mistral.
- Creates text embeddings locally with the `all-MiniLM-L6-v2` Sentence
  Transformer model running on CPU.
- Can gather live system information using a human-approved terminal tool.
- Keeps the latest ten question-and-answer pairs available as conversation
  context while the application is running.

## Requirements

- Linux and Python 3.10 or newer.
- API keys for Google Gemini and Mistral.
- The source document `files/linuxfun.pdf`.
- Internet access to install dependencies, call the language models, and
  download the embedding model on first use.

The virtual environment is created inside this project as `.venv`; it is not
included in the requirements file.

## Configuration

Create a `.env` file in the project root:

```dotenv
GOOGLE_API_KEY=your_google_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

Keep `.env` private and do not commit or share your API keys. The application
loads these values with `python-dotenv`.

## Setup and run

Run these commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-linux.txt
python3 DB.py
python3 proj.py
```

`DB.py` reads `files/linuxfun.pdf`, splits it into chunks, and builds the
`Chroma_DB/` vector database. The first database build may take time while the
embedding model is downloaded and the document is processed. If you change the
PDF, rebuild the database so retrieval uses the updated document.

On Debian or Ubuntu, if `venv` is unavailable, install it with:

```bash
sudo apt update
sudo apt install python3-venv
```

At the `You:` prompt, enter a question. Type `exit` or `quit` to stop the
assistant. Deactivate the virtual environment with `deactivate`.

## Terminal tool and approvals

Questions that need current system information may cause the assistant to
request a terminal command. Each tool call is shown in the terminal and must be
approved by entering `yes`; any other response rejects it.

The tool allows only a fixed set of read-oriented commands, runs them without a
shell, applies a 15-second timeout, and truncates output longer than 12,000
characters. Shell operators, pipes, redirection, command substitution, and
specific modifying options such as `find -delete` are rejected. The assistant
is intended for diagnostics, not for modifying the computer.

## Project files

```text
.
├── DB.py                    # Builds the document vector database
├── Chroma_DB/               # Chroma data created by DB.py
├── embeddings.py            # Local Sentence Transformer embeddings
├── files/
│   └── linuxfun.pdf         # Linux documentation used for retrieval
├── proj.py                  # CLI assistant and LangGraph workflow
└── requirements-linux.txt   # Python dependencies
```

Keep `files/linuxfun.pdf` in place when rebuilding the database. The database
and downloaded model cache can use additional disk space.
