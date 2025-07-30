[简体中文](README_CN.md) | [English](#english)

<a name="english"></a>

# A2A Steganography Communication System

This project implements a steganographic communication system based on the Agent-to-Agent (A2A) protocol. It leverages Large Language Models (LLMs) to embed secret messages within seemingly normal conversations, ensuring covert data transmission.

## Features

- **Covert Agent Communication Protocol**: Designed and implemented a complete protocol for secure, covert communication between agents.
- **Multiple Steganography Algorithms**: Supports various algorithms including `AC` (Arithmetic Coding), `discop`, and the Artifacts Framework (`differential_based`, `binary_based`, `stability_based`).
- **A2A Protocol Integration**: Built upon the A2A communication protocol for standardized agent interaction.
- **Dynamic Checksum**: Automatically selects checksum algorithms (`CRC-16`, `SHA-256`, `BLAKE2s-128`) based on message length to ensure data integrity.
- **LLM-Powered**: Uses LLMs for generating carrier messages, making the communication appear natural.
- **Configurable**: Easily configurable through `config.py` for different models, algorithms, and keys.

## Project Structure

```
.
├── config.py                 # System configuration file
├── server/                   # Server-side code
│   ├── main.py              # Server entry point
│   └── a2aserver/           # Core server modules
│       ├── agent.py         # Agent implementation
│       └── agent_executor.py # Agent executor
├── client/                   # Client-side code
│   ├── main.py              # Client entry point
│   └── a2aclient/           # Core client modules
│       └── client.py        # Client implementation
├── modules/                  # Core functional modules
│   ├── checkcode/           # Checksum management
│   ├── logging/             # Logging management
│   ├── math/                # Hashing and math utility functions
│   ├── package_head/        # Data packet header management
│   ├── stego/               # Core steganography algorithms
│   └── timestamp/           # Timestamp management
├── data/                     # Data directory (for questions, logs, etc.)
└── requirements.txt          # Project dependencies
```

## Core Modules Explained

### 1. `modules/checkcode` - Checksum Management
Provides a multi-level checksum generation and verification mechanism to ensure message integrity. It supports a four-tier system that automatically selects the appropriate checksum algorithm based on the message length.

### 2. `modules/math` - Math Utilities
Offers various mathematical and hashing functions, including `CRC-16`, `SHA-256`, `BLAKE2s-128`, and conversions between binary, hex, and base64 formats.

### 3. `modules/package_head` - Packet Header Management
Handles the header information for data packets, enabling segmented transmission of large messages. The header includes fields for Total Data Segments (TDS), Segment Number (SN), a Final packet flag (F), and a 4-bit CRC checksum.

- **TDS** (12 bits): Total data segments
- **SN** (6 bits): Segment number (0-63)
- **F** (1 bit): Final packet flag
- **Checksum** (4 bits): A CRC-4 checksum to ensure header integrity

**Header Format**:
- **First Packet**: TDS(12) + SN(6) + F(1) + Checksum(4) = 23 bits
- **Subsequent Packets**: SN(6) + F(1) + Checksum(4) = 11 bits

### 4. `modules/stego` - Core Steganography Module
Contains the core implementations of the steganography algorithms. It integrates with LLMs to perform encoding and decoding of secret messages.

### 5. `modules/timestamp` - Timestamp Management
Implements a key-based timestamp verification mechanism to initiate and confirm the steganographic communication securely.

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/haha1128/a2a-stego-project.git
    cd a2a-stego-project
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the NLTK `punkt` tokenizer data**
    ```python
    import nltk
    nltk.download('punkt')
    ```

## Configuration (`config.py`)

The `config.py` file contains the core configurations for the system. Key settings include:

- **`AGENT_MODEL_CONFIG`**: Configures the agent model used by the server to generate public responses.
  ```python
  AGENT_MODEL_CONFIG={
    "model": "gemini-2.0-flash",
    "api_key": "YOUR_API_KEY_HERE",
  }
  ```

- **`LLM_CONFIG`**: Configures the parameters for the language model used in steganography.
  ```python
  LLM_CONFIG={
    "topk": 50,
    "max_new_tokens": 256,
    ...
  }
  ```

- **`ALGORITHM_CONFIG`**: Configures specific parameters for different steganography algorithms, such as `seed` and `precision`.
  ```python
  ALGORITHM_CONFIG = {
      "seed": 42,
      "discop": {
          "precision": 52,
          ...
      }
  }
  ```
**Note**: The **model path** and the specific **algorithm name** for steganography are provided as command-line arguments when starting the client and server, not in this file. Please see the "Usage" section for details.

## Usage

The project uses a client-server architecture. You need to start the server in one terminal and then run the client in another to initiate communication.

### 1. Start the Server

Open a terminal and run the following command to start the server:

```bash
python server/main.py [OPTIONS]
```

**Server Arguments:**
*   `--stego_model_path`: Path to the steganography LLM.
*   `--stego_algorithm`: Steganography algorithm to use (e.g., `discop`).
*   `--stego_key`: Secret key for steganography.
*   `--decrypted_bits_path`: Path to save the decrypted secret message.
*   `--session_id`: A unique ID for the communication session.
*   `--server_url`: URL for the server to listen on (e.g., `http://0.0.0.0:9999`).

**Example:**
```bash
python server/main.py \
    --stego_model_path "/path/to/your/model" \
    --stego_algorithm "discop" \
    --decrypted_bits_path "data/stego/decrypted_bits.txt" \
    --session_id "my-secret-session-123"
```
The server will start and wait for a client connection.

### 2. Start the Client

Open a second terminal and run the following command to start the client:

```bash
python client/main.py [OPTIONS]
```

**Client Arguments:**
*   `--stego_model_path`: Path to the steganography LLM (must match the server).
*   `--stego_algorithm`: Steganography algorithm to use (must match the server).
*   `--question_path`: Path to the file containing handshake questions.
*   `--question_index`: Index of the question to use from the file (starts at 0).
*   `--secret_bit_path`: Path to the file containing the secret message (in binary format).
*   `--session_id`: The same session ID used for the server.
*   `--server_url`: URL of the server to connect to (e.g., `http://localhost:9999`).
*   `--stego_key`: Secret key for steganography (must match the server).

**Example:**
```bash
python client/main.py \
    --stego_model_path "/path/to/your/model" \
    --stego_algorithm "discop" \
    --question_path "data/question/general.txt" \
    --question_index 0 \
    --secret_bit_path "data/stego/secret_bits_512.txt" \
    --session_id "my-secret-session-123"
```
The client will connect to the server and begin the steganographic communication.

## Communication Flow

1.  **Handshake**: The client initiates contact by sending a message with a special timestamp to the server to establish a secure channel.
2.  **Steganographic Transmission**: The client embeds the secret message into a series of seemingly normal questions or statements and sends them to the server.
3.  **Message Reassembly**: The server receives the messages, extracts the hidden bits from each, and reassembles the complete secret message.
4.  **Integrity Verification**: The server verifies the integrity of the reassembled message using the embedded checksum.
5.  **Acknowledgment**: The server sends a confirmation back to the client using another special timestamp to signal successful or failed reception.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
[Back to top](#a2a-steganography-communication-system)
