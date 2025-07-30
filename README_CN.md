[English](README.md) | [简体中文](#简体中文)

<a name="简体中文"></a>

# A2A 隐写通信系统

本项目实现了一个基于代理对代理（A2A）协议的隐写通信系统。它利用大型语言模型（LLM）将秘密消息嵌入到看似正常的对话中，以确保数据的隐蔽传输。

## 项目特性

- **隐蔽Agent通信协议**: 设计并实现了一套完整的协议，用于在Agent之间进行安全的隐蔽通信。
- **多种隐写算法**: 支持多种算法，包括 `AC`（算术编码）、`discop` 以及 Artifacts Framework（`differential_based`, `binary_based`, `stability_based`）。
- **A2A 协议集成**: 基于 A2A 通信协议构建，用于标准化的代理交互。
- **动态校验码**: 根据消息长度自动选择校验码算法（`CRC-16`, `SHA-256`, `BLAKE2s-128`），以确保数据完整性。
- **LLM 驱动**: 使用大型语言模型生成载体消息，使通信看起来更自然。
- **可配置**: 可通过 `config.py` 文件轻松配置不同的模型、算法和密钥。

## 项目结构

```
.
├── config.py                 # 系统配置文件
├── server/                   # 服务端代码
│   ├── main.py              # 服务端启动入口
│   └── a2aserver/           # 服务端核心模块
│       ├── agent.py         # Agent 实现
│       └── agent_executor.py # Agent 执行器
├── client/                   # 客户端代码
│   ├── main.py              # 客户端启动入口
│   └── a2aclient/           # 客户端核心模块
│       └── client.py        # 客户端实现
├── modules/                  # 核心功能模块
│   ├── checkcode/           # 校验码管理
│   ├── logging/             # 日志管理
│   ├── math/                # 哈希与数学工具函数
│   ├── package_head/        # 数据包头管理
│   ├── stego/               # 隐写核心算法
│   └── timestamp/           # 时间戳管理
├── data/                     # 数据目录 (用于存放问题、日志等)
└── requirements.txt          # 项目依赖
```

## 核心模块详解

### 1. `modules/checkcode` - 校验码管理
提供多级校验码生成和验证机制，以确保消息完整性。它支持一个四级系统，可根据消息长度自动选择合适的校验码算法。

### 2. `modules/math` - 数学工具
提供多种数学和哈希函数，包括 `CRC-16`、`SHA-256`、`BLAKE2s-128`，以及二进制、十六进制和 base64 格式之间的转换。

### 3. `modules/package_head` - 包头管理
处理数据包的头部信息，支持大消息的分段传输。头部包含总数据段（TDS）、段号（SN）、结束标志位（F）以及一个4位的CRC校验码。

- **TDS** (12位): 传输数据段总数
- **SN** (6位): 数据段序号 (0-63)
- **F** (1位): 结束标志位
- **Checksum** (4位): CRC-4 校验码，确保包头完整性

**包头格式**:
- **第一个包**: TDS(12) + SN(6) + F(1) + Checksum(4) = 23位
- **后续包**: SN(6) + F(1) + Checksum(4) = 11位

### 4. `modules/stego` - 隐写核心模块
包含隐写算法的核心实现。它与大型语言模型集成，执行秘密消息的编码和解码。

### 5. `modules/timestamp` - 时间戳管理
实现了一个基于密钥的时间戳验证机制，用于安全地发起和确认隐写通信。

## 安装步骤

1.  克隆仓库：
    ```bash
    git clone https://github.com/haha1128/a2a-stego-project.git
    cd a2a-stego-project
    ```

2.  安装所需依赖：
    ```bash
    pip install -r requirements.txt
    ```

3.  下载 NLTK 的 `punkt` 分词器数据：
    ```python
    import nltk
    nltk.download('punkt')
    ```

## 配置 (`config.py`)

`config.py` 文件包含了系统的核心配置。主要配置项如下：

- **`AGENT_MODEL_CONFIG`**: 配置服务端用于生成公开答复的 Agent 模型。
  ```python
  AGENT_MODEL_CONFIG={
    "model": "gemini-2.0-flash",
    "api_key": "YOUR_API_KEY_HERE",
  }
  ```

- **`LLM_CONFIG`**: 配置用于隐写生成的语言模型参数。
  ```python
  LLM_CONFIG={
    "topk": 50,
    "max_new_tokens": 256,
    ...
  }
  ```

- **`ALGORITHM_CONFIG`**: 配置不同隐写算法的特定参数，如 `seed`（随机种子）和 `precision`（精度）。
  ```python
  ALGORITHM_CONFIG = {
      "seed": 42,
      "discop": {
          "precision": 52,
          ...
      }
  }
  ```
**注意**: 隐写所使用的 **模型路径** 和具体 **算法名称** 是通过命令行参数在启动客户端和服务器时指定的，而不是在此文件中配置。详情请参阅"使用方法"部分。

## 使用方法

该项目采用客户端-服务器架构。您需要在一个终端中启动服务器，然后在另一个终端中运行客户端以发起通信。

### 1. 启动服务器

打开一个终端并运行以下命令来启动服务器：

```bash
python server/main.py [OPTIONS]
```

**服务器参数:**
*   `--stego_model_path`: 隐写语言模型的路径。
*   `--stego_algorithm`: 要使用的隐写算法 (例如, `discop`)。
*   `--stego_key`: 用于隐写的密钥。
*   `--decrypted_bits_path`: 用于保存解密后秘密消息的路径。
*   `--session_id`: 通信会话的唯一 ID。
*   `--server_url`: 服务器监听的 URL (例如, `http://0.0.0.0:9999`)。

**示例:**
```bash
python server/main.py \
    --stego_model_path "/path/to/your/model" \
    --stego_algorithm "discop" \
    --decrypted_bits_path "data/stego/decrypted_bits.txt" \
    --session_id "my-secret-session-123"
```
服务器将启动并等待客户端连接。

### 2. 启动客户端

打开第二个终端并运行以下命令来启动客户端：

```bash
python client/main.py [OPTIONS]
```

**客户端参数:**
*   `--stego_model_path`: 隐写语言模型的路径 (必须与服务器匹配)。
*   `--stego_algorithm`: 要使用的隐写算法 (必须与服务器匹配)。
*   `--question_path`: 包含握手问题的文件的路径。
*   `--question_index`: 要从文件中使用的问题索引 (从 0 开始)。
*   `--secret_bit_path`: 包含秘密消息 (二进制格式) 的文件的路径。
*   `--session_id`: 与服务器使用的会话 ID 相同。
*   `--server_url`: 要连接的服务器的 URL (例如, `http://localhost:9999`)。
*   `--stego_key`: 用于隐写的密钥 (必须与服务器匹配)。

**示例:**
```bash
python client/main.py \
    --stego_model_path "/path/to/your/model" \
    --stego_algorithm "discop" \
    --question_path "data/question/general.txt" \
    --question_index 0 \
    --secret_bit_path "data/stego/secret_bits_512.txt" \
    --session_id "my-secret-session-123"
```
客户端将连接到服务器并开始隐写通信。

## 通信流程

1.  **握手**: 客户端通过发送带有特殊时间戳的消息与服务器建立安全通道。
2.  **隐写传输**: 客户端将秘密消息嵌入一系列看似正常的问答或陈述中，并发送给服务器。
3.  **消息重组**: 服务器接收消息，从每个消息中提取隐藏的比特，并重组完整的秘密消息。
4.  **完整性验证**: 服务器使用嵌入的校验码验证重组消息的完整性。
5.  **确认**: 服务器使用另一个特殊时间戳向客户端发送确认，以示接收成功或失败。

## 许可证

本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件。

---
[返回顶部](#a2a-隐写通信系统)
