from datetime import datetime
import gc
import os
import httpx
import torch
import config
from modules.logging.logging_mannager import LoggingMannager
from typing import Any
from uuid import uuid4
import json
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams,SendMessageRequest
from modules.math.math import Math
from modules.timestamp.timestamp_mannager import TimestampMannager
from modules.package_head.package_head_mannager import PackageHead
from modules.checkcode.checkcode_mannager import CheckCodeMannager
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from modules.stego.stego import encrypt,generate_text
logger = LoggingMannager.get_logger(__name__)

class Client:
    def __init__(self, stego_model_path:str,stego_algorithm:str,question_path:str,question_index:int,stego_key:str,secret_bit_path:str,server_url:str,session_id:str):
        
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        self.stego_model_path = stego_model_path
        self.stego_algorithm = stego_algorithm
        self.question = open(question_path,'r',encoding='utf-8').read().splitlines()[question_index]
        self.stego_key = stego_key
        self.secret_bit = open(secret_bit_path,'r',encoding='utf-8').read()
        self.TDS = len(self.secret_bit)
        self.LLM_CONFIG = config.LLM_CONFIG

        # Some control parameters
        self.chat_history = ""
        self.enable_stego = False
        self.SN = 0
        self.package_head = PackageHead()
        self.checkcode_handler = CheckCodeMannager()

        # Save the server URL, initialize the client later
        self.server_url = server_url
        self.httpx_client = None
        self.a2a_client = None

        # Information to be saved
        self.session_id = session_id
        self.conversation={
            "session_id":self.session_id,
            "sessionInfo":{
                "topic": question_path,
                "questionIndex":question_index,
                "steganographyAlgorithm":stego_algorithm,
                "clientModel":stego_model_path,
                "serverResponderModel":None,
                "keyId":stego_key,
                "initiationRule":"hash(key+ts) ends with '0'"
            },
            "secretMessage":{
                "originalData_base64":Math.binary_string_to_base64(self.secret_bit),
                "totalSizeBytes":len(self.secret_bit),
                "integrityHash_sha256":Math.binary_to_hex(Math.calculate_sha256_binary(self.secret_bit))
            },
            "rounds":[],
            "finalVerification":{
                "serverAckTimestamp":None,
                "verificationSignal":None,
                "status":None
            }
        }

        # Lazy load the steganography model, load after a successful handshake
        self.is_loaded_stego_model = False
        self.stego_model = None
        self.stego_tokenizer = None

        logger.info(f"Model path: {self.stego_model_path}")
        logger.info(f"Steganography algorithm: {self.stego_algorithm}")
        logger.info(f"Session ID: {self.session_id}")


    async def initialize_client(self):
        """Initializes the A2A client."""
        try:
            self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0), trust_env=False)
            resolver = A2ACardResolver(
                httpx_client=self.httpx_client,
                base_url=self.server_url,
            )
            # Get Agent identity
            public_card = await resolver.get_agent_card()
            self.a2a_client = A2AClient(
                httpx_client=self.httpx_client, agent_card=public_card
            )
            self.conversation["sessionInfo"]["serverResponderModel"] = public_card.name
            logger.info(f"Successfully initialized A2A client.")
            
        except Exception as e:
            logger.error(f"Failed to initialize A2A client: {e}")
            raise

    def load_stego_model(self):
        # Load with half precision by default
        self.stego_model = AutoModelForCausalLM.from_pretrained(self.stego_model_path).half().cuda()
        self.stego_tokenizer = AutoTokenizer.from_pretrained(self.stego_model_path)
        self.stego_model.eval()
        if self.stego_tokenizer.pad_token is None:
            self.stego_tokenizer.pad_token = self.stego_tokenizer.eos_token
        self.is_loaded_stego_model = True
        
    async def send_message(self, message_text: str, send_timestamp: float) -> dict[str, Any]:
        """
        Sends a message and automatically adds it to the chat history.
        Args:
            message_text: The message content.
            send_timestamp: The send timestamp.
        Returns:
            dict[str, Any]: The returned response dictionary.
        """
        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': message_text}
                ],
                'messageId': uuid4().hex,
                'metadata': {
                    'sendTimestamp': send_timestamp  # Place the send timestamp in the metadata.
                },
            },
        }
        # Add user message to chat history
        self.chat_history += f"User: {message_text}\n"
        # Send the message
        response = await self.a2a_client.send_message(SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        ))
        response_dict = response.model_dump(mode='json', exclude_none=True)
        # Extract the actual assistant reply content according to the correct format.
        try:
            assistant_content = response_dict['result']['parts'][0].get('text', '')
            self.chat_history += f"Expert: {assistant_content.strip()}\n"
        except:
            logger.error(f"Received an unknown format response from the server: {response_dict}")
            raise
        return response_dict

    async def start_stego_chat(self):
        """
        Starts the steganographic communication.
        """
        # Initialize the client if not already initialized
        if self.a2a_client is None:
            await self.initialize_client()
            
        if(not self.is_loaded_stego_model):
            self.load_stego_model()
        if not await self._ensure_stego_enabled():
            return
        # Add check code
        checkcode, tier = self.checkcode_handler.create_checkcode(self.secret_bit)
        message_with_checkcode = self.secret_bit + checkcode
        logger.info(f"Added tier {tier} check code.")
        # Actual total length to be processed (including check code)
        total_bits_with_checkcode = len(message_with_checkcode)
        processed_bits = 0
        while processed_bits < total_bits_with_checkcode:
            remaining_bits = total_bits_with_checkcode - processed_bits
            logger.info(f"Remaining bits: {remaining_bits}")
            # Get the remaining part of the secret message (including check code)
            remaining_message = message_with_checkcode[processed_bits:]
            # Create packet head
            is_final = False
            tds_value = self.TDS if self.SN == 0 else 0
            header = self.package_head.create_package_head(tds_value, self.SN, is_final)
            # Concatenate the header with the message
            message_with_header = header + remaining_message
            logger.info(f"Processing packet {self.SN+1}")
            prompt = self.LLM_CONFIG["base_prompt"].format(conversation_history=self.chat_history)
            encrypted_text, bits_encoded, _ = encrypt(
                model=self.stego_model,
                tokenizer=self.stego_tokenizer,
                algorithm=self.stego_algorithm,
                bit_stream=message_with_header, 
                prompt_text=prompt
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            normal_text,_ = generate_text(
                model=self.stego_model,
                tokenizer=self.stego_tokenizer, 
                prompt_text=prompt
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Calculate the number of actually encoded secret bits (subtracting the header length)
            header_length = len(header)
            actual_message_bits = bits_encoded - header_length if bits_encoded > header_length else 0
            if actual_message_bits <= 0 and self.SN > 0:
                logger.warning(f"No bits were embedded in packet {self.SN+1}.")
                break
            processed_bits += actual_message_bits
            # Check if it is the last packet
            if processed_bits >= total_bits_with_checkcode:
                is_final = True
                header = self.package_head.create_package_head(0, self.SN, is_final)
                final_message_chunk = message_with_checkcode[processed_bits-actual_message_bits if actual_message_bits > 0 else 0 : total_bits_with_checkcode]
                message_with_header = header + final_message_chunk
                encrypted_text, _, _ = encrypt(
                    model=self.stego_model,
                    tokenizer=self.stego_tokenizer,
                    algorithm=self.stego_algorithm,
                    bit_stream=message_with_header, 
                    prompt_text=prompt
                )
                logger.info(f"Last packet detected.")
            # Send the message and check the response
            client_sendTimestamp = datetime.now().timestamp()
            response_dict = await self.send_message(encrypted_text, client_sendTimestamp)
            server_sendTimestamp = response_dict['result']['metadata'].get('sendTimestamp')
            if(is_final):
                hash_condition = lambda x:Math.calculate_sha256_binary(self.stego_key+str(x)).endswith('0')
                if(TimestampMannager.is_valid_timestamp(server_sendTimestamp,hash_condition)):
                    logger.info("Server successfully decrypted the message, communication terminated.")
                    self.conversation["finalVerification"]["serverAckTimestamp"] = Math.timestamp_to_iso8601(server_sendTimestamp)
                    self.conversation["finalVerification"]["verificationSignal"] = "timestamp_used_by_server_for_ack"
                    self.conversation["finalVerification"]["status"] = "SUCCESS"
                    logger.info(f"Transmission efficiency: {self.TDS / self.SN:.2f} bits/round")
                    logger.info(f"Round overhead: {self.SN / self.TDS:.4f} rounds/bit")
                else:
                    logger.warning("Server verification failed, communication terminated.")
                    self.conversation["finalVerification"]["serverAckTimestamp"] = Math.timestamp_to_iso8601(server_sendTimestamp)
                    self.conversation["finalVerification"]["verificationSignal"] = "timestamp_used_by_server_for_ack"
                    self.conversation["finalVerification"]["status"] = "FAIL"
                    logger.info(f"Transmission efficiency: {self.TDS / self.SN:.2f} bits/round")
                    logger.info(f"Round overhead: {self.SN / self.TDS:.4f} rounds/bit")
            # Information for a single round to be saved
            round={
                "roundNumber":self.SN+1,
                "clientTurn": {
                    "timestamp":Math.timestamp_to_iso8601(client_sendTimestamp),
                    "publicCarrierMessage":encrypted_text,
                    "normalMessage":normal_text,
                    "covertPacket":{
                    "header":{
                        "tds":tds_value,
                        "sn":self.SN,
                        "finFlag":is_final,
                        "checksum_hex":Math.binary_to_hex(header[-4:])
                        },
                    "payload_base64":Math.binary_string_to_base64(message_with_header[header_length:bits_encoded])
                    }
                },
                "serverTurn":{
                    "timestamp":Math.timestamp_to_iso8601(server_sendTimestamp),
                    "publicResponseMessage":response_dict['result']['parts'][0].get('text', '')
                }
            }
            self.conversation["rounds"].append(round)
            with open(f"data/conversation/conversation_{self.session_id}.json","w",encoding="utf-8") as f:
                json.dump(self.conversation,f)
            self.SN += 1
                    
    async def _ensure_stego_enabled(self):
        """Ensures that steganography is enabled."""
        if self.enable_stego:
            return True
        logger.info("Steganography is not enabled, trying to enable it.")
        hash_condition = lambda x:Math.calculate_sha256_binary(self.stego_key+str(x)).endswith('0')
        timestamp = TimestampMannager.get_valid_timestamp(hash_condition)
        if timestamp:
            logger.info(f"Steganography enabled successfully, timestamp: {timestamp}")
            self.SN = 0
            await self.send_message(self.question,timestamp)
            self.enable_stego = True
            return True
        return False
        
            

            
