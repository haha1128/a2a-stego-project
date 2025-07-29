#!/usr/bin/env python3
import gc
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
from modules.checkcode.checkcode_mannager import CheckCodeMannager
from openai import AsyncOpenAI
import os
from modules.logging.logging_mannager import LoggingMannager
from a2a.server.agent_execution import RequestContext
from modules.timestamp.timestamp_mannager import TimestampMannager
from modules.math.math import Math
from modules.package_head.package_head_mannager import PackageHead
from modules.stego.stego import decrypt

logger = LoggingMannager.get_logger(__name__)
class ServerAgent:
    """
    Generates responses using the OpenAI API.
    Supports integration with the A2A protocol.
    """
    def __init__(self,stego_model_path,stego_algorithm,stego_key,decrypted_bits_path,session_id) -> None:
        self.stego_model_path = stego_model_path
        self.stego_algorithm = stego_algorithm
        self.stego_key = stego_key
        self.decrypted_bits_path = decrypted_bits_path
        self.session_id = session_id  # Add session_id attribute
        
        self.enable_stego = False
        self.TDS = 0
        self.SN = 0
        self.secret_message = ""
        self.package_head = PackageHead()
        self.checkcode_handler = CheckCodeMannager()
        
        # Lazy load the model
        self.stego_model = None
        self.stego_tokenizer = None
        self.is_loaded_stego_model = False

        # Initialize the OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=config.AGENT_MODEL_CONFIG["api_key"],
            base_url="https://your_openai_compatible_api_base_url/v1" # TODO: Please replace with your actual base_url
        )
        
        # Store conversation history
        self.conversation_history = {}
        
        logger.info(f"Steganography model path: {self.stego_model_path}")
        logger.info(f"Steganography algorithm: {self.stego_algorithm}")
        logger.info(f"Session ID: {self.session_id}")  # Add session_id log


    async def send_message_to_agent(self, query: str, user_id: str = "default_user") -> str:
        """
        Calls the OpenAI API to process a query.
        
        Args:
            query: The user's question.
            user_id: The user's ID, used to distinguish conversations of different users.
            
        Returns:
            The generated response.
        """
        
        # Get or initialize the user's conversation history
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        # Add the user's message to the history
        self.conversation_history[user_id].append({
            "role": "user",
            "content": query
        })
        
        try:
            # Call the OpenAI API
            response = await self.openai_client.chat.completions.create(
                model=config.AGENT_MODEL_CONFIG["model"],
                messages=[
                    {"role": "system", "content": "answer the question simply and directly"},
                    *self.conversation_history[user_id]
                ]
            )
            
            assistant_response = response.choices[0].message.content
            
            # Add the response to the history
            self.conversation_history[user_id].append({
                "role": "assistant", 
                "content": assistant_response
            })
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "Sorry, an error occurred while processing your request."

    async def response_client_message(self, context: RequestContext, user_id: str = "default_user"):
        """
        Processes messages sent from the client.
        Args:
            context: The request context.
            user_id: The user ID.
        Returns:
            The response from the large model, and a timestamp (if verification passes).
        """
        if(not self.is_loaded_stego_model):
            self.load_stego_model()
        user_input = context.get_user_input()
        if(not self.enable_stego):
            if(TimestampMannager.is_valid_timestamp(context.message.metadata.get('sendTimestamp'),lambda x:Math.calculate_sha256_binary(self.stego_key+str(x)).endswith('0'))):
                self.enable_stego = True
                self.SN = 0
                logger.info("Steganography enabled successfully.")
                answer = await self.send_message_to_agent(user_input, user_id=user_id)
                return answer,None
            else:
                logger.info("Failed to enable steganography.")
                raise
        
        logger.info("Steganography is enabled, starting to decrypt the message.")
              
        chat_history = await self.get_chat_history(user_id)
        base_prompt = config.LLM_CONFIG["base_prompt"]
        decrypted_bits, _, _ = decrypt(self.stego_model,self.stego_tokenizer,self.stego_algorithm,user_input, base_prompt.format(conversation_history=chat_history))
        answer = await self.send_message_to_agent(user_input, user_id=user_id)      
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Decrypted bitstream: {decrypted_bits}")
        # Process header and reassemble data packets
        is_final = False
        if(self.SN == 0):
            head_lenth = 23
            self.TDS = int(decrypted_bits[:12], 2)            
            SN = int(decrypted_bits[12:18], 2)
            assert SN == 0
        else:
            head_lenth = 11
            SN = int(decrypted_bits[:6], 2)
        
        logger.info(f"head_lenth:{head_lenth}")
        head_bits = decrypted_bits[:head_lenth]
        
        is_final = bool(int(head_bits[-5:-4], 2))
        head_checkcode_receive = head_bits[-4:]
        # Calculate by removing the last four bits
        head_checkcode_calculate = Math.calculate_crc4_binary(head_bits[:head_lenth-4])
        logger.info(f"head_checkcode_receive: {head_checkcode_receive}")
        logger.info(f"head_checkcode_calculate: {head_checkcode_calculate}")
        assert head_checkcode_receive == head_checkcode_calculate
        self.SN += 1
        self.secret_message += decrypted_bits[head_lenth:]
        # Perform checksum verification only when the last packet (F=1) is received
        timestamp = datetime.now().timestamp()
        if is_final:
            # Verify the checksum
            checkcode_length = self.checkcode_handler.get_checkcode_length_from_tier(self.checkcode_handler.get_checkcode_tier_from_length(self.TDS))
            if len(self.secret_message) >= self.TDS + checkcode_length:
                # Separate the ciphertext and the checksum
                received_message = self.secret_message[:self.TDS]
                received_checksum = self.secret_message[self.TDS:self.TDS + checkcode_length]
                is_pass,_ =self.checkcode_handler.verify_checkcode(received_message, received_checksum)
                if(is_pass):
                    logger.info("Checksum passed, returning specific timestamp.")
                    timestamp = TimestampMannager.get_valid_timestamp(lambda x: Math.calculate_sha256_binary(self.stego_key+str(x)).endswith('0'))
                    logger.info(f"Successfully found timestamp.") if timestamp is not None else logger.error("Failed to find timestamp.")
                else:
                    logger.warning("Checksum failed, returning specific timestamp.")
                    timestamp = TimestampMannager.get_valid_timestamp(lambda x: Math.calculate_sha256_binary(self.stego_key+str(x)).endswith('1'))
                    logger.info(f"Successfully found timestamp.") if timestamp is not None else logger.error("Failed to find timestamp.")
            else:
                logger.warning(f"Message too short for checksum verification. Length: {len(self.secret_message)}, Expected: {self.TDS + checkcode_length}")
                logger.warning("Checksum failed, returning specific timestamp.")
                timestamp = TimestampMannager.get_valid_timestamp(lambda x: Math.calculate_sha256_binary(self.stego_key+str(x)).endswith('1'))
                logger.info(f"Successfully found timestamp.") if timestamp is not None else logger.error("Failed to find timestamp.")
            self.secret_message = self.secret_message[:self.TDS]
            with open(self.decrypted_bits_path, 'w') as f:
                f.write(self.secret_message)
            logger.info(f"This conversation has ended.")
            await self.clear_all_user_data(user_id)
            self.enable_stego = False
            self.TDS = 0
            self.SN = 0
            self.secret_message = ""
        return answer,timestamp
            
    async def get_chat_history(self, user_id: str) -> str:
        """
        Gets all chat history from a user's session and formats it into a readable string.
        
        Args:
            user_id: The user's ID.
            
        Returns:
            The formatted chat history string.
        """
        try:
            # Get chat history from the in-memory session history
            if user_id not in self.conversation_history or not self.conversation_history[user_id]:
                return "No chat history yet."
            
            # Format the chat history
            formatted_lines = []
            for message in self.conversation_history[user_id]:
                role = message["role"]
                content = message["content"].strip()
                
                if role == 'user':
                    formatted_lines.append(f"User: {content}")
                elif role == 'assistant':
                    formatted_lines.append(f"Expert: {content}")
                else:
                    formatted_lines.append(f"{role}: {content}")
            
            # Convert to a string
            formatted_lines_str = ""
            for line in formatted_lines:
                formatted_lines_str += line + "\n"
            return formatted_lines_str
            
        except Exception as e:
            logger.error(f"Error getting formatted chat history for user {user_id}: {e}")
            return "An error occurred while getting the chat history."


    def load_stego_model(self):
        self.stego_model = AutoModelForCausalLM.from_pretrained(self.stego_model_path).half().cuda()
        self.stego_tokenizer = AutoTokenizer.from_pretrained(self.stego_model_path)
        if self.stego_tokenizer.pad_token is None:
            self.stego_tokenizer.pad_token = self.stego_tokenizer.eos_token
        self.is_loaded_stego_model = True
    
    async def clear_all_user_data(self, user_id: str) -> bool:
        """
        Deletes the data of a specified user.

        Args:
            user_id: The user's ID.
            
        Returns:
            bool: Whether the deletion was successful.
        """
        try:
            # Clear the user's conversation history
            if user_id in self.conversation_history:
                del self.conversation_history[user_id]
            return True
            
        except Exception as e:
            logger.error(f"An error occurred while deleting data for user {user_id}: {e}")
            return False