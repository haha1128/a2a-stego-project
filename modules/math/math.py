from datetime import datetime, timezone
import hashlib
import crcmod
import crcmod.predefined
import base64
class Math:
    def __init__(self):
        pass
    
    @staticmethod
    def timestamp_to_iso8601(timestamp: float) -> str:
        """
        Converts a Unix timestamp to ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ).
        
        Args:
            timestamp: The Unix timestamp (float).
            
        Returns:
            The time string in ISO 8601 format.
        """
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    @staticmethod
    def iso8601_to_timestamp(iso_string: str) -> float:
        """
        Converts an ISO 8601 format time string to a Unix timestamp.
        
        Args:
            iso_string: The time string in ISO 8601 format.
            
        Returns:
            The Unix timestamp (float).
        """
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt.timestamp()
    
    @staticmethod
    def current_timestamp_iso8601() -> str:
        """
        Gets the current time as an ISO 8601 format string.
        
        Returns:
            The current time string in ISO 8601 format.
        """
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    @staticmethod
    def binary_string_to_base64(binary_str: str) -> str:
        """
        Converts a binary string ('01' format) to base64 encoding.
        
        Args:
            binary_str: A string consisting of '0's and '1's.
            
        Returns:
            A base64 encoded string.
        """
        # Ensure the binary string length is a multiple of 8 (pad with zeros on the left).
        remainder = len(binary_str) % 8
        if remainder != 0:
            binary_str = '0' * (8 - remainder) + binary_str
        
        # Convert the binary string to bytes.
        byte_data = bytearray()
        for i in range(0, len(binary_str), 8):
            byte_chunk = binary_str[i:i+8]
            byte_value = int(byte_chunk, 2)
            byte_data.append(byte_value)
        
        # Convert to base64.
        base64_encoded = base64.b64encode(bytes(byte_data)).decode('utf-8')
        return base64_encoded
    
    @staticmethod
    def base64_to_binary_string(base64_str: str) -> str:
        """
        Converts a base64 string back to a binary string ('01' format).
        
        Args:
            base64_str: A base64 encoded string.
            
        Returns:
            A binary string consisting of '0's and '1's.
        """
        # Decode base64.
        byte_data = base64.b64decode(base64_str.encode('utf-8'))
        
        # Convert to binary string.
        binary_str = ''.join(format(byte, '08b') for byte in byte_data)
        return binary_str

    @staticmethod
    def string_to_binary(message: str) -> str:
        """
        Converts an arbitrary string to a binary string ('01' format).
        Args:
            message: The arbitrary string to convert.
        Returns:
            A binary string ('01' format).
        """
        byte_data = message.encode('utf-8')
        return ''.join(format(byte, '08b') for byte in byte_data)

    @staticmethod
    def hex_to_binary(hex_string: str) -> str:
        """
        Converts a hexadecimal string to a binary string ('01' format).
        Args:
            hex_string: The hexadecimal string.
        Returns:
            A binary string ('01' format).
        """
        return bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)
    @staticmethod
    def calculate_crc4_binary(message: str) -> str:
        """
        Calculates the CRC-4 checksum of an arbitrary string and returns a binary string.
        Args:
            message: The arbitrary string for which to calculate the checksum.
        Returns:
            A 4-bit binary string of the CRC-4 checksum.
        """
        # Convert the string to byte data.
        byte_data = message.encode('utf-8')
        
        # Manually implement the CRC-4 algorithm using the polynomial x^4 + x + 1 (0x13).
        # Initialize CRC value to 0.
        crc = 0
        
        # Process each byte.
        for b in byte_data:
            # XOR the byte with the high 4 bits of the CRC.
            crc ^= (b << 4) & 0xF0
            
            # Process the 8 bits of this byte.
            for _ in range(8):
                # If the most significant bit is 1, shift and XOR with the polynomial.
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x13) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        
        # Take the high 4 bits of the final result as the CRC-4 value.
        crc_value = (crc >> 4) & 0x0F
        
        # Convert to a 4-bit binary string.
        return format(crc_value, '04b')
    
    @staticmethod
    def calculate_crc16_binary(message: str) -> str:
        """
        Calculates the CRC-16 checksum of an arbitrary string and returns a binary string.
        Args:
            message: The arbitrary string for which to calculate the checksum.
        Returns:
            A 16-bit binary string of the CRC-16 checksum.
        """
        # Convert the string to byte data.
        byte_data = message.encode('utf-8')
        
        # Use the predefined CRC-16 algorithm.
        crc16_func = crcmod.predefined.mkPredefinedCrcFun('crc-16')
        crc_value = crc16_func(byte_data)
        
        # Convert to a 16-bit binary string.
        return format(crc_value, '016b')

    @staticmethod
    def calculate_sha256_truncated_64_binary(message: str) -> str:
        """
        Calculates the SHA-256 hash of an arbitrary string and returns the first 64 bits as a binary string.
        Args:
            message: The arbitrary string to be hashed.
        Returns:
            A 64-bit binary string of the first 64 bits of the SHA-256 hash.
        """
        # Convert the string to byte data.
        byte_data = message.encode('utf-8')
        hex_hash = hashlib.sha256(byte_data).hexdigest()
        # Take the first 16 hexadecimal characters (64 bits).
        truncated_hex = hex_hash[:16]
        return Math.hex_to_binary(truncated_hex)

    @staticmethod
    def calculate_blake2s_128_binary(message: str) -> str:
        """
        Calculates the BLAKE2s-128 hash of an arbitrary string and returns a binary string.
        Args:
            message: The arbitrary string to be hashed.
        Returns:
            A 128-bit binary string of the BLAKE2s-128 hash.
        """
        # Convert the string to byte data.
        byte_data = message.encode('utf-8')
        hash_obj = hashlib.blake2s(byte_data, digest_size=16)  # 16 bytes = 128 bits
        hex_hash = hash_obj.hexdigest()
        return Math.hex_to_binary(hex_hash)

    @staticmethod
    def calculate_sha256_binary(message: str) -> str:
        """
        Calculates the full SHA-256 hash of an arbitrary string and returns a binary string.
        Args:
            message: The arbitrary string to be hashed.
        Returns:
            A 256-bit binary string of the full SHA-256 hash.
        """
        # Convert the string to byte data.
        byte_data = message.encode('utf-8')
        hex_hash = hashlib.sha256(byte_data).hexdigest()
        return Math.hex_to_binary(hex_hash)
    
    @staticmethod
    def binary_to_hex(binary_str: str) -> str:
        """
        Converts a binary string ('01' format) to a hexadecimal string.
        
        Args:
            binary_str: A string of '0's and '1's, e.g., "0110100001100101".
            
        Returns:
            A hexadecimal string, e.g., "6865".
        """
        # Ensure the binary string length is a multiple of 4 (pad with zeros on the left).
        remainder = len(binary_str) % 4
        if remainder != 0:
            binary_str = '0' * (4 - remainder) + binary_str
        
        # Convert the binary string to hexadecimal.
        hex_result = ""
        for i in range(0, len(binary_str), 4):
            # Take a 4-bit chunk.
            chunk = binary_str[i:i+4]
            # Convert to decimal then to hexadecimal.
            hex_digit = hex(int(chunk, 2))[2:]  # [2:] to remove the '0x' prefix.
            hex_result += hex_digit
        
        return hex_result.upper()  # Return uppercase hexadecimal.