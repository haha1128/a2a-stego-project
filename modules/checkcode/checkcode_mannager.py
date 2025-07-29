from typing import Tuple
from modules.logging.logging_mannager import LoggingMannager
from modules.math.math import Math
logger=LoggingMannager.get_logger(__name__)
class CheckCodeMannager:
    def __init__(self):
        """Initializes the checksum handler."""
        # Checksum tier definitions
        self.TIER_1_MAX = 64     # Tier 1: 1-64 bits
        self.TIER_2_MAX = 512    # Tier 2: 65-512 bits
        self.TIER_3_MAX = 2048   # Tier 3: 513-2048 bits
        self.TIER_4_MAX = 4096   # Tier 4: 2049-4096 bits
        
        # Checksum lengths (in bits)
        self.CHECKSUM_BITS = {
            1: 16,   # CRC-16
            2: 64,   # Truncated SHA-256
            3: 128,  # BLAKE2s-128
            4: 256   # SHA-256
        }
    
    
    def create_checkcode(self, message: str) -> Tuple[str, int]:
        """
        Generates a checksum for a message.
        
        Args:
            message: The original message (a '01' string).
        Returns:
            tuple: (binary string of the checksum, checksum tier)
        """
        message_length = len(message)
        tier = self.get_checkcode_tier_from_length(message_length)
        
        logger.info(f"Message length: {message_length} bits, using checksum tier {tier}")
        
        if tier == 1:
            # Tier 1: CRC-16
            checksum_bits = Math.calculate_crc16_binary(message)
            logger.info(f"CRC-16 checksum: ({checksum_bits})")
            
        elif tier == 2:
            # Tier 2: Truncated SHA-256 (first 64 bits)
            checksum_bits = Math.calculate_sha256_truncated_64_binary(message)
            logger.info(f"Truncated SHA-256 checksum: ({checksum_bits})")
            
        elif tier == 3:
            # Tier 3: BLAKE2s-128 (128 bits)
            checksum_bits = Math.calculate_blake2s_128_binary(message)
            logger.info(f"BLAKE2s-128 checksum: ({checksum_bits})")
            
        elif tier == 4:
            # Tier 4: Full SHA-256 (256 bits)
            checksum_bits = Math.calculate_sha256_binary(message)
            logger.info(f"SHA-256 checksum: ({checksum_bits})")
        
        return checksum_bits, tier
    def verify_checkcode(self, message: str, received_checksum: str) -> Tuple[bool, int]:
        """
        Verifies a checksum.
        
        Args:
            message: The received message (a '01' string).
            received_checksum: The received checksum (binary string).
            
        Returns:
            tuple: (whether the checksum is valid, checksum tier)
        """
        # Recalculate the checksum
        expected_checksum, tier = self.create_checkcode(message)
        
        # Compare the checksums
        is_valid = expected_checksum == received_checksum
        
        if not is_valid:
            logger.warning("Checksum verification failed.")
            logger.warning(f"Expected checksum: {expected_checksum}")
            logger.warning(f"Received checksum: {received_checksum}")
        
        return is_valid, tier
    def get_checkcode_length_from_tier(self, tier: int) -> int:
        """
        Gets the checksum length from the checksum tier.
        Args:
            tier: The checksum tier.
        Returns:
            The checksum length (in bits).
        """
        return self.CHECKSUM_BITS[tier]
    def get_checkcode_tier_from_length(self, message_length: int) -> int:
        """
        Determines the checksum tier based on the message length.
        Args:
            message_length: The message length (in bits).
            
        Returns:
            The checksum tier (1-4).
        """
        if message_length <= self.TIER_1_MAX:
            return 1
        elif message_length <= self.TIER_2_MAX:
            return 2
        elif message_length <= self.TIER_3_MAX:
            return 3
        elif message_length <= self.TIER_4_MAX:
            return 4
        else:
            raise ValueError(f"Message length {message_length} bits exceeds the maximum supported {self.TIER_4_MAX} bits.")