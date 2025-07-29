from typing import Optional
from modules.math.math import Math


class PackageHead:
    """
    Package header handler, responsible for encoding and decoding the TDS, SN, and F header parameters.
    """
    
    def __init__(self):
        """Initializes the package header handler."""
        self.TDS_BITS = 12  # TDS field length is 12 bits
        self.SN_BITS = 6    # SN field length is 6 bits
        self.F_BITS = 1     # F field length is 1 bit
        self.CHECK_BITS = 4 # Checksum length is 4 bits
        self.MAX_TDS = (1 << self.TDS_BITS) - 1  # Maximum value for TDS: 4095
        self.MAX_SN = (1 << self.SN_BITS) - 1    # Maximum value for SN: 63
    def create_package_head(self, TDS:int , SN:int , is_final:bool)->str:
        """
        Creates the package header.
        Args:
            TDS: Total data segments (only needed for the first packet).
            SN: Segment number (SN=0 for the first packet, SN!=0 for subsequent packets).
            is_final: Whether this is the last data segment.
        Returns:
            str: The package header as a binary string.
        """
        if(TDS > self.MAX_TDS or SN > self.MAX_SN):
            raise ValueError("TDS or SN is out of range")
        F = 1 if is_final else 0
        
        if SN == 0:
            # First packet: TDS+SN+F
            package_head = f"{TDS:0{self.TDS_BITS}b}{SN:0{self.SN_BITS}b}{F:0{self.F_BITS}b}"
        else:
            # Subsequent packets: SN+F
            package_head = f"{SN:0{self.SN_BITS}b}{F:0{self.F_BITS}b}"
        package_head += Math.calculate_crc4_binary(package_head)
        return package_head
    
    def parse_first_package(self, package_head: str) -> tuple[int, int ,bool, str]:
        """
        Parses the header of the first packet.
        Args:
            package_head: The package header as a binary string.
        Returns:
            tuple[int, int ,bool, str]: (TDS, SN, is_final, checkcode)
        """
        if len(package_head) <= self.TDS_BITS + self.SN_BITS + self.F_BITS:
            raise ValueError(f"The header length of the first packet should be {self.TDS_BITS + self.SN_BITS + self.F_BITS} bits.")
        
        # Parse TDS (first 12 bits)
        TDS = int(package_head[:self.TDS_BITS], 2)
        
        # Parse SN (middle 6 bits) - SN should be 0 for the first packet
        SN = int(package_head[self.TDS_BITS:self.TDS_BITS + self.SN_BITS], 2)
        if SN != 0:
            raise ValueError("SN must be 0 for the first packet.")
        
        # Parse F (1 bit)
        F = int(package_head[self.TDS_BITS + self.SN_BITS:self.TDS_BITS + self.SN_BITS+1], 2)
        is_final = bool(F)

        # Parse checksum (last 4 bits)
        checkcode = package_head[self.TDS_BITS + self.SN_BITS + self.F_BITS:]
        
        return TDS, SN, is_final, checkcode
    
    def parse_other_package(self, package_head: str) -> tuple[int, bool, str]:
        """
        Parses the header of subsequent packets (not the first one).
        Args:
            package_head: The package header as a binary string.
        Returns:
            tuple[int, bool, str]: (SN, is_final, checkcode)    
        """
        if len(package_head) <= self.SN_BITS + self.F_BITS:
            raise ValueError(f"The header length of subsequent packets should be {self.SN_BITS + self.F_BITS} bits.")
        
        # Parse SN (first 6 bits) - SN should not be 0 for subsequent packets
        SN = int(package_head[:self.SN_BITS], 2)
        if SN == 0:
            raise ValueError("SN cannot be 0 for subsequent packets.")
        
        # Parse F (1 bit)
        F = int(package_head[self.SN_BITS:self.SN_BITS+1], 2)
        is_final = bool(F)
        
        # Parse checksum (last 4 bits)
        checkcode = package_head[self.SN_BITS + self.F_BITS:]
        
        return SN, is_final, checkcode
    
    

    
