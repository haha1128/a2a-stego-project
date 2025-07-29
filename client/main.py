import sys
import os
import argparse
import asyncio
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from a2aclient.client import Client
from modules.logging.logging_mannager import LoggingMannager


LoggingMannager.configure_global()
logger = LoggingMannager.get_logger(__name__)
load_dotenv()

def prase_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AgentStego Client')

    parser.add_argument('--stego_model_path','-smp',
    default='/path/to/your/model',
    help='Select the path for the steganography model')

    parser.add_argument('--stego_algorithm','-sa',
    choices=['discop','discop_base','ac','differential_based','binary_based','stability_based'],
    default='discop',
    help='Select the steganography algorithm')

    parser.add_argument('--question_path','-qp',
    default='data/question/general.txt',
    help='Select the path for the handshake prompt')

    parser.add_argument('--question_index','-qi',
    type=int,default=0,help='Select the handshake prompt number')

    parser.add_argument('--stego_key','-sk',
    default='7b9ec09254aa4a7589e4d0cfd80d46cc',
    help='Select the steganography key')

    parser.add_argument('--session_id','-sid',
    default='covert-session-uuid-44195c6d-d09e-4191-9bcb-d22a85b7d126',
    help='Select the session ID')

    parser.add_argument('--secret_bit_path','-sbp',
    default='data/stego/secret_bits_512.txt',
    help='Select the path for the secret bits')

    parser.add_argument('--server_url','-su',
    default='http://localhost:9999',
    help='Select the server URL')

    return parser.parse_args()
        

async def main():
    args = prase_args()
    client = Client(args.stego_model_path,
                    args.stego_algorithm,
                    args.question_path,
                    args.question_index,
                    args.stego_key,
                    args.secret_bit_path,
                    args.server_url,
                    args.session_id)
    await client.start_stego_chat()

if __name__ == '__main__':
    asyncio.run(main())
    

    
    
