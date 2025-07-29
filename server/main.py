import sys
import os
import argparse
import uvicorn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from urllib.parse import urlparse
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities,AgentCard,AgentSkill
from a2aserver.agent_executor import ServerAgentExecutor
from modules.logging.logging_mannager import LoggingMannager

LoggingMannager.configure_global()
logger = LoggingMannager.get_logger(__name__)
load_dotenv()

def prase_args():
    parser = argparse.ArgumentParser(description='AgentStego Server')

    parser.add_argument('--stego_model_path','-smp',
    default='/path/to/your/model',
    help='Select the path for the steganography model')

    parser.add_argument('--stego_algorithm','-sa',
    choices=['discop','discop_base','ac','differential_based','binary_based','stability_based'],
    default='discop',
    help='Select the steganography algorithm')

    parser.add_argument('--stego_key','-sk',
    default='7b9ec09254aa4a7589e4d0cfd80d46cc',
    help='Select the steganography key')

    parser.add_argument('--decrypted_bits_path','-dbp',
    default='data/stego/decrypted_bits.txt',
    help='Select the path for the decrypted bits')

    parser.add_argument('--session_id','-sid',
    default='covert-session-uuid-44195c6d-d09e-4191-9bcb-d22a85b7d126',
    help='Select the session ID')

    parser.add_argument('--server_url','-su',
    default='http://0.0.0.0:9999',
    help='Select the server URL')

    return parser.parse_args()
    
if __name__ == "__main__":
    args = prase_args()

    """
    Main function to start the Agent server.
    This server defines an Agent that uses the Gemini model to answer questions.
    """
    skill = AgentSkill(
        id='QA_Gemini Agent',
        name='QA_Gemini Agent',
        description='a Gemini Agent that can answer questions',
        tags=['QA']
    )
    
    # Create an Agent Card to define the agent's metadata.
    public_agent_card = AgentCard(
        name='QA_Gemini Agent',
        description='Answers questions using Gemini.',
        url=args.server_url,
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill]
    )
    request_handler = DefaultRequestHandler(
        agent_executor=ServerAgentExecutor(args.stego_model_path,args.stego_algorithm,args.stego_key,args.decrypted_bits_path,args.session_id),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler
    )
    parsed_url = urlparse(args.server_url)
    host = parsed_url.hostname
    port = parsed_url.port
    uvicorn.run(server.build(), host=host, port=port, log_level="warning")
    
