from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Part, Role
from uuid import uuid4
from .agent import ServerAgent

class ServerAgentExecutor(AgentExecutor):
    def __init__(self,stego_model_path,stego_algorithm,stego_key,decrypted_bits_path,session_id):
        self.agent = ServerAgent(stego_model_path,stego_algorithm,stego_key,decrypted_bits_path,session_id)

   
    async def execute(self,context: RequestContext,event_queue: EventQueue) -> None:
        user_id = "default_user"
        answer,timestamp = await self.agent.response_client_message(context, user_id=user_id)
        message = Message(
            kind='message',
            messageId=uuid4().hex,
            parts=[Part(kind='text', text=answer)],
            role=Role.agent,
            metadata={
                "sendTimestamp":timestamp
            }
        )
        await event_queue.enqueue_event(message)

    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception('cancel not supported')

