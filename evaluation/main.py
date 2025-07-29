import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoModelForCausalLM, AutoTokenizer
from algo.evaluation import parse_conversation

def parse_args():
    parser = argparse.ArgumentParser(description='AgentStego Evaluation')

    parser.add_argument('--evaluation_model','-em',
    default='/path/to/your/evaluation_model', 
    help='Select the path for the evaluation model')

    parser.add_argument('--evaluation_conversation','-ec',
    default='data/conversation/conversation_covert-session-uuid-44195c6d-d09e-4191-9bcb-d22a85b7d126.json',
    help='Select the path for the conversation to be evaluated')

    parser.add_argument('--evaluation_precision','-ep',
    default='half',
    help='Select the evaluation precision')

    parser.add_argument('--result_path','-rp',
    default='data/evaluation/',
    help='Select the path to save the evaluation results')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.evaluation_precision == 'half':
        model = AutoModelForCausalLM.from_pretrained(args.evaluation_model).half().cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.evaluation_model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.evaluation_model).cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.evaluation_model)

    parse_conversation(model,tokenizer,args.evaluation_conversation,args.result_path)