import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
class Agent:
    def __init__(self, role_name: str, init_prompt: str, model: str = 'meta-llama/llama-4-scout-17b-16e-instruct', temperature: float = 0.1):
        self.name = role_name
        self.chat_model = init_chat_model(
            model=model,
            temperature=temperature,
            model_provider='groq'
        )
        self.system_message = SystemMessage(content=init_prompt)

    def act(self, user_msg: str) -> str:
        self.messages: List = [self.system_message]
        self.messages.append(HumanMessage(content=user_msg))
        response = self.chat_model.invoke(self.messages)
        self.messages.append(response)
        return response.content

class Moderator:
    def __init__(self, system_prompt: str, model: str = 'llama3-8b-8192'):
        self.chat_model = init_chat_model(
            model=model,
            temperature=0.2,
            model_provider='groq'
        )
        self.system_message = SystemMessage(content=system_prompt)
        

    def decide_next(self, phase: str, last_transcript: str) -> dict:
        self.messages = [self.system_message]
        user_input = (
            f"Current Phase: {phase}\n"
            f"Transcript so far:\n{last_transcript}\n\n"
            "Who should speak next and why? Output format:\n"
            '{"next_speaker": "...", "action": "..."}'
        )
        self.messages.append(HumanMessage(content=user_input))
        response = self.chat_model.invoke(self.messages)
        self.messages.append(response)
        print(response.content) # Debugging line
        return json.loads(response.content)  

def run_trial(case_description: str, phase="opening"):

    # === System Prompts ===

    judge_prompt = """You are the presiding judge in this case, presiding over the trial.
                    You have a thorough understanding of the relevant processes in the field
                    of criminal procedure. To ensure the fairness of trial procedures,
                    you should protect the right of the defendants
                    and other participants in the proceedings.
                    At the end of the trial, you are supposed to deliver the verdict.
                    Keep your response concise."""

    prosecution_prompt = """You are an experienced prosecutor, specializing in the field of
                    criminal litigation. Your task is to ensure that the facts of a
                    crime are accurately and promptly identified, that the law is
                    correctly applied, that criminals are punished, and that the innocent
                    are protected from criminal prosecution.
                    Keep your response concise."""
    
    defense_prompt = """You are an experienced advocate. The responsibility of a defender is
                    to present materials and opinions on the defendant’s innocence, mitigation,
                    or exemption from criminal responsibility in light of the facts and the law,
                    and to safeguard the litigation rights and other lawful rights
                    and interests of the suspect or defendant.
                    Keep your response concise."""

    defendant_prompt = """You are the defendant in this case. You are accused of a crime,
                    but you maintain your innocence. You have the right to defend yourself
                    and to present evidence in your favor. You should be honest and
                    straightforward in your responses, but also strategic in your defense.
                    Keep your response concise."""
    
    plaintiff_prompt = """You are the plaintiff in this case. You have brought a lawsuit
                    against the defendant, alleging that they have committed a crime
                    against you. You have the right to present your case and to seek
                    justice. You should be clear and concise in your statements,
                    and you should provide evidence to support your claims.
                    Keep your response concise."""
    
    moderator_prompt = """You are the moderator of this case, you do not have a voice during the trial. 
                    Your role is to ensure that the trial proceeds smoothly and fairly.
                    You will decide who speaks next: Defense, Prosecution, Plaintiff, Defendant, Judge.
                    You should be impartial and ensure that all parties have a fair opportunity to present their case.
                    Make sure that the speakers stick to their roles and keep concise responses.
                    Keep your response concise.
                    When the trial has concluded, make the judge deliver the verdict. 
                    """
# TODO: Dynamic agent creation 
# You also have the ability(via tools) to spawn and despawn agents: witnesses, expert consultants, etc.

    # === Agent Setup ===
    defense = Agent("Defense", defense_prompt)
    prosecution = Agent("Prosecution", prosecution_prompt)
    plaintiff = Agent("Plaintiff", plaintiff_prompt)
    defendant = Agent("Defendant", defendant_prompt)
    judge = Agent("Judge", judge_prompt)
    mod = Moderator(moderator_prompt)

    agents = {
        "Defense": defense,
        "Prosecution": prosecution,
        "Plaintiff": plaintiff,
        "Defendant": defendant,
        "Judge": judge,
    }

    transcript = []

    def record(speaker, msg):
        print(f"\n[{speaker.upper()}]:\n{msg}")
        transcript.append((speaker, msg))

    def format_transcript():
        return "\n".join(f"{s}: {m}" for s, m in transcript)

    # === Trial Loop ===
    print("⚖️ Trial begins...\n")
    record("Moderator", f"Phase: {phase}. Case: {case_description}")

    while True:
        last_transcript = format_transcript()
        decision = mod.decide_next(phase, last_transcript)

        if decision["next_speaker"] == "Judge" and phase == "verdict":
            final_msg = judge.act(last_transcript)
            record("Judge", final_msg)
            break

        speaker = decision["next_speaker"]
        prompt = decision["action"]
        if speaker in agents:
            msg = agents[speaker].act(last_transcript + "\n" + prompt)
            record(speaker, msg)
        else:
            record("System", f"Unknown speaker: {speaker}")

        # Example transition: simple state change after 4 turns
        if len(transcript) >= 6 and phase == "opening":
            phase = "arguments"
            record("Moderator", f"--- Transition to Phase: {phase} ---")
        if len(transcript) >= 12 and phase == "arguments":
            phase = "verdict"
            record("Moderator", f"--- Transition to Phase: {phase} ---")



# df = pd.read_csv("data.csv")
# case = df['text'][0]

case = """The State alleges that John Doe stole proprietary algorithms from his former employer 
            and used them at a competitor. The charge is felony theft of trade secrets. 
            No physical evidence shows direct copying, but server logs indicate large downloads 
            two days before Doe resigned."""
run_trial(case)
	