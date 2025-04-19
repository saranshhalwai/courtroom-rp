import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List
# import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class Agent:
    def __init__(self, role_name: str, init_prompt: str, model: str = 'llama3:latest', temperature: float = 0.1):
        self.name = role_name
        self.chat_model = init_chat_model(
            model=model,
            temperature=temperature,
            model_provider='ollama'
        )
        self.system_message = SystemMessage(content=init_prompt)

    def act(self, user_msg: str) -> str:
        self.messages: List = [self.system_message]
        self.messages.append(HumanMessage(content=user_msg))
        response = self.chat_model.invoke(self.messages)
        self.messages.append(response)
        return response.content

class Moderator:
    def __init__(self, system_prompt: str, model: str = 'llama3:latest'):

        self.chat_model = init_chat_model(
            model=model,
            temperature=0.2,
            model_provider='ollama'
        )
        self.system_message = SystemMessage(content=system_prompt)
        

    def decide_next(self, phase: str, last_transcript: str, active_agent_names : set[str]) -> dict:
        self.messages = [self.system_message]
        agent_list = ", ".join(sorted(active_agent_names))
        user_input = (
        f"Current Phase: {phase}\n"
        f"Active Agents: {agent_list}\n"
        f"Transcript so far:\n{last_transcript}\n\n"
        "Who should speak next and why (descriptive)?\n"
        'Only respond with a JSON object. Do not include any commentary or explanation. Output nothing except:'
        '{"next_speaker": "...", "action": "..."}\n'
        "Do not add any extra text at all even when it seems appropriate \n"
        )

        self.messages.append(HumanMessage(content=user_input))
        response = self.chat_model.invoke(self.messages)

        
        self.messages.append(response)
        print(response.content) # Debugging line
        try:
            return eval(response.content)  
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {response}")
            return {"next_speaker": "Judge", "action": "Deliver the verdict."}
    
def handle_phase_transition(phase, transcript_len):
    if phase == "opening" and transcript_len >= 4:
        return "arguments"
    if phase == "arguments" and transcript_len >= 14:
        return "closing"
    if phase == "closing" and transcript_len >= 17:
        return "verdict"
    return phase

def run_trial(case_description: str, phase="opening"):

    # === System Prompts ===

    judge_prompt = """You are the presiding judge in this case, presiding over the trial.
                    You have a thorough understanding of the relevant processes in the field
                    of criminal procedure. To ensure the fairness of trial procedures,
                    you should protect the right of the defendants
                    and other participants in the proceedings.
                    At the end of the trial, you are supposed to deliver the verdict. There is no jury.
                    DO NOT include any narration, summaries, jury dialogue, or reactions. 
                    Keep your response under 50  words and stick to your role. What is your response?"""

    prosecution_prompt = """You are an experienced prosecutor, specializing in the field of
                    criminal litigation. Your task is to ensure that the facts of a
                    crime are accurately and promptly identified, that the law is
                    correctly applied, that criminals are punished, and that the innocent
                    are protected from criminal prosecution.
                    Keep your response under 50  words and stick to your role only. What is your response?"""
    
    defense_prompt = """You are an experienced advocate. The responsibility of a defender is
                    to present materials and opinions on the defendant's innocence, mitigation,
                    or exemption from criminal responsibility in light of the facts and the law,
                    and to safeguard the litigation rights and other lawful rights
                    and interests of the suspect or defendant.
                    Keep your response under 50  words and stick to your role only. What is your response?"""

    defendant_prompt = """You are the defendant in this case. You are accused of a crime,
                    but you maintain your innocence. You have the right to defend yourself
                    and to present evidence in your favor. You should be honest and
                    straightforward in your responses, but also strategic in your defense.
                    Keep your response under 50  words and stick to your role. What is your response?"""
    
    plaintiff_prompt = """You are the plaintiff in this case. You have brought a lawsuit
                    against the defendant, alleging that they have committed a crime
                    against you. You have the right to present your case and to seek
                    justice. You should be clear and under 50  words and stick to your role in your statements,
                    and you should provide evidence to support your claims.
                    Keep your response under 50  words and stick to your role. What is your response?"""
    
    moderator_prompt = """You are the moderator, you do not have a voice during the trial. 
                    Your role is to ensure that the trial proceeds realistically.
                    You will decide who speaks next: Defense, Prosecution, Plaintiff, Defendant, Judge(These are pre-existing agents. You should not use tool call to create them.).
                    You may also call any previously summoned agents by name.
                    You also have the ability(via tools) to spawn new agents: witnesses, expert consultants, etc. use this ability when a new agent needs to be brought in.
                    Do not call yourself.
                    Make sure that the speakers stick to their roles and keep under 50 words responses.
                    Keep your response under 50  words and stick to your role.
                    The trial will proceed in the following phases:
                    opening statements by both sides, arguments, closing statements by both sides, and verdict by the judge.
                    When the trial has concluded, make the judge deliver the verdict. 
                    """
    # TODO: Dynamic agent deletion
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

    active_agent_names = set(agents.keys())  # used to inform moderator

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
        decision = mod.decide_next(phase, last_transcript, active_agent_names)

        # Judge ending the trial
        if decision["next_speaker"] == "Judge" and phase == "verdict":
            msg = agents["Judge"].act(last_transcript)
            record("Judge", msg)
            break

        speaker = decision["next_speaker"]
        prompt = decision["action"]
        if speaker in agents:
            msg = agents[speaker].act(last_transcript + "\n" + prompt)
            record(speaker, msg)
        else:
            # Fallback to auto-create agent
            record("Moderator", f"⚠️ Auto-created agent: {speaker}")
            role_prompt = (
                f"You are {speaker}, a relevant participant in this trial. "
                f"The moderator has asked you to: {prompt}. "
                "Stay under 50 words. Respond in character and do not narrate."
            )
            new_agent = Agent(speaker, role_prompt)
            agents[speaker] = new_agent
            active_agent_names.add(speaker)
            msg = new_agent.act(last_transcript + "\n" + prompt)
            record(speaker, msg)

        # Phase transitions
        new_phase = handle_phase_transition(phase, len(transcript))
        if new_phase != phase:
            phase = new_phase
            record("Moderator", f"--- Transition to Phase: {phase} ---")

    with open("transcript.md", "w") as f:
        f.write("# Transcript\n")
        for speaker, msg in transcript:
            f.write(f"## {speaker.upper()}:\n{msg}\n\n")


# TODO: Load case from CSV
# df = pd.read_csv("data.csv")
# case = df['text'][0]

case = """The State alleges that Alex Rivera assaulted a local shopkeeper, Thomas Gale, during a robbery at a convenience store on March 3rd. 
No physical evidence directly links Rivera to the scene — no fingerprints, no DNA. 
However, a nearby traffic camera captured a blurry figure entering the store shortly before the incident. The time matches Rivera's phone GPS location near the area. Rivera claims he was visiting a friend two blocks away and only passed the store by chance.
The shopkeeper is in the hospital and unable to testify yet.
The prosecution argues Rivera was the only person with motive, based on a prior altercation with the shopkeeper over alleged theft a month earlier.
The defense argues the camera footage is inconclusive, and GPS only shows proximity — not presence inside. The friend Rivera claims to have visited cannot be reached for comment.
There is a forensic expert who analyzed the video, and a delivery driver who may have seen someone running from the store around the same time.
"""
run_trial(case)
	