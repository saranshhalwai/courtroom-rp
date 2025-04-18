# Courtroom Simulation System

This project is a courtroom simulation system designed to model legal proceedings using AI agents. The system facilitates a mock trial environment where various roles such as Judge, Prosecutor, Defense, Plaintiff, Defendant, and Moderator interact to simulate a legal case.

## Features

- **AI-Powered Agents**: Each role in the trial is represented by an AI agent with specific responsibilities and behaviors.
- **Dynamic Trial Phases**: The trial progresses through phases such as `opening`, `arguments`, and `verdict`.
- **Moderator Control**: A moderator ensures the trial proceeds smoothly by deciding the next speaker and managing transitions between phases.
- **Customizable Prompts**: Each agent is initialized with a role-specific prompt to guide its behavior.
- **Transcript Recording**: The system maintains a detailed transcript of the trial for review.

## Components

### Agents

- **Judge**: Presides over the trial, ensures fairness, and delivers the final ruling.
- **Prosecution**: Represents the state, presenting evidence and arguments against the defendant.
- **Defense**: Advocates for the defendant, ensuring their rights are protected.
- **Plaintiff**: Brings the case against the defendant, presenting claims and evidence.
- **Defendant**: Responds to accusations, maintaining innocence and presenting a defense.
- **Moderator**: Manages the trial flow, deciding who speaks next and ensuring fairness.

### Key Classes

- `Agent`: Represents a generic participant in the trial.
- `Moderator`: Manages trial phases and speaker decisions.

## How It Works

1. **Initialization**: Agents are initialized with role-specific prompts and a chat model.
2. **Trial Execution**: The trial begins with the `opening` phase, progressing through `arguments` and ending with the `verdict`.
3. **Dynamic Interaction**: The moderator decides the next speaker based on the trial phase and transcript.
4. **Final Ruling**: The judge delivers a verdict based on the complete trial transcript.

## Prerequisites

- Python 3.8+
- Required Python libraries: `langchain`, `pandas`, `dotenv`

## Setup

1. Clone the repository:

```bash
git clone https://github.com/saranshhalwai/courtroom-rp.git
cd courtroom-rp
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add a `.env` file with necessary environment variables for the chat model.

<!-- 4. Prepare a `data.csv` file containing case descriptions. -->

## Usage

Run the simulation with:

```bash
python main.py
```

## Example

The system reads a case description from `data.csv` and simulates a trial. The transcript is printed to the console, showing interactions between agents.

## Future Enhancements

- Add support for dynamic agent creation.
- Improve decision-making logic for the moderator.
- Enhance the user interface for better visualization of the trial.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using the [LangChain](https://github.com/hwchase17/langchain) framework.
- Inspired by real-world courtroom procedures.
