#!/usr/bin/env python
import warnings
from crew import MultiAgentCustomerSupport

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")




def run():
    """
    Run the crew.
    """
    inputs = {
        'customer': 'DeepLearningAI',
        "person": "Andrew Ng",
        "inquiry": "I need help with setting up a Crew and kicking it off specifically how can I add memory to my crew? Can you provide guidance?"
    }
    
    try:
        MultiAgentCustomerSupport().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == "__main__":
    run()
