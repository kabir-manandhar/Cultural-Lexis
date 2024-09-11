def create_system_prompt() -> str:
    """Creates the system prompt for financial classification with structured output."""
    system_prompt = (
        "Can you answer this question from a Survey to the best of your knowledge."
    )
    return system_prompt