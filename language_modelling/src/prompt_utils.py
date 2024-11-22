def create_system_prompt(country_name) -> str:
    """Creates the system prompt for financial classification with structured output."""
    system_prompt = (
        "You are from "+country_name+" .Can you answer this question from a Survey to the best of your knowledge.\n"
        "Ensure your output follows this exact format:\n"
        "prediction - <category> | reason - <reason>"
    )
    return system_prompt