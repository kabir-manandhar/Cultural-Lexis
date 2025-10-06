import spacy

nlp = spacy.load("en_core_web_sm")  # Load Spacy NLP model

def extract_keywords(question: str, top_n=5, include_adverbs=False) -> list:
    """
    Extracts culturally relevant keywords from the question.
    
    Args:
        question (str): The input question text.
        top_n (int): Number of keywords to extract.
        include_adverbs (bool): If True, also extracts adverbs.
    
    Returns:
        list: A list of extracted keywords.
    """
    doc = nlp(question)
    
    # Extract NOUN, PROPN, and ADJ (culturally significant words)
    pos_tags = {"NOUN", "PROPN", "ADJ"}
    
    if include_adverbs:
        pos_tags.add("ADV")  # Optionally include adverbs (low priority)
    
    keywords = [token.text.lower() for token in doc if token.pos_ in pos_tags]
    
    # Return the top N unique keywords
    return list(set(keywords))[:top_n]
