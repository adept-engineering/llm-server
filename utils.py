def _format_messages(messages):
    """Format messages into a single string for token counting"""
    formatted_text = ""
    
    # Handle different message formats
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "")
                
                # Handle content which could be a string or a list of content parts
                content = message.get("content", "")
                if isinstance(content, list):
                    # Extract text from content parts
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            formatted_text += f"{role}: {item['text']}\n"
                else:
                    # Direct string content
                    formatted_text += f"{role}: {content}\n"
    
    return formatted_text