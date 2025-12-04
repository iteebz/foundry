"""Conversation formatting for chat data."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    """Single message in a conversation."""

    role: str
    content: str


@dataclass
class Conversation:
    """Multi-turn conversation."""

    messages: list[Message]

    @classmethod
    def from_sharegpt(cls, data: dict[str, Any]) -> "Conversation":
        """Parse ShareGPT format conversation."""
        messages = []
        for msg in data.get("conversations", []):
            messages.append(Message(role=msg.get("from", "unknown"), content=msg.get("value", "")))
        return cls(messages=messages)

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> "Conversation":
        """Parse OpenAI format conversation."""
        messages = []
        for msg in data.get("messages", []):
            messages.append(Message(role=msg.get("role", "user"), content=msg.get("content", "")))
        return cls(messages=messages)


def format_chatml(conversation: Conversation, tokenizer=None) -> str:
    """Format conversation in ChatML format.

    ChatML format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>
    """
    formatted = []

    for msg in conversation.messages:
        role = msg.role
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"

        formatted.append(f"<|im_start|>{role}\n{msg.content}<|im_end|>")

    return "\n".join(formatted)


def format_llama3(conversation: Conversation) -> str:
    """Format conversation in Llama3 format.

    Llama3 format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        Hi there!<|eot_id|>
    """
    formatted = ["<|begin_of_text|>"]

    for msg in conversation.messages:
        role = msg.role
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"

        formatted.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{msg.content}<|eot_id|>")

    return "".join(formatted)


def format_alpaca(conversation: Conversation) -> str:
    """Format conversation in Alpaca format.

    Alpaca format (instruction tuning):
        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Response:
        {response}
    """
    if len(conversation.messages) < 2:
        return ""

    instruction = conversation.messages[0].content
    response = conversation.messages[1].content if len(conversation.messages) > 1 else ""

    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{response}"
    )


def pack_conversations(
    conversations: list[Conversation],
    max_length: int = 2048,
    format_fn=format_chatml,
    separator: str = "\n\n",
) -> list[str]:
    """Pack multiple conversations into fixed-length sequences.

    Args:
        conversations: List of conversations to pack
        max_length: Maximum sequence length (in characters, approximate)
        format_fn: Formatting function (format_chatml, format_llama3, etc)
        separator: Separator between packed conversations

    Returns:
        List of packed conversation strings
    """
    packed = []
    current = []
    current_len = 0

    for conv in conversations:
        formatted = format_fn(conv)
        conv_len = len(formatted)

        if current_len + conv_len + len(separator) > max_length:
            if current:
                packed.append(separator.join(current))
            current = [formatted]
            current_len = conv_len
        else:
            current.append(formatted)
            current_len += conv_len + len(separator)

    if current:
        packed.append(separator.join(current))

    return packed


if __name__ == "__main__":
    conv = Conversation(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="4"),
        ]
    )

    print("ChatML:")
    print(format_chatml(conv))
    print("\nLlama3:")
    print(format_llama3(conv))
    print("\nAlpaca:")
    print(format_alpaca(conv))
