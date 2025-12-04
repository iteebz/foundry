"""Tests for conversation formatting."""

from foundry.data.conversation import (
    Conversation,
    Message,
    format_alpaca,
    format_chatml,
    format_llama3,
    pack_conversations,
)


def test_message_creation():
    """Message dataclass works."""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_conversation_creation():
    """Conversation dataclass works."""
    conv = Conversation(
        messages=[
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello"),
        ]
    )
    assert len(conv.messages) == 2


def test_sharegpt_format():
    """ShareGPT format parses correctly."""
    data = {
        "conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "4"},
        ]
    }
    conv = Conversation.from_sharegpt(data)
    assert len(conv.messages) == 2
    assert conv.messages[0].role == "human"
    assert conv.messages[0].content == "What is 2+2?"
    assert conv.messages[1].role == "gpt"
    assert conv.messages[1].content == "4"


def test_openai_format():
    """OpenAI format parses correctly."""
    data = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    conv = Conversation.from_openai(data)
    assert len(conv.messages) == 2
    assert conv.messages[0].role == "user"
    assert conv.messages[1].role == "assistant"


def test_format_chatml():
    """ChatML formatting works."""
    conv = Conversation(
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello"),
        ]
    )
    formatted = format_chatml(conv)

    assert "<|im_start|>system" in formatted
    assert "You are helpful.<|im_end|>" in formatted
    assert "<|im_start|>user" in formatted
    assert "<|im_start|>assistant" in formatted


def test_format_chatml_role_mapping():
    """ChatML maps human/gpt to user/assistant."""
    conv = Conversation(
        messages=[
            Message(role="human", content="Hi"),
            Message(role="gpt", content="Hello"),
        ]
    )
    formatted = format_chatml(conv)

    assert "<|im_start|>user" in formatted
    assert "<|im_start|>assistant" in formatted
    assert "human" not in formatted
    assert "gpt" not in formatted


def test_format_llama3():
    """Llama3 formatting works."""
    conv = Conversation(
        messages=[
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello"),
        ]
    )
    formatted = format_llama3(conv)

    assert "<|begin_of_text|>" in formatted
    assert "<|start_header_id|>user<|end_header_id|>" in formatted
    assert "<|start_header_id|>assistant<|end_header_id|>" in formatted
    assert "<|eot_id|>" in formatted


def test_format_alpaca():
    """Alpaca formatting works."""
    conv = Conversation(
        messages=[
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="4"),
        ]
    )
    formatted = format_alpaca(conv)

    assert "### Instruction:" in formatted
    assert "What is 2+2?" in formatted
    assert "### Response:" in formatted
    assert "4" in formatted


def test_pack_conversations():
    """Conversation packing works."""
    convs = [
        Conversation(messages=[Message(role="user", content="Hi")]),
        Conversation(messages=[Message(role="user", content="Hello")]),
        Conversation(messages=[Message(role="user", content="Hey")]),
    ]

    packed = pack_conversations(convs, max_length=100)

    assert len(packed) >= 1
    assert all(isinstance(p, str) for p in packed)


def test_pack_conversations_respects_max_length():
    """Packing respects max_length constraint."""
    long_content = "x" * 1000
    convs = [
        Conversation(messages=[Message(role="user", content=long_content)]),
        Conversation(messages=[Message(role="user", content=long_content)]),
    ]

    packed = pack_conversations(convs, max_length=500)

    assert len(packed) == 2


if __name__ == "__main__":
    test_message_creation()
    test_conversation_creation()
    test_sharegpt_format()
    test_openai_format()
    test_format_chatml()
    test_format_chatml_role_mapping()
    test_format_llama3()
    test_format_alpaca()
    test_pack_conversations()
    test_pack_conversations_respects_max_length()
    print("\n✓ All conversation tests passed")
