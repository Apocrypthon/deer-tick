from src.agents.middlewares.memory_middleware import _filter_messages_for_memory

class MockMessage:
    def __init__(self, type: str, content: str = "", tool_calls: list | None = None):
        self.type = type
        self.content = content
        self.tool_calls = tool_calls

def test_filter_messages_basic():
    # Test normal conversation
    messages = [
        MockMessage("human", "Hello!"),
        MockMessage("ai", "Hi there!"),
        MockMessage("human", "How are you?"),
        MockMessage("ai", "I'm good, thanks!")
    ]
    filtered = _filter_messages_for_memory(messages)
    assert len(filtered) == 4
    assert filtered[0].content == "Hello!"
    assert filtered[1].content == "Hi there!"

def test_filter_messages_tool_calls():
    # Test filtering out intermediate tool calls
    messages = [
        MockMessage("human", "What's the weather?"),
        MockMessage("ai", "", tool_calls=[{"name": "get_weather"}]),
        MockMessage("tool", "Sunny and 75F"),
        MockMessage("ai", "It is sunny and 75F today.")
    ]
    filtered = _filter_messages_for_memory(messages)
    assert len(filtered) == 2
    assert filtered[0].type == "human"
    assert filtered[1].type == "ai"
    assert filtered[1].content == "It is sunny and 75F today."

def test_filter_messages_upload_block_partial():
    # Test human message with both upload block and user question
    messages = [
        MockMessage("human", "<uploaded_files>file1.txt</uploaded_files>\nCan you read this?"),
        MockMessage("ai", "Yes, I can read it.")
    ]
    filtered = _filter_messages_for_memory(messages)
    assert len(filtered) == 2
    assert filtered[0].type == "human"
    assert filtered[0].content == "Can you read this?"
    assert filtered[1].type == "ai"

def test_filter_messages_upload_block_full():
    # Test human message with ONLY an upload block
    messages = [
        MockMessage("human", "<uploaded_files>file1.txt</uploaded_files>\n"),
        MockMessage("ai", "I have received the file.")
    ]
    filtered = _filter_messages_for_memory(messages)
    # Both the upload-only message and its paired AI response should be filtered out
    assert len(filtered) == 0

def test_filter_messages_upload_block_multiple():
    # Test complex conversation with uploads and normal messages
    messages = [
        MockMessage("human", "<uploaded_files>file1.txt</uploaded_files>\n"),
        MockMessage("ai", "File received."),
        MockMessage("human", "<uploaded_files>file2.txt</uploaded_files>\nWhat is in the second file?"),
        MockMessage("ai", "It contains some text.")
    ]
    filtered = _filter_messages_for_memory(messages)
    assert len(filtered) == 2
    assert filtered[0].content == "What is in the second file?"
    assert filtered[1].content == "It contains some text."

def test_filter_messages_human_list_content():
    # Test when human content is a list of dicts
    messages = [
        MockMessage("human", [{"text": "Hello "}, {"text": "world!"}]),
        MockMessage("ai", "Hi!")
    ]
    filtered = _filter_messages_for_memory(messages)
    assert len(filtered) == 2
    assert filtered[0].content == "Hello  world!"
    assert filtered[1].content == "Hi!"
