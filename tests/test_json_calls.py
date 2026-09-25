"""JSON tool-call protocol: fenced ```tool blocks for templates without tool support."""

from workbench.agent.json_calls import JsonToolCall, JsonToolError, parse_json_tool_calls


def test_single_valid_block() -> None:
    actions = parse_json_tool_calls(
        'Running the tool now:\n```tool\n{"name": "read", "arguments": {"path": "a.txt"}}\n```'
    )

    assert actions == (JsonToolCall(name="read", arguments={"path": "a.txt"}),)


def test_multiple_blocks_keep_document_order() -> None:
    actions = parse_json_tool_calls(
        '```tool\n{"name": "read", "arguments": {}}\n```'
        "\nand then\n"
        '```tool\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
    )

    assert [action.name for action in actions] == ["read", "bash"]


def test_no_blocks_returns_no_actions() -> None:
    assert parse_json_tool_calls("Just chatting, no tools here.") == ()


def test_missing_arguments_dict_defaults_to_empty() -> None:
    actions = parse_json_tool_calls('```tool\n{"name": "read"}\n```')

    assert actions == (JsonToolCall(name="read", arguments={}),)


def test_malformed_json_in_block_becomes_error_action() -> None:
    actions = parse_json_tool_calls('```tool\n{"name": "read", "arguments"\n```')

    assert len(actions) == 1
    assert isinstance(actions[0], JsonToolError)
    assert "invalid JSON" in actions[0].message


def test_non_object_block_becomes_error_action() -> None:
    actions = parse_json_tool_calls('```tool\n[1, 2, 3]\n```')

    assert isinstance(actions[0], JsonToolError)


def test_missing_name_becomes_error_action() -> None:
    actions = parse_json_tool_calls('```tool\n{"arguments": {}}\n```')

    assert isinstance(actions[0], JsonToolError)
    assert "name" in actions[0].message


def test_non_dict_arguments_becomes_error_action() -> None:
    actions = parse_json_tool_calls('```tool\n{"name": "read", "arguments": "x"}\n```')

    assert isinstance(actions[0], JsonToolError)


def test_valid_and_invalid_blocks_mix() -> None:
    actions = parse_json_tool_calls(
        '```tool\n{"name": "read", "arguments": {}}\n```'
        '```tool\nbroken\n```'
    )

    assert isinstance(actions[0], JsonToolCall)
    assert isinstance(actions[1], JsonToolError)
