"""Model context behavior for current and historical image inputs."""

from __future__ import annotations

from gewu_agent_runtime.context import ConversationContextBuilder
from gewu_agent_runtime.domain import Conversation, MessageKind, NewConversationMessage
from gewu_agent_runtime.identity import PrincipalRef
from gewu_agent_runtime.llm import ContentPart, ContentPartType, Message, MessageRole
from gewu_agent_runtime.persistence import InMemoryRuntimeStore


async def test_context_skips_current_run_input_and_appends_real_image_after_meta(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal))
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="inspect",
                payload={"attachments": [{"attachment_id": "img-1"}]},
                run_id="run-current",
            ),
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.META,
                content="<system-reminder>scene</system-reminder>",
                run_id="run-current",
            ),
        ),
    )
    current = Message.user_parts(
        (
            ContentPart.text_part("inspect"),
            ContentPart.encoded_image(
                mime_type="image/png",
                base64_data="YWJj",
                resource_id="img-1",
                name="evidence.png",
            ),
        )
    )

    messages = await ConversationContextBuilder(store).build(
        conversation.conversation_id,
        skip_input_run_id="run-current",
        current_input_message=current,
    )

    assert len(messages) == 1
    assert messages[0].content.startswith("<system-reminder>scene</system-reminder>")
    assert [part.part_type for part in messages[0].content_parts] == [
        ContentPartType.TEXT,
        ContentPartType.TEXT,
        ContentPartType.IMAGE,
    ]
    assert messages[0].content_parts[-1].base64_data == "YWJj"


async def test_context_uses_public_placeholders_for_historical_attachments(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal))
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="please view image",
                payload={
                    "attachments": [
                        {
                            "attachment_id": "img-1",
                            "original_name": "diagram.png",
                            "mime_type": "image/png",
                            "size_bytes": 3,
                        }
                    ]
                },
            ),
        ),
    )

    messages = await ConversationContextBuilder(store).build(conversation.conversation_id)

    assert messages == [
        Message.user("please view image\n\n[Image: diagram.png, image/png, attachment_id=img-1]")
    ]


def test_context_normalization_preserves_parts_when_merging_adjacent_users() -> None:
    image = ContentPart.encoded_image(
        mime_type="image/png",
        base64_data="YWJj",
        resource_id="img-1",
    )

    normalized = ConversationContextBuilder.normalize(
        (Message.user("reminder"), Message.user_parts((ContentPart.text_part("input"), image)))
    )

    assert len(normalized) == 1
    assert [part.part_type for part in normalized[0].content_parts] == [
        ContentPartType.TEXT,
        ContentPartType.TEXT,
        ContentPartType.IMAGE,
    ]
    assert normalized[0].content == "reminder\ninput"
