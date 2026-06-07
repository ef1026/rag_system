from __future__ import annotations

from backend.schemas import ImageAssetPublic, QuizChoice, QuizQuestion
from backend.services import quiz_service


def _image(image_id: str = "img-1") -> ImageAssetPublic:
    return ImageAssetPublic(
        image_id=image_id,
        document_id="doc-1",
        document_name="Doc 1",
        url=f"/api/documents/doc-1/images/{image_id}",
        caption="Figure 1",
        page=2,
    )


def _question(source_ids: list[str]) -> QuizQuestion:
    return QuizQuestion(
        id="qq-1",
        prompt="Question",
        choices=[
            QuizChoice(id="A", text="A"),
            QuizChoice(id="B", text="B"),
        ],
        correct_choice_id="A",
        explanation="Explanation",
        source_message_ids=source_ids,
    )


def test_attach_related_images_to_questions_from_source_ids() -> None:
    image = _image()
    source = quiz_service.QuizSource(
        id="learn_msg_1",
        role="assistant",
        content="learning content",
        related_images=(image,),
    )

    questions = quiz_service._attach_related_images_to_questions(
        [_question(["learn_msg_1"])],
        [source],
    )

    assert questions[0].related_images[0].image_id == "img-1"
    assert questions[0].related_images[0].page == 2


def test_attach_related_images_accepts_raw_message_source_id() -> None:
    image = _image("img-2")
    source = quiz_service.QuizSource(
        id="learn_msg_2",
        role="assistant",
        content="learning content",
        related_images=(image,),
    )

    questions = quiz_service._attach_related_images_to_questions(
        [_question(["msg_2"])],
        [source],
    )

    assert questions[0].related_images[0].image_id == "img-2"


def test_message_ids_from_wrong_question_text_refs() -> None:
    message_ids = quiz_service._message_ids_from_source_refs(
        ["learn_msg_1"],
        ["evidence [msg_2] and [wrong_wrong_1]"],
    )

    assert message_ids == {"msg_1", "msg_2"}
