from hallucination_guard.generation import ResponseGenerator


def test_empty_question_is_rejected():
    class Dummy:
        pass

    generator = ResponseGenerator(
        tokenizer=Dummy(),
        model=Dummy(),
        device="cpu",
    )

    try:
        generator.generate("   ")
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for empty question"
