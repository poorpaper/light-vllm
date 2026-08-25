from light_vllm.serving import ChatMessage, HuggingFaceTextProcessor


class FakeTokenizer:
    eos_token_id = 99
    chat_template = "fake"

    def __len__(self):
        return 128

    def encode(self, prompt, *, add_special_tokens):
        assert not add_special_tokens
        return [len(prompt), 7]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        assert add_generation_prompt
        self.messages = messages
        return [3, 4, 5]

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        values = {
            (): "",
            (1,): "\ufffd",
            (1, 2): "你",
            (1, 2, 3): "你好",
            (3,): "好",
        }
        return values.get(tuple(token_ids), "".join(str(value) for value in token_ids))


def test_huggingface_processor_encodes_prompt_and_chat_template() -> None:
    tokenizer = FakeTokenizer()
    processor = HuggingFaceTextProcessor(tokenizer)

    assert processor.eos_token_id == 99
    assert processor.vocab_size == 128
    assert processor.encode_prompt("hello") == (5, 7)
    assert processor.encode_chat((ChatMessage("user", "hi"),)) == (3, 4, 5)
    assert tokenizer.messages == [{"role": "user", "content": "hi"}]


def test_incremental_decoder_holds_incomplete_unicode() -> None:
    decoder = HuggingFaceTextProcessor(FakeTokenizer()).new_decoder()

    assert decoder.push(1) == ""
    assert decoder.push(2) == "你"
    assert decoder.push(3) == "好"
    assert decoder.finish() == ""


def test_from_pretrained_is_local_and_disables_remote_code(tmp_path, monkeypatch) -> None:
    from transformers import AutoTokenizer

    calls: list[tuple[object, dict[str, object]]] = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return FakeTokenizer()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)

    processor = HuggingFaceTextProcessor.from_pretrained(tmp_path)

    assert processor.vocab_size == 128
    assert calls == [
        (
            tmp_path,
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_fast": True,
            },
        )
    ]
