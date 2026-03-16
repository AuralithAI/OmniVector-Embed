import logging

import pytest

logger = logging.getLogger(__name__)


class TestAudioPreprocessing:
    def test_preprocess_audio_invalid_path(self):
        from omnivector.data.preprocessing import preprocess_audio

        with pytest.raises(ValueError):
            preprocess_audio("/nonexistent/audio.wav")

    def test_preprocess_audio_invalid_input(self):
        from omnivector.data.preprocessing import preprocess_audio

        with pytest.raises(ValueError):
            preprocess_audio(123)

    def test_detect_audio_file(self, sample_audio_path):
        from omnivector.data.preprocessing import detect_modality

        modality = detect_modality(sample_audio_path)
        assert modality == "audio"


class TestVideoPreprocessing:
    def test_preprocess_video_invalid_path(self):
        from omnivector.data.preprocessing import preprocess_video

        with pytest.raises(ValueError):
            preprocess_video("/nonexistent/video.mp4")

    def test_preprocess_video_invalid_input(self):
        from omnivector.data.preprocessing import preprocess_video

        with pytest.raises(ValueError):
            preprocess_video(123)

    def test_detect_video_file(self, sample_video_path):
        from omnivector.data.preprocessing import detect_modality

        modality = detect_modality(sample_video_path)
        assert modality == "video"


class TestCodePreprocessing:
    def test_preprocess_code_basic(self):
        from omnivector.data.preprocessing import preprocess_code

        code = "def hello():\n    pass"
        result = preprocess_code(code)
        assert result == code

    def test_preprocess_code_with_language(self):
        from omnivector.data.preprocessing import preprocess_code

        code = "print('hello')"
        result = preprocess_code(code, language="python")
        assert "Language: python" in result
        assert code in result

    def test_preprocess_code_truncation(self):
        from omnivector.data.preprocessing import preprocess_code

        code = "x" * 3000
        result = preprocess_code(code, max_length=1000)
        assert len(result) <= 1000

    def test_preprocess_code_empty_raises(self):
        from omnivector.data.preprocessing import preprocess_code

        with pytest.raises(ValueError):
            preprocess_code("")

    def test_preprocess_code_invalid_type(self):
        from omnivector.data.preprocessing import preprocess_code

        with pytest.raises(ValueError):
            preprocess_code(123)


class TestModalityDetection:
    def test_detect_text_modality(self):
        from omnivector.data.preprocessing import detect_modality

        assert detect_modality("This is text") == "text"

    def test_detect_code_modality(self):
        from omnivector.data.preprocessing import detect_modality

        assert detect_modality("def function():\n    pass") == "code"

    def test_detect_code_with_class(self):
        from omnivector.data.preprocessing import detect_modality

        assert detect_modality("class MyClass:\n    pass") == "code"

    def test_detect_code_with_triple_backticks(self):
        from omnivector.data.preprocessing import detect_modality

        assert detect_modality("```python\ncode\n```") == "code"

    def test_detect_modality_from_dict_video(self):
        from omnivector.data.preprocessing import detect_modality

        data = {"frames": []}
        assert detect_modality(data) == "video"

    def test_detect_modality_from_dict_audio(self):
        from omnivector.data.preprocessing import detect_modality

        data = {"audio": [], "mfcc": []}
        assert detect_modality(data) == "audio"


class TestDataLoaders:
    def test_msmarco_loader_instantiation(self):
        from omnivector.data.loaders.base import MSMARCOLoader

        loader = MSMARCOLoader()
        assert loader.name == "msmarco"

    def test_hotpotqa_loader_instantiation(self):
        from omnivector.data.loaders.base import HotpotQALoader

        loader = HotpotQALoader()
        assert loader.name == "hotpotqa"

    def test_beir_loader_instantiation(self):
        from omnivector.data.loaders.base import BEIRLoader

        loader = BEIRLoader()
        assert loader.name == "beir"

    def test_msmarco_loader_load(self):
        from omnivector.data.loaders.base import MSMARCOLoader

        loader = MSMARCOLoader()
        data = loader.load()
        assert isinstance(data, list)

    def test_hotpotqa_loader_load(self):
        from omnivector.data.loaders.base import HotpotQALoader

        loader = HotpotQALoader()
        data = loader.load()
        assert isinstance(data, list)

    def test_beir_loader_load(self):
        from omnivector.data.loaders.base import BEIRLoader

        loader = BEIRLoader()
        data = loader.load()
        assert isinstance(data, list)
