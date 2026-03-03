# test_import.py
import sys

print("=" * 60)
print("Import 테스트")
print("=" * 60)

# 1. Huggingface Hub 버전 확인
try:
    import huggingface_hub

    print(f"✅ huggingface-hub 버전: {huggingface_hub.__version__}")

    # cached_download 함수 확인
    from huggingface_hub import cached_download

    print(f"✅ cached_download 함수 사용 가능")
except ImportError as e:
    print(f"❌ Huggingface Hub 오류: {e}")
    sys.exit(1)

# 2. Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer

    print(f"✅ Sentence Transformers import 성공")
except ImportError as e:
    print(f"❌ Sentence Transformers 오류: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("모델 로딩 테스트")
print("=" * 60)

try:
    print("모델 다운로드 중... (시간 걸릴 수 있음)")
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    print(f"✅ 모델 로드 완료")

    # 임베딩 테스트
    sentences = ["안녕하세요", "테스트 문장입니다"]
    embeddings = model.encode(sentences)
    print(f"✅ 임베딩 생성 성공")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Sample (first 5): {embeddings[0][:5]}")

except Exception as e:
    print(f"❌ 모델 로딩 오류: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 모든 테스트 통과!")
print("=" * 60)
