from app.core.errors import AppError
from app.core.security import BCRYPT_MAX_BYTES, hash_password, verify_password


def test_hash_and_verify_password_round_trip() -> None:
    password = "password123"

    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrong-password", hashed) is False


def test_hash_password_rejects_inputs_over_bcrypt_limit() -> None:
    too_long_password = "a" * (BCRYPT_MAX_BYTES + 1)

    try:
        hash_password(too_long_password)
    except AppError as exc:
        assert exc.message_key == "auth.password_too_long"
        assert exc.status_code == 422
        assert exc.details == {"max_bytes": BCRYPT_MAX_BYTES}
    else:
        raise AssertionError("Expected AppError for overlong password")
