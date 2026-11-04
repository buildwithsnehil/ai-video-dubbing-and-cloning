from services.avatar_service import AvatarService

# Singleton instance of AvatarService used across request handlers
_avatar_service = AvatarService()


def get_avatar_service() -> AvatarService:
    return _avatar_service
