import uuid
from django.db import models

class AudioFile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.FileField(upload_to='uploads/')
    created_at = models.DateTimeField(auto_now_add=True)

    # معلومات اختيارية هنملأها بعد الرفع
    sample_rate = models.IntegerField(null=True, blank=True)
    channels = models.IntegerField(null=True, blank=True)
    duration_sec = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.file.name}"
