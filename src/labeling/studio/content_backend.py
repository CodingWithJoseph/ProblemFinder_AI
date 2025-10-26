from label_studio_ml.model import LabelStudioMLBase
from labeling.studio.models import ContentTypeModel


class ContentTypeBackend(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(ContentTypeBackend, self).__init__(**kwargs)
        self.model = ContentTypeModel()

    def predict(self, tasks, **kwargs):
        return self.model.predict(tasks)

    def fit(self, tasks, workdir=None, **kwargs):
        return {'status': '200 OK'}