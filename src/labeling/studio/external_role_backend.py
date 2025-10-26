from label_studio_ml.model import LabelStudioMLBase
from labeling.studio.models import ExternalRoleModel


class ExternalRoleBackend(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(ExternalRoleBackend, self).__init__(**kwargs)
        self.model = ExternalRoleModel()

    def predict(self, tasks, **kwargs):
        return self.model.predict(tasks)

    def fit(self, tasks, workdir=None, **kwargs):
        return {'status': '200 OK'}