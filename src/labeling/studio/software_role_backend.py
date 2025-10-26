from label_studio_ml.model import LabelStudioMLBase
from labeling.studio.models import SoftwareRoleModel


class SoftwareRoleBackend(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(SoftwareRoleBackend, self).__init__(**kwargs)
        self.model = SoftwareRoleModel()

    def predict(self, tasks, **kwargs):
        return self.model.predict(tasks)

    def fit(self, tasks, workdir=None, **kwargs):
        return {'status': '200 OK'}